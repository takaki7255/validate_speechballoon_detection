#!/usr/bin/env python3
"""
evaluate_track_a.py
===================

Track A: 画素レベル評価（セマンティックセグメンテーション）

指標:
- IoU (Jaccard)
- Dice (F1 of pixels)
- Pixel Precision / Recall
- Boundary F-score (BF-score)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import csv
from scipy.ndimage import distance_transform_edt, binary_erosion


class TrackAEvaluator:
    """画素レベル評価クラス"""
    
    def __init__(self, boundary_radius: int = 2):
        """
        Parameters:
            boundary_radius: Boundary F-scoreの境界幅（ピクセル）
        """
        self.boundary_radius = boundary_radius
        self.results = []
    
    def evaluate_image(self, pred_mask: np.ndarray, gt_mask: np.ndarray, 
                      image_id: str) -> Dict:
        """
        1枚の画像に対する評価
        
        Parameters:
            pred_mask: 予測マスク (H, W), 0 or 255
            gt_mask: GT マスク (H, W), 0 or 255
            image_id: 画像ID
            
        Returns:
            結果辞書
        """
        # 二値化
        pred_binary = (pred_mask > 127).astype(np.uint8)
        gt_binary = (gt_mask > 127).astype(np.uint8)
        
        # 基本メトリクス
        intersection = np.sum(pred_binary & gt_binary)
        union = np.sum(pred_binary | gt_binary)
        pred_sum = np.sum(pred_binary)
        gt_sum = np.sum(gt_binary)
        
        # IoU
        iou = intersection / (union + 1e-7)
        
        # Dice
        dice = (2 * intersection) / (pred_sum + gt_sum + 1e-7)
        
        # Pixel Precision / Recall
        precision = intersection / (pred_sum + 1e-7)
        recall = intersection / (gt_sum + 1e-7)
        
        # F1
        f1 = (2 * precision * recall) / (precision + recall + 1e-7)
        
        # Boundary F-score
        bf_score, bf_precision, bf_recall = self._compute_boundary_fscore(
            pred_binary, gt_binary
        )
        
        result = {
            'image_id': image_id,
            'iou': float(iou),
            'dice': float(dice),
            'pixel_precision': float(precision),
            'pixel_recall': float(recall),
            'pixel_f1': float(f1),
            'boundary_fscore': float(bf_score),
            'boundary_precision': float(bf_precision),
            'boundary_recall': float(bf_recall),
            'pred_pixels': int(pred_sum),
            'gt_pixels': int(gt_sum),
            'intersection_pixels': int(intersection)
        }
        
        self.results.append(result)
        return result
    
    def _compute_boundary_fscore(self, pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float]:
        """
        Boundary F-score を計算
        
        境界を一定半径で膨張させ、相互のヒット率からF-scoreを計算
        """
        # 境界抽出（エッジ検出）
        pred_boundary = self._extract_boundary(pred)
        gt_boundary = self._extract_boundary(gt)
        
        if np.sum(pred_boundary) == 0 and np.sum(gt_boundary) == 0:
            return 1.0, 1.0, 1.0
        if np.sum(pred_boundary) == 0 or np.sum(gt_boundary) == 0:
            return 0.0, 0.0, 0.0
        
        # 境界を膨張
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (2*self.boundary_radius+1, 2*self.boundary_radius+1))
        pred_boundary_dilated = cv2.dilate(pred_boundary, kernel)
        gt_boundary_dilated = cv2.dilate(gt_boundary, kernel)
        
        # Precision: pred境界点がGT膨張境界に含まれる割合
        pred_hits = np.sum(pred_boundary & gt_boundary_dilated)
        boundary_precision = pred_hits / (np.sum(pred_boundary) + 1e-7)
        
        # Recall: GT境界点がpred膨張境界に含まれる割合
        gt_hits = np.sum(gt_boundary & pred_boundary_dilated)
        boundary_recall = gt_hits / (np.sum(gt_boundary) + 1e-7)
        
        # F-score
        bf_score = (2 * boundary_precision * boundary_recall) / \
                   (boundary_precision + boundary_recall + 1e-7)
        
        return bf_score, boundary_precision, boundary_recall
    
    def _extract_boundary(self, mask: np.ndarray) -> np.ndarray:
        """マスクの境界を抽出"""
        # エロージョンして差分を取る
        eroded = binary_erosion(mask, structure=np.ones((3, 3)))
        boundary = mask.astype(np.uint8) - eroded.astype(np.uint8)
        return boundary
    
    def compute_statistics(self) -> Dict:
        """全画像の統計を計算"""
        if len(self.results) == 0:
            return {}
        
        metrics = ['iou', 'dice', 'pixel_precision', 'pixel_recall', 'pixel_f1',
                  'boundary_fscore', 'boundary_precision', 'boundary_recall']
        
        stats = {}
        for metric in metrics:
            values = [r[metric] for r in self.results]
            stats[f'{metric}_mean'] = float(np.mean(values))
            stats[f'{metric}_std'] = float(np.std(values))
            stats[f'{metric}_median'] = float(np.median(values))
            stats[f'{metric}_min'] = float(np.min(values))
            stats[f'{metric}_max'] = float(np.max(values))
        
        # 総計
        total_pred_pixels = sum(r['pred_pixels'] for r in self.results)
        total_gt_pixels = sum(r['gt_pixels'] for r in self.results)
        total_intersection = sum(r['intersection_pixels'] for r in self.results)
        
        stats['total_iou'] = total_intersection / (total_pred_pixels + total_gt_pixels - total_intersection + 1e-7)
        stats['total_dice'] = (2 * total_intersection) / (total_pred_pixels + total_gt_pixels + 1e-7)
        stats['num_images'] = len(self.results)
        
        return stats
    
    def save_results(self, output_dir: Path):
        """結果をCSVとJSONで保存"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 画像ごとの結果をCSV保存
        csv_path = output_dir / "track_a_per_image.csv"
        if len(self.results) > 0:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
            print(f"Saved per-image results: {csv_path}")
        
        # 統計をJSON保存
        stats = self.compute_statistics()
        json_path = output_dir / "track_a_statistics.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Saved statistics: {json_path}")
        
        # サマリーを表示
        print("\n" + "="*60)
        print("Track A: Pixel-Level Evaluation Summary")
        print("="*60)
        print(f"Number of images: {stats.get('num_images', 0)}")
        print(f"\nMean IoU: {stats.get('iou_mean', 0):.4f} ± {stats.get('iou_std', 0):.4f}")
        print(f"Mean Dice: {stats.get('dice_mean', 0):.4f} ± {stats.get('dice_std', 0):.4f}")
        print(f"Mean Pixel F1: {stats.get('pixel_f1_mean', 0):.4f} ± {stats.get('pixel_f1_std', 0):.4f}")
        print(f"Mean Boundary F-score: {stats.get('boundary_fscore_mean', 0):.4f} ± {stats.get('boundary_fscore_std', 0):.4f}")
        print(f"\nTotal IoU: {stats.get('total_iou', 0):.4f}")
        print(f"Total Dice: {stats.get('total_dice', 0):.4f}")
        print("="*60 + "\n")


def evaluate_directory(pred_dir: Path, gt_dir: Path, output_dir: Path, 
                       file_extension: str = 'png'):
    """
    ディレクトリ内の全画像を評価
    
    Parameters:
        pred_dir: 予測マスクディレクトリ
        gt_dir: GTマスクディレクトリ
        output_dir: 結果出力ディレクトリ
        file_extension: ファイル拡張子
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    
    # 予測ファイル一覧
    pred_files = sorted(pred_dir.glob(f"*.{file_extension}"))
    
    if len(pred_files) == 0:
        print(f"No prediction files found in {pred_dir}")
        return
    
    evaluator = TrackAEvaluator(boundary_radius=2)
    
    print(f"Evaluating {len(pred_files)} images...")
    
    for pred_path in pred_files:
        image_id = pred_path.stem
        gt_path = gt_dir / pred_path.name
        
        if not gt_path.exists():
            print(f"Warning: GT not found for {image_id}, skipping...")
            continue
        
        # 読み込み
        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        
        if pred_mask is None or gt_mask is None:
            print(f"Warning: Failed to load {image_id}, skipping...")
            continue
        
        # 評価
        evaluator.evaluate_image(pred_mask, gt_mask, image_id)
    
    # 結果保存
    evaluator.save_results(output_dir)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python evaluate_track_a.py <pred_dir> <gt_dir> <output_dir>")
        print("\nEvaluates pixel-level metrics between prediction and GT masks.")
        print("\nExample:")
        print("  python evaluate_track_a.py ./predictions/masks ./gt_masks ./results")
        sys.exit(1)
    
    pred_dir = sys.argv[1]
    gt_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    evaluate_directory(pred_dir, gt_dir, output_dir)
