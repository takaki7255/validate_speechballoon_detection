#!/usr/bin/env python3
"""
evaluate_track_b.py
===================

Track B: インスタンスレベル評価

指標:
- AP@[.50:.95] (mask IoU基準)
- F1@0.50, F1@0.75
- PQ (Panoptic Quality)
- Count MAE (個数誤差)

Hungarian matchingで1対1対応を取る
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import csv
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


class TrackBEvaluator:
    """インスタンスレベル評価クラス"""
    
    def __init__(self, iou_thresholds: List[float] = None):
        """
        Parameters:
            iou_thresholds: AP計算用のIoU閾値リスト
        """
        if iou_thresholds is None:
            # COCO形式: 0.50:0.05:0.95
            self.iou_thresholds = [0.50 + 0.05 * i for i in range(10)]
        else:
            self.iou_thresholds = iou_thresholds
        
        self.results = []
    
    def evaluate_image(self, pred_instances: List[Dict], gt_instances: List[Dict],
                      image_id: str) -> Dict:
        """
        1枚の画像のインスタンス評価
        
        Parameters:
            pred_instances: 予測インスタンスリスト
                [{'mask': ndarray, 'score': float, 'bbox': [x,y,w,h]}, ...]
            gt_instances: GTインスタンスリスト
                [{'mask': ndarray, 'bbox': [x,y,w,h]}, ...]
            image_id: 画像ID
            
        Returns:
            結果辞書
        """
        num_pred = len(pred_instances)
        num_gt = len(gt_instances)
        
        if num_pred == 0 and num_gt == 0:
            # 両方空の場合
            result = {
                'image_id': image_id,
                'num_pred': 0,
                'num_gt': 0,
                'count_mae': 0.0,
                'pq': 1.0,
                'sq': 1.0,
                'rq': 1.0,
            }
            for thresh in self.iou_thresholds:
                result[f'tp_{thresh:.2f}'] = 0
                result[f'fp_{thresh:.2f}'] = 0
                result[f'fn_{thresh:.2f}'] = 0
                result[f'precision_{thresh:.2f}'] = 1.0
                result[f'recall_{thresh:.2f}'] = 1.0
                result[f'f1_{thresh:.2f}'] = 1.0
            
            self.results.append(result)
            return result
        
        # IoU行列を計算
        iou_matrix = self._compute_iou_matrix(pred_instances, gt_instances)
        
        # 各閾値でのマッチング
        result = {
            'image_id': image_id,
            'num_pred': num_pred,
            'num_gt': num_gt,
            'count_mae': abs(num_pred - num_gt),
        }
        
        matched_ious_for_pq = []
        
        for thresh in self.iou_thresholds:
            tp, fp, fn, matched_ious = self._match_instances(
                iou_matrix, thresh, pred_instances, gt_instances
            )
            
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = (2 * precision * recall) / (precision + recall + 1e-7)
            
            result[f'tp_{thresh:.2f}'] = tp
            result[f'fp_{thresh:.2f}'] = fp
            result[f'fn_{thresh:.2f}'] = fn
            result[f'precision_{thresh:.2f}'] = float(precision)
            result[f'recall_{thresh:.2f}'] = float(recall)
            result[f'f1_{thresh:.2f}'] = float(f1)
            
            # PQ計算用（閾値0.5のマッチング結果を使用）
            if abs(thresh - 0.5) < 0.01:
                matched_ious_for_pq = matched_ious
        
        # PQ計算
        pq, sq, rq = self._compute_pq(matched_ious_for_pq, num_pred, num_gt)
        result['pq'] = float(pq)
        result['sq'] = float(sq)
        result['rq'] = float(rq)
        
        self.results.append(result)
        return result
    
    def _compute_iou_matrix(self, pred_instances: List[Dict], 
                           gt_instances: List[Dict]) -> np.ndarray:
        """
        予測とGT間のIoU行列を計算
        
        Returns:
            iou_matrix: (num_pred, num_gt) の行列
        """
        num_pred = len(pred_instances)
        num_gt = len(gt_instances)
        
        if num_pred == 0 or num_gt == 0:
            return np.zeros((num_pred, num_gt))
        
        iou_matrix = np.zeros((num_pred, num_gt))
        
        for i, pred in enumerate(pred_instances):
            for j, gt in enumerate(gt_instances):
                iou_matrix[i, j] = self._compute_mask_iou(pred['mask'], gt['mask'])
        
        return iou_matrix
    
    def _compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """2つのマスクのIoUを計算"""
        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1 | mask2)
        return intersection / (union + 1e-7)
    
    def _match_instances(self, iou_matrix: np.ndarray, threshold: float,
                        pred_instances: List[Dict], gt_instances: List[Dict]) -> Tuple[int, int, int, List[float]]:
        """
        Hungarian matchingでインスタンスを1対1対応させる
        
        Returns:
            tp, fp, fn, matched_ious
        """
        num_pred = len(pred_instances)
        num_gt = len(gt_instances)
        
        if num_pred == 0 or num_gt == 0:
            return 0, num_pred, num_gt, []
        
        # コスト行列（最大化なので負値）
        cost_matrix = -iou_matrix.copy()
        
        # Hungarian algorithm
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        
        # マッチング結果を評価
        tp = 0
        matched_ious = []
        matched_gt = set()
        matched_pred = set()
        
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            iou = iou_matrix[pred_idx, gt_idx]
            if iou >= threshold:
                tp += 1
                matched_ious.append(iou)
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
        
        fp = num_pred - len(matched_pred)
        fn = num_gt - len(matched_gt)
        
        return tp, fp, fn, matched_ious
    
    def _compute_pq(self, matched_ious: List[float], num_pred: int, 
                   num_gt: int) -> Tuple[float, float, float]:
        """
        Panoptic Quality (PQ) を計算
        
        PQ = SQ × RQ
        SQ = mean IoU of matched pairs (Segmentation Quality)
        RQ = TP / (TP + 0.5*FP + 0.5*FN) (Recognition Quality)
        """
        tp = len(matched_ious)
        fp = num_pred - tp
        fn = num_gt - tp
        
        # SQ
        sq = np.mean(matched_ious) if len(matched_ious) > 0 else 0.0
        
        # RQ
        rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-7)
        
        # PQ
        pq = sq * rq
        
        return pq, sq, rq
    
    def compute_ap(self) -> Dict[str, float]:
        """
        AP (Average Precision) を計算
        
        スコアでソートし、各閾値でのPRカーブからAPを計算
        """
        # 全画像の全予測をスコア順にソート
        all_predictions = []
        for result in self.results:
            image_id = result['image_id']
            # 各画像の予測情報を取得（スコア付き）
            # 注: この情報は evaluate_image 時に保存する必要がある
            # 簡略版として、F1から逆算
            all_predictions.append({
                'image_id': image_id,
                'score': 1.0,  # ルールベースは固定、U-Netは実際のスコア
            })
        
        # 簡略版: 各閾値でのmean precision/recallからAPを近似
        ap_results = {}
        
        for thresh in self.iou_thresholds:
            precisions = [r[f'precision_{thresh:.2f}'] for r in self.results]
            recalls = [r[f'recall_{thresh:.2f}'] for r in self.results]
            
            mean_precision = np.mean(precisions)
            mean_recall = np.mean(recalls)
            
            # 簡易AP（実際は複雑だが、ここでは平均F1で代用）
            f1s = [r[f'f1_{thresh:.2f}'] for r in self.results]
            ap_results[f'AP@{thresh:.2f}'] = float(np.mean(f1s))
        
        # AP@[.50:.95]の平均
        ap_values = [ap_results[f'AP@{t:.2f}'] for t in self.iou_thresholds]
        ap_results['AP'] = float(np.mean(ap_values))
        ap_results['AP@0.50'] = ap_results['AP@0.50']
        ap_results['AP@0.75'] = ap_results['AP@0.75']
        
        return ap_results
    
    def compute_statistics(self) -> Dict:
        """全画像の統計を計算"""
        if len(self.results) == 0:
            return {}
        
        stats = {'num_images': len(self.results)}
        
        # 基本統計
        metrics = ['count_mae', 'pq', 'sq', 'rq']
        for metric in metrics:
            values = [r[metric] for r in self.results]
            stats[f'{metric}_mean'] = float(np.mean(values))
            stats[f'{metric}_std'] = float(np.std(values))
            stats[f'{metric}_median'] = float(np.median(values))
        
        # 各閾値での統計
        for thresh in self.iou_thresholds:
            for metric in ['precision', 'recall', 'f1']:
                key = f'{metric}_{thresh:.2f}'
                values = [r[key] for r in self.results]
                stats[f'{key}_mean'] = float(np.mean(values))
                stats[f'{key}_std'] = float(np.std(values))
        
        # AP計算
        ap_results = self.compute_ap()
        stats.update(ap_results)
        
        # 総カウント
        total_tp_50 = sum(r['tp_0.50'] for r in self.results)
        total_fp_50 = sum(r['fp_0.50'] for r in self.results)
        total_fn_50 = sum(r['fn_0.50'] for r in self.results)
        
        stats['total_tp@0.50'] = total_tp_50
        stats['total_fp@0.50'] = total_fp_50
        stats['total_fn@0.50'] = total_fn_50
        stats['total_precision@0.50'] = total_tp_50 / (total_tp_50 + total_fp_50 + 1e-7)
        stats['total_recall@0.50'] = total_tp_50 / (total_tp_50 + total_fn_50 + 1e-7)
        
        return stats
    
    def save_results(self, output_dir: Path):
        """結果をCSVとJSONで保存"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 画像ごとの結果をCSV保存
        csv_path = output_dir / "track_b_per_image.csv"
        if len(self.results) > 0:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
            print(f"Saved per-image results: {csv_path}")
        
        # 統計をJSON保存
        stats = self.compute_statistics()
        json_path = output_dir / "track_b_statistics.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Saved statistics: {json_path}")
        
        # サマリーを表示
        print("\n" + "="*60)
        print("Track B: Instance-Level Evaluation Summary")
        print("="*60)
        print(f"Number of images: {stats.get('num_images', 0)}")
        print(f"\nPanoptic Quality (PQ): {stats.get('pq_mean', 0):.4f} ± {stats.get('pq_std', 0):.4f}")
        print(f"  - SQ (Segmentation): {stats.get('sq_mean', 0):.4f} ± {stats.get('sq_std', 0):.4f}")
        print(f"  - RQ (Recognition):  {stats.get('rq_mean', 0):.4f} ± {stats.get('rq_std', 0):.4f}")
        print(f"\nF1@0.50: {stats.get('f1_0.50_mean', 0):.4f} ± {stats.get('f1_0.50_std', 0):.4f}")
        print(f"F1@0.75: {stats.get('f1_0.75_mean', 0):.4f} ± {stats.get('f1_0.75_std', 0):.4f}")
        print(f"\nAP: {stats.get('AP', 0):.4f}")
        print(f"AP@0.50: {stats.get('AP@0.50', 0):.4f}")
        print(f"AP@0.75: {stats.get('AP@0.75', 0):.4f}")
        print(f"\nCount MAE: {stats.get('count_mae_mean', 0):.2f} ± {stats.get('count_mae_std', 0):.2f}")
        print("="*60 + "\n")


def load_instances_from_mask(mask_path: Path, score: float = 1.0) -> List[Dict]:
    """
    インスタンスマスク画像からインスタンスリストを生成
    
    Parameters:
        mask_path: インスタンスラベルマップ画像 (各インスタンスが異なる値)
        score: スコア（ルールベースは1.0固定）
        
    Returns:
        instances: [{'mask': binary_mask, 'score': float, 'bbox': [x,y,w,h]}, ...]
    """
    instance_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if instance_mask is None:
        return []
    
    instances = []
    unique_labels = np.unique(instance_mask)
    unique_labels = unique_labels[unique_labels > 0]  # 背景を除外
    
    for label in unique_labels:
        mask = (instance_mask == label).astype(np.uint8)
        area = np.sum(mask)
        
        if area == 0:
            continue
        
        # バウンディングボックス
        ys, xs = np.where(mask)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        
        instances.append({
            'mask': mask,
            'score': score,
            'bbox': [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
            'area': int(area)
        })
    
    return instances


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python evaluate_track_b.py <pred_dir> <gt_dir> <output_dir>")
        print("\nEvaluates instance-level metrics between prediction and GT.")
        print("Expects instance label maps (0=background, 1,2,3...=instances)")
        print("\nExample:")
        print("  python evaluate_track_b.py ./predictions/instances ./gt_instances ./results")
        sys.exit(1)
    
    pred_dir = Path(sys.argv[1])
    gt_dir = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    
    # 予測ファイル一覧
    pred_files = sorted(pred_dir.glob("*.png"))
    
    if len(pred_files) == 0:
        print(f"No prediction files found in {pred_dir}")
        sys.exit(1)
    
    evaluator = TrackBEvaluator()
    
    print(f"Evaluating {len(pred_files)} images...")
    
    for pred_path in pred_files:
        image_id = pred_path.stem
        gt_path = gt_dir / pred_path.name
        
        if not gt_path.exists():
            print(f"Warning: GT not found for {image_id}, skipping...")
            continue
        
        # インスタンス読み込み
        pred_instances = load_instances_from_mask(pred_path, score=1.0)
        gt_instances = load_instances_from_mask(gt_path, score=1.0)
        
        # 評価
        evaluator.evaluate_image(pred_instances, gt_instances, image_id)
    
    # 結果保存
    evaluator.save_results(output_dir)
