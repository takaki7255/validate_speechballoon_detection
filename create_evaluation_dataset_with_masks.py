#!/usr/bin/env python3
"""
create_evaluation_dataset_with_masks.py
========================================

評価用データセットの作成とマスク生成を一括で行う統合スクリプト

機能:
1. Manga109から吹き出しを含む画像をランダムサンプリング
2. アノテーション（RLEセグメンテーション付き）をJSONで保存
3. 二値マスク（Track A用）とインスタンスマスク（Track B用）を自動生成

使用例:
    python create_evaluation_dataset_with_masks.py ../Manga109_released_2023_12_07/images ../Manga109_released_2023_12_07/manga_seg_jsons ./evaluation_dataset 100
"""

import json
import os
import sys
import shutil
import random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from pycocotools import mask as maskUtils
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class SampleInfo:
    """サンプリングした画像の情報"""
    title: str
    page_filename: str
    image_path: str
    output_image_path: str
    output_annotation_path: str
    output_binary_mask_path: str
    output_instance_mask_path: str
    num_balloons: int
    has_annotation: bool


class EvaluationDatasetCreator:
    """
    評価用データセット作成クラス（マスク生成統合版）
    """
    
    def __init__(self, images_root: str, jsons_root: str, output_root: str, target_count: int = 100):
        """
        Parameters:
            images_root: Manga109の画像ルートディレクトリ
            jsons_root: COCO形式JSONファイルのルートディレクトリ
            output_root: 出力先ディレクトリ
            target_count: サンプリングする画像数
        """
        self.images_root = Path(images_root)
        self.jsons_root = Path(jsons_root)
        self.output_root = Path(output_root)
        self.target_count = target_count
        
        # 出力ディレクトリの設定
        self.sampled_images_dir = self.output_root / 'sampled_images'
        self.gt_annotations_dir = self.output_root / 'gt_annotations'
        self.gt_masks_dir = self.output_root / 'gt_masks'
        self.gt_instances_dir = self.output_root / 'gt_instances'
        
        # ディレクトリ作成
        for directory in [self.sampled_images_dir, self.gt_annotations_dir, 
                         self.gt_masks_dir, self.gt_instances_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_all_titles(self) -> List[str]:
        """全タイトルのリストを取得"""
        if not self.images_root.exists():
            print(f"Error: Images root directory does not exist: {self.images_root}")
            return []
        
        titles = [d.name for d in self.images_root.iterdir() if d.is_dir()]
        return sorted(titles)
    
    def get_image_paths_for_title(self, title: str) -> List[Path]:
        """指定タイトルの全画像パスを取得"""
        title_dir = self.images_root / title
        if not title_dir.exists():
            return []
        
        # 一般的な画像拡張子
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(title_dir.glob(ext))
        
        return sorted(image_paths)
    
    def load_annotation_json(self, title: str) -> Optional[Dict]:
        """タイトルに対応するCOCO形式のアノテーションJSONを読み込み"""
        json_path = self.jsons_root / f"{title}.json"
        
        if not json_path.exists():
            print(f"Warning: Annotation file not found for title '{title}': {json_path}")
            return None
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error: Failed to load JSON for title '{title}': {e}")
            return None
    
    def extract_balloon_annotations_for_page(self, annotation_data: Dict, page_filename: str, 
                                            title: str) -> Tuple[List[Dict], Optional[int], Optional[int]]:
        """
        指定ページの吹き出しアノテーションを抽出（COCO形式、RLEセグメンテーション付き）
        
        Returns:
            (balloon_list, image_width, image_height)
        """
        if not annotation_data:
            return [], None, None
        
        try:
            # file_name形式: "タイトル/ページ番号.jpg"
            target_file_name = f"{title}/{page_filename}"
            
            # imagesから該当ページを探す
            images = annotation_data.get('images', [])
            target_image = None
            for img in images:
                if img.get('file_name') == target_file_name:
                    target_image = img
                    break
            
            if not target_image:
                return [], None, None
            
            image_id = target_image.get('id')
            image_width = target_image.get('width')
            image_height = target_image.get('height')
            
            # annotationsから該当画像のballoon（category_id=5）を抽出
            annotations = annotation_data.get('annotations', [])
            balloon_list = []
            
            for ann in annotations:
                # category_id=5がballoon
                if ann.get('category_id') == 5 and ann.get('image_id') == image_id:
                    bbox = ann.get('bbox', [0, 0, 0, 0])
                    # COCO形式のbbox: [x, y, width, height]
                    x, y, w, h = bbox
                    
                    balloon_info = {
                        'id': str(ann.get('id', '')),
                        'xmin': int(x),
                        'ymin': int(y),
                        'xmax': int(x + w),
                        'ymax': int(y + h),
                        'width': int(w),
                        'height': int(h),
                        'area': ann.get('area', w * h)
                    }
                    
                    # セグメンテーション情報も保存
                    if 'segmentation' in ann:
                        balloon_info['segmentation'] = ann['segmentation']
                    
                    balloon_list.append(balloon_info)
            
            return balloon_list, image_width, image_height
            
        except Exception as e:
            print(f"Error extracting balloon annotations for {page_filename}: {e}")
            import traceback
            traceback.print_exc()
            return [], None, None
    
    def create_mask_from_bbox(self, balloon: dict, height: int, width: int) -> np.ndarray:
        """
        バウンディングボックスから矩形マスクを生成（フォールバック用）
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        xmin = balloon.get('xmin', 0)
        ymin = balloon.get('ymin', 0)
        xmax = balloon.get('xmax', width)
        ymax = balloon.get('ymax', height)
        
        # 範囲チェック
        xmin = max(0, min(xmin, width - 1))
        xmax = max(0, min(xmax, width))
        ymin = max(0, min(ymin, height - 1))
        ymax = max(0, min(ymax, height))
        
        mask[ymin:ymax, xmin:xmax] = 1
        
        return mask
    
    def generate_masks_from_balloons(self, balloons: List[Dict], width: int, height: int,
                                    output_binary_path: Path, output_instance_path: Path) -> Tuple[int, int]:
        """
        吹き出しリストからマスクを生成
        
        重複する領域の処理:
        - 重複がある場合、面積が大きい方を優先
        - または、重複部分は最初のインスタンスに割り当て
        
        Returns:
            (segmentation_count, bbox_fallback_count)
        """
        import cv2
        
        # Track A: 二値マスク（0=背景、255=balloon）
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Track B: インスタンスマスク（0=背景、1,2,3...=各balloonインスタンス）
        instance_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 優先度マスク（重複時に面積が大きい方を優先）
        priority_mask = np.zeros((height, width), dtype=np.int32)  # 各ピクセルの優先度（面積）
        
        segmentation_count = 0
        bbox_fallback_count = 0
        
        # 各balloonの面積を計算して優先度付け
        balloon_data = []
        for idx, balloon in enumerate(balloons, start=1):
            # セグメンテーションマスクがある場合はそれを使用
            if 'segmentation' in balloon:
                segmentation = balloon['segmentation']
                
                # RLE形式の場合
                if isinstance(segmentation, dict) and 'counts' in segmentation:
                    # pycocotoolsでデコード
                    if 'size' not in segmentation:
                        segmentation['size'] = [height, width]
                    
                    try:
                        mask = maskUtils.decode(segmentation)
                        if len(mask.shape) == 3:
                            mask = np.any(mask, axis=2).astype(np.uint8)
                        else:
                            mask = mask.astype(np.uint8)
                        segmentation_count += 1
                    except Exception as e:
                        print(f"Warning: Failed to decode RLE mask: {e}")
                        # フォールバック: bboxを使用
                        mask = self.create_mask_from_bbox(balloon, height, width)
                        bbox_fallback_count += 1
                else:
                    # ポリゴン形式の場合（未実装、bboxにフォールバック）
                    mask = self.create_mask_from_bbox(balloon, height, width)
                    bbox_fallback_count += 1
            else:
                # セグメンテーションがない場合はbboxから生成
                mask = self.create_mask_from_bbox(balloon, height, width)
                bbox_fallback_count += 1
            
            # Track A: 二値マスクに統合（OR演算）
            binary_mask = np.logical_or(binary_mask, mask).astype(np.uint8)
            
            # 面積を計算（優先度として使用）
            area = np.sum(mask > 0)
            
            balloon_data.append({
                'id': idx,
                'mask': mask,
                'area': area
            })
        
        # 面積が大きい順にソート（大きいものを優先）
        balloon_data.sort(key=lambda x: x['area'], reverse=True)
        
        # インスタンスマスク生成（面積が大きい順に描画）
        for data in balloon_data:
            instance_id = data['id']
            mask = data['mask']
            
            if instance_id > 255:  # 8bit制限
                print(f"Warning: Too many instances ({len(balloons)}), truncating to 255")
                continue
            
            # まだ割り当てられていない領域にのみIDを設定
            # これにより、面積が大きいものが優先的に領域を確保
            unassigned = (instance_mask == 0) & (mask > 0)
            instance_mask[unassigned] = instance_id
        
        # 保存
        # Track A: 二値マスク（0 or 255）
        Image.fromarray((binary_mask * 255).astype(np.uint8)).save(output_binary_path)
        
        # Track B: インスタンスマスク（0, 1, 2, 3, ...）
        Image.fromarray(instance_mask.astype(np.uint8)).save(output_instance_path)
        
        return segmentation_count, bbox_fallback_count
    
    def sample_and_create_dataset(self) -> List[SampleInfo]:
        """
        全タイトルから画像を無作為にサンプリングし、アノテーションとマスクを生成
        吹き出しが1つ以上含まれる画像のみを選択
        """
        titles = self.get_all_titles()
        if not titles:
            print("No titles found.")
            return []
        
        # 全タイトルの画像を収集し、吹き出しの有無をチェック
        print("Collecting images with balloon annotations...")
        all_image_data = []
        
        for title in titles:
            # アノテーションデータを事前に読み込み
            annotation_data = self.load_annotation_json(title)
            if not annotation_data:
                continue
            
            image_paths = self.get_image_paths_for_title(title)
            
            for img_path in image_paths:
                page_filename = img_path.name
                
                # この画像に吹き出しがあるかチェック
                balloons, _, _ = self.extract_balloon_annotations_for_page(
                    annotation_data, page_filename, title
                )
                
                # 吹き出しが1つ以上ある画像のみを候補に追加
                if len(balloons) > 0:
                    all_image_data.append((title, img_path, annotation_data, balloons))
        
        print(f"Total images found: {len(all_image_data)}")
        print(f"Images with balloons: {len(all_image_data)}")
        
        if len(all_image_data) == 0:
            print("No images with balloons found.")
            return []
        
        # ランダムにサンプリング
        sample_count = min(self.target_count, len(all_image_data))
        sampled_data = random.sample(all_image_data, sample_count)
        
        print(f"\nSampling {sample_count} images (all with balloons)...")
        print("=" * 60)
        
        # サンプリングした画像を処理
        samples = []
        total_segmentation_count = 0
        total_bbox_fallback_count = 0
        
        for idx, (title, img_path, annotation_data, balloons) in enumerate(sampled_data, 1):
            page_filename = img_path.name
            
            # 出力ファイル名: タイトル_ページ番号.拡張子
            output_image_filename = f"{title}_{page_filename}"
            output_annotation_filename = f"{title}_{Path(page_filename).stem}.json"
            output_mask_filename = f"{title}_{Path(page_filename).stem}.png"
            
            output_image_path = self.sampled_images_dir / output_image_filename
            output_annotation_path = self.gt_annotations_dir / output_annotation_filename
            output_binary_mask_path = self.gt_masks_dir / output_mask_filename
            output_instance_mask_path = self.gt_instances_dir / output_mask_filename
            
            # 画像をコピー
            try:
                shutil.copy2(img_path, output_image_path)
            except Exception as e:
                print(f"Error copying image {img_path}: {e}")
                continue
            
            # 画像サイズを取得（balloonsは既に取得済みだが、セグメンテーション付きで再取得）
            balloons_with_size, img_width, img_height = self.extract_balloon_annotations_for_page(
                annotation_data, page_filename, title
            )
            
            # アノテーションJSONを保存
            has_annotation = len(balloons_with_size) > 0
            annotation_output = {
                'title': title,
                'page': page_filename,
                'image_width': img_width,
                'image_height': img_height,
                'balloons': balloons_with_size
            }
            
            try:
                with open(output_annotation_path, 'w', encoding='utf-8') as f:
                    json.dump(annotation_output, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error writing annotation file {output_annotation_path}: {e}")
                continue
            
            # マスク生成
            try:
                seg_count, bbox_count = self.generate_masks_from_balloons(
                    balloons_with_size, img_width, img_height,
                    output_binary_mask_path, output_instance_mask_path
                )
                total_segmentation_count += seg_count
                total_bbox_fallback_count += bbox_count
                
                # 重複チェック
                from PIL import Image as PILImage
                instance_check = np.array(PILImage.open(output_instance_mask_path))
                num_instances_in_mask = len(np.unique(instance_check)) - 1  # 0を除く
                overlap_count = len(balloons_with_size) - num_instances_in_mask
                
                if overlap_count > 0:
                    print(f"  ⚠️  {title}/{page_filename}: {overlap_count} overlapping instances "
                          f"({len(balloons_with_size)} balloons → {num_instances_in_mask} instances)")
                
            except Exception as e:
                print(f"Error generating masks for {page_filename}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # サンプル情報を記録
            sample_info = SampleInfo(
                title=title,
                page_filename=page_filename,
                image_path=str(img_path),
                output_image_path=str(output_image_path),
                output_annotation_path=str(output_annotation_path),
                output_binary_mask_path=str(output_binary_mask_path),
                output_instance_mask_path=str(output_instance_mask_path),
                num_balloons=len(balloons_with_size),
                has_annotation=has_annotation
            )
            samples.append(sample_info)
            
            # 進捗表示
            if idx % 10 == 0:
                print(f"  [{idx}/{sample_count}] Processed {title}/{page_filename} "
                      f"({len(balloons_with_size)} balloons)")
        
        print("=" * 60)
        print(f"\n✓ Successfully created {len(samples)} samples")
        print(f"\nSegmentation statistics:")
        print(f"  With RLE segmentation: {total_segmentation_count}")
        print(f"  Bbox fallback: {total_bbox_fallback_count}")
        
        return samples
    
    def save_summary(self, samples: List[SampleInfo]):
        """サマリー情報を保存"""
        summary_path = self.output_root / 'dataset_summary.json'
        
        summary = {
            'total_samples': len(samples),
            'images_with_balloons': sum(1 for s in samples if s.has_annotation),
            'images_without_balloons': sum(1 for s in samples if not s.has_annotation),
            'total_balloons': sum(s.num_balloons for s in samples),
            'avg_balloons_per_image': sum(s.num_balloons for s in samples) / len(samples) if samples else 0,
            'samples': [asdict(s) for s in samples]
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Dataset summary saved to: {summary_path}")
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {summary['total_samples']}")
        print(f"  Images with balloons: {summary['images_with_balloons']}")
        print(f"  Images without balloons: {summary['images_without_balloons']}")
        print(f"  Total balloons: {summary['total_balloons']}")
        print(f"  Average balloons per image: {summary['avg_balloons_per_image']:.2f}")
        
        print(f"\n✓ Output directories:")
        print(f"  Images: {self.sampled_images_dir}")
        print(f"  Annotations: {self.gt_annotations_dir}")
        print(f"  Binary masks (Track A): {self.gt_masks_dir}")
        print(f"  Instance masks (Track B): {self.gt_instances_dir}")


def main():
    if len(sys.argv) < 5:
        print("Usage: python create_evaluation_dataset_with_masks.py <images_root> <jsons_root> <output_root> <sample_count>")
        print("\nCreates evaluation dataset with images, annotations, and masks in one go.")
        print("\nExample:")
        print("  python create_evaluation_dataset_with_masks.py \\")
        print("      ../Manga109_released_2023_12_07/images \\")
        print("      ../Manga109_released_2023_12_07/manga_seg_jsons \\")
        print("      ./evaluation_dataset \\")
        print("      100")
        sys.exit(1)
    
    images_root = sys.argv[1]
    jsons_root = sys.argv[2]
    output_root = sys.argv[3]
    sample_count = int(sys.argv[4])
    
    print("=" * 60)
    print("Evaluation Dataset Creator with Masks")
    print("=" * 60)
    print(f"Images root: {images_root}")
    print(f"JSONs root: {jsons_root}")
    print(f"Output root: {output_root}")
    print(f"Target sample count: {sample_count}")
    print("=" * 60)
    
    # データセット作成
    creator = EvaluationDatasetCreator(images_root, jsons_root, output_root, sample_count)
    samples = creator.sample_and_create_dataset()
    
    if samples:
        creator.save_summary(samples)
        print("\n✓ All done! Evaluation dataset with masks is ready.")
    else:
        print("\n✗ Failed to create dataset.")
        sys.exit(1)


if __name__ == "__main__":
    main()
