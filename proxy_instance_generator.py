#!/usr/bin/env python3
"""
proxy_instance_generator.py
===========================

GT二値マスクから疑似インスタンス（Proxy-Instance）を生成するスクリプト

アルゴリズム:
1. 前処理: 穴埋め、微小ノイズ除去
2. 距離変換 + marker-based watershed で分割
3. 後処理: 過分割の統合、極小片の統合

パラメータは固定して公開し、公平性を担保
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, label
from skimage import morphology
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from typing import List, Tuple, Dict
import json


class ProxyInstanceGenerator:
    """GT二値マスクから疑似インスタンスを生成するクラス"""
    
    # 固定パラメータ（事前公開）
    PARAMS = {
        'min_area_threshold': 100,           # 最小面積（これより小さいノイズを除去）
        'hole_fill_area_threshold': 500,     # 埋める穴の最大面積
        'distance_sigma': 1.5,               # 距離変換の平滑化σ
        'min_distance': 10,                  # 局所極大の最小距離
        'watershed_compactness': 0.001,      # watershedのコンパクトネス
        'merge_iou_threshold': 0.3,          # 統合するIoU閾値
        'merge_area_ratio_threshold': 0.1,   # 小片を統合する面積比閾値
        'elongation_threshold': 5.0,         # 統合する細長さ閾値
    }
    
    def __init__(self, params: Dict = None):
        """
        Parameters:
            params: パラメータ辞書（Noneの場合はデフォルト使用）
        """
        self.params = self.PARAMS.copy()
        if params:
            self.params.update(params)
    
    def generate(self, binary_mask: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        GT二値マスクから疑似インスタンスを生成
        
        Parameters:
            binary_mask: 二値マスク (H, W), 255=balloon, 0=background
            
        Returns:
            instance_mask: インスタンスラベルマップ (H, W), 0=background, 1,2,3...=instances
            instances_info: インスタンス情報のリスト
                [{'id': 1, 'bbox': [x, y, w, h], 'area': 1234, 'mask': binary}, ...]
        """
        # 1. 前処理
        preprocessed = self._preprocess(binary_mask)
        
        # 2. 距離変換 + watershed分割
        instance_mask = self._watershed_segmentation(preprocessed)
        
        # 3. 後処理: 過分割の統合
        instance_mask = self._merge_oversegmented(instance_mask, preprocessed)
        
        # 4. インスタンス情報を抽出
        instances_info = self._extract_instance_info(instance_mask)
        
        return instance_mask, instances_info
    
    def _preprocess(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        前処理: 穴埋め、微小ノイズ除去
        """
        # 二値化（0 or 255 -> 0 or 1）
        mask = (binary_mask > 127).astype(np.uint8)
        
        # 穴埋め（各連結成分内の小穴を埋める）
        # まず全体の連結成分を取得
        num_labels, labels = cv2.connectedComponents(mask)
        
        filled = mask.copy()
        for label_id in range(1, num_labels):
            component_mask = (labels == label_id).astype(np.uint8)
            
            # この成分の穴を検出（背景から成分を引く）
            inverted = 1 - component_mask
            num_holes, hole_labels = cv2.connectedComponents(inverted)
            
            # 穴を埋める（外周を除く）
            for hole_id in range(1, num_holes):
                hole_mask = (hole_labels == hole_id).astype(np.uint8)
                hole_area = np.sum(hole_mask)
                
                # 画像境界に接していない かつ 小さい穴のみ埋める
                if hole_area < self.params['hole_fill_area_threshold']:
                    # 境界チェック
                    h, w = hole_mask.shape
                    if not (np.any(hole_mask[0, :]) or np.any(hole_mask[-1, :]) or
                           np.any(hole_mask[:, 0]) or np.any(hole_mask[:, -1])):
                        filled = filled | hole_mask
        
        # 微小ノイズ除去（小さすぎる連結成分を削除）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
        
        cleaned = np.zeros_like(filled)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area >= self.params['min_area_threshold']:
                cleaned[labels == label_id] = 1
        
        return cleaned
    
    def _watershed_segmentation(self, mask: np.ndarray) -> np.ndarray:
        """
        距離変換 + marker-based watershedで分割
        """
        if np.sum(mask) == 0:
            return np.zeros_like(mask, dtype=np.int32)
        
        # 距離変換
        distance = distance_transform_edt(mask)
        
        # ガウシアン平滑化
        distance_smooth = ndimage.gaussian_filter(distance, sigma=self.params['distance_sigma'])
        
        # 局所極大を検出（seeds）
        local_max = peak_local_max(
            distance_smooth,
            min_distance=self.params['min_distance'],
            labels=mask.astype(int),
            exclude_border=False
        )
        
        # マーカー作成
        markers = np.zeros_like(mask, dtype=int)
        for idx, (y, x) in enumerate(local_max, start=1):
            markers[y, x] = idx
        
        # マーカーを膨張（連結成分化）
        markers = morphology.dilation(markers, morphology.disk(2))
        markers = markers * mask  # マスク内のみ
        
        # Watershed
        if np.max(markers) > 0:
            labels = watershed(
                -distance_smooth,
                markers,
                mask=mask,
                compactness=self.params['watershed_compactness']
            )
        else:
            # マーカーがない場合は単一の連結成分として扱う
            labels = mask.astype(np.int32)
        
        return labels
    
    def _merge_oversegmented(self, instance_mask: np.ndarray, original_mask: np.ndarray) -> np.ndarray:
        """
        過分割されたインスタンスを統合
        
        統合条件:
        - 隣接している
        - 境界が細線（IoUが高い）
        - 片方が極小（面積比が小さい）
        - 細長い形状
        """
        merged = instance_mask.copy()
        num_instances = np.max(merged)
        
        if num_instances <= 1:
            return merged
        
        # 各インスタンスの情報を取得
        instance_info = {}
        for inst_id in range(1, num_instances + 1):
            mask = (merged == inst_id)
            area = np.sum(mask)
            
            # バウンディングボックス
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            
            # 細長さ計算
            bbox_area = (x_max - x_min + 1) * (y_max - y_min + 1)
            elongation = bbox_area / (area + 1e-6)
            
            instance_info[inst_id] = {
                'mask': mask,
                'area': area,
                'bbox': (x_min, y_min, x_max, y_max),
                'elongation': elongation
            }
        
        # 統合処理（貪欲法）
        merged_ids = set()
        for inst_id in range(1, num_instances + 1):
            if inst_id not in instance_info or inst_id in merged_ids:
                continue
            
            info = instance_info[inst_id]
            mask_i = info['mask']
            area_i = info['area']
            
            # 膨張して隣接を検出
            dilated = morphology.binary_dilation(mask_i, morphology.disk(1))
            neighbors = np.unique(merged[dilated]) 
            neighbors = [n for n in neighbors if n != 0 and n != inst_id and n not in merged_ids]
            
            best_neighbor = None
            best_score = -1
            
            for neighbor_id in neighbors:
                if neighbor_id not in instance_info:
                    continue
                
                info_j = instance_info[neighbor_id]
                mask_j = info_j['mask']
                area_j = info_j['area']
                
                # 統合判定条件
                # 1. 面積比が小さい（小片）
                area_ratio = min(area_i, area_j) / max(area_i, area_j)
                
                # 2. 細長い形状
                is_elongated = (info['elongation'] > self.params['elongation_threshold'] or
                               info_j['elongation'] > self.params['elongation_threshold'])
                
                # 3. IoU（統合したときの妥当性）
                intersection = np.sum(mask_i & mask_j)
                union = np.sum(mask_i | mask_j)
                # 隣接のため直接のIoUは0、膨張で接触面積を評価
                contact_area = np.sum(dilated & mask_j)
                contact_ratio = contact_area / min(area_i, area_j)
                
                # スコアリング
                if area_ratio < self.params['merge_area_ratio_threshold'] or is_elongated:
                    score = contact_ratio
                    if score > best_score:
                        best_score = score
                        best_neighbor = neighbor_id
            
            # 統合実行
            if best_neighbor is not None and best_score > 0.1:
                merged[merged == best_neighbor] = inst_id
                merged_ids.add(best_neighbor)
                # 情報更新
                instance_info[inst_id]['mask'] = (merged == inst_id)
                instance_info[inst_id]['area'] += instance_info[best_neighbor]['area']
        
        # ラベルを再割り当て（詰める）
        unique_labels = np.unique(merged)
        unique_labels = unique_labels[unique_labels > 0]
        
        relabeled = np.zeros_like(merged)
        for new_id, old_id in enumerate(sorted(unique_labels), start=1):
            relabeled[merged == old_id] = new_id
        
        return relabeled
    
    def _extract_instance_info(self, instance_mask: np.ndarray) -> List[Dict]:
        """
        インスタンスマスクから各インスタンスの情報を抽出
        """
        instances = []
        num_instances = np.max(instance_mask)
        
        for inst_id in range(1, num_instances + 1):
            mask = (instance_mask == inst_id).astype(np.uint8)
            area = np.sum(mask)
            
            if area == 0:
                continue
            
            # バウンディングボックス
            ys, xs = np.where(mask)
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            
            instances.append({
                'id': inst_id,
                'bbox': [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
                'area': int(area),
                'mask': mask
            })
        
        return instances
    
    @staticmethod
    def save_params(filepath: str):
        """パラメータをJSONで保存"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(ProxyInstanceGenerator.PARAMS, f, indent=2, ensure_ascii=False)
        print(f"Parameters saved to: {filepath}")


def visualize_instances(image: np.ndarray, instance_mask: np.ndarray, output_path: str):
    """インスタンスを色分けして可視化"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    num_instances = np.max(instance_mask)
    
    # ランダムカラーマップ生成
    np.random.seed(42)
    colors = np.random.rand(num_instances + 1, 3)
    colors[0] = [0, 0, 0]  # 背景は黒
    cmap = ListedColormap(colors)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 元画像
    if len(image.shape) == 2:
        axes[0].imshow(image, cmap='gray')
    else:
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # インスタンスマスク
    axes[1].imshow(instance_mask, cmap=cmap, interpolation='nearest')
    axes[1].set_title(f'Proxy Instances (n={num_instances})')
    axes[1].axis('off')
    
    # オーバーレイ
    if len(image.shape) == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    overlay = img_rgb.copy()
    for inst_id in range(1, num_instances + 1):
        mask = (instance_mask == inst_id)
        color = (colors[inst_id] * 255).astype(np.uint8)
        overlay[mask] = overlay[mask] * 0.5 + color * 0.5
    
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 3:
        print("Usage: python proxy_instance_generator.py <gt_mask_path> <output_dir>")
        print("\nExample:")
        print("  python proxy_instance_generator.py gt_mask.png ./output")
        sys.exit(1)
    
    gt_mask_path = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GT二値マスク読み込み
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print(f"Error: Cannot load {gt_mask_path}")
        sys.exit(1)
    
    # 生成器実行
    generator = ProxyInstanceGenerator()
    instance_mask, instances_info = generator.generate(gt_mask)
    
    print(f"Generated {len(instances_info)} proxy instances")
    
    # 保存
    # インスタンスマスク保存
    instance_output = output_dir / "proxy_instances.png"
    # ラベルマップを可視化用に正規化
    if len(instances_info) > 0:
        vis_mask = (instance_mask * (255 // len(instances_info))).astype(np.uint8)
        cv2.imwrite(str(instance_output), vis_mask)
    
    # インスタンス情報をJSON保存
    json_output = output_dir / "proxy_instances.json"
    instances_serializable = []
    for inst in instances_info:
        inst_copy = inst.copy()
        # maskはRLE形式で保存（簡略版: バイナリマスクは保存しない）
        inst_copy.pop('mask')
        instances_serializable.append(inst_copy)
    
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump({
            'num_instances': len(instances_info),
            'instances': instances_serializable
        }, f, indent=2)
    
    print(f"Saved instance mask: {instance_output}")
    print(f"Saved instance info: {json_output}")
    
    # パラメータ保存
    param_output = output_dir / "proxy_generator_params.json"
    ProxyInstanceGenerator.save_params(str(param_output))
