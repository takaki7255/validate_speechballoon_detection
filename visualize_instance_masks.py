#!/usr/bin/env python3
"""
visualize_instance_masks.py
============================

インスタンスマスクを視覚化するツール

インスタンスマスクは0〜255の小さい値（0=背景、1,2,3...=各インスタンス）を持つため、
通常の画像ビューアでは真っ黒に見えます。このツールは各インスタンスに異なる色を割り当てて
視覚的に確認できるようにします。

使用例:
    # 単一ファイル
    python visualize_instance_masks.py evaluation_dataset/gt_instances/ARMS_017.png output_vis.png
    
    # ディレクトリ全体
    python visualize_instance_masks.py evaluation_dataset/gt_instances/ output_vis_dir/
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def generate_color_map(num_instances):
    """
    インスタンス数に応じたカラーマップを生成
    
    Parameters:
        num_instances: インスタンスの数
        
    Returns:
        color_map: (num_instances+1, 3) のRGBカラーマップ（0は黒=背景）
    """
    if num_instances == 0:
        return np.array([[0, 0, 0]])
    
    # カラーマップを生成（背景は黒）
    cmap = plt.cm.get_cmap('tab20', num_instances)
    colors = [cmap(i)[:3] for i in range(num_instances)]
    
    # より多くのインスタンスがある場合は別のカラーマップも使用
    if num_instances > 20:
        cmap2 = plt.cm.get_cmap('tab20b', num_instances - 20)
        colors.extend([cmap2(i)[:3] for i in range(num_instances - 20)])
    
    # 背景（ID=0）は黒
    color_map = np.array([[0, 0, 0]] + colors)
    return (color_map * 255).astype(np.uint8)


def visualize_instance_mask(instance_mask_path, output_path=None, show_stats=True):
    """
    インスタンスマスクを可視化
    
    Parameters:
        instance_mask_path: インスタンスマスクのパス
        output_path: 出力パス（Noneの場合は元のファイル名_vis.png）
        show_stats: 統計情報を表示するかどうか
    """
    # 読み込み
    instance_mask = np.array(Image.open(instance_mask_path))
    
    # ユニークなインスタンスID
    unique_ids = np.unique(instance_mask)
    num_instances = len(unique_ids) - 1 if 0 in unique_ids else len(unique_ids)
    
    if show_stats:
        print(f"File: {Path(instance_mask_path).name}")
        print(f"  Shape: {instance_mask.shape}")
        print(f"  Number of instances: {num_instances}")
        print(f"  Instance IDs: {unique_ids}")
        print(f"  Value range: [{instance_mask.min()}, {instance_mask.max()}]")
    
    # カラーマップ生成
    color_map = generate_color_map(max(unique_ids) if len(unique_ids) > 0 else 0)
    
    # カラー画像に変換
    h, w = instance_mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for instance_id in unique_ids:
        if instance_id < len(color_map):
            colored_mask[instance_mask == instance_id] = color_map[instance_id]
    
    # 保存
    if output_path is None:
        stem = Path(instance_mask_path).stem
        output_path = Path(instance_mask_path).parent / f"{stem}_vis.png"
    
    Image.fromarray(colored_mask).save(output_path)
    
    if show_stats:
        print(f"  Saved visualization to: {output_path}\n")
    
    return colored_mask


def visualize_directory(input_dir, output_dir):
    """
    ディレクトリ内の全インスタンスマスクを可視化
    
    Parameters:
        input_dir: インスタンスマスクディレクトリ
        output_dir: 出力ディレクトリ
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mask_files = sorted(input_dir.glob("*.png"))
    
    if len(mask_files) == 0:
        print(f"No PNG files found in {input_dir}")
        return
    
    print(f"Visualizing {len(mask_files)} instance masks...")
    print("=" * 60)
    
    total_instances = 0
    
    for idx, mask_file in enumerate(mask_files, 1):
        output_path = output_dir / f"{mask_file.stem}_vis.png"
        
        # 可視化
        instance_mask = np.array(Image.open(mask_file))
        unique_ids = np.unique(instance_mask)
        num_instances = len(unique_ids) - 1 if 0 in unique_ids else len(unique_ids)
        total_instances += num_instances
        
        visualize_instance_mask(mask_file, output_path, show_stats=False)
        
        if idx % 10 == 0:
            print(f"  Processed {idx}/{len(mask_files)} files...")
    
    print("=" * 60)
    print(f"\n✓ Completed! Visualized {len(mask_files)} instance masks")
    print(f"  Total instances: {total_instances}")
    print(f"  Average instances per image: {total_instances / len(mask_files):.2f}")
    print(f"  Output directory: {output_dir}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_instance_masks.py <input_path> [output_path]")
        print("\nVisualize instance masks by assigning different colors to each instance.")
        print("\nExamples:")
        print("  # Single file:")
        print("  python visualize_instance_masks.py evaluation_dataset/gt_instances/ARMS_017.png output_vis.png")
        print("\n  # Directory:")
        print("  python visualize_instance_masks.py evaluation_dataset/gt_instances/ output_vis_dir/")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    if input_path.is_dir():
        # ディレクトリ処理
        if output_path is None:
            output_path = input_path.parent / f"{input_path.name}_visualized"
        visualize_directory(input_path, output_path)
    else:
        # 単一ファイル処理
        visualize_instance_mask(input_path, output_path, show_stats=True)


if __name__ == "__main__":
    main()
