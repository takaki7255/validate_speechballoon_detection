import json
import os
import glob
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils

# ===== 設定 =====
annotations_dir = "./manga_seg_jsons"  # JSONファイルが複数あるフォルダ
image_root = "images"            # 対応する画像のルートパス
output_root = "masks"            # マスク画像出力先

# ===== 全アノテーションファイルを処理 =====
json_files = glob.glob(os.path.join(annotations_dir, "*.json"))

for json_path in json_files:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # カテゴリID → クラス名
    category_map = {cat['id']: cat['name'] for cat in data.get("categories", [])}

    # 画像ID → ファイル名のマッピング
    id_to_filename = {img['id']: img['file_name'] for img in data['images']}

    # 画像ID × クラス名 → マスク統合用辞書
    combined_masks = {}

    # ===== 各アノテーションを読み込んでマスクを統合 =====
    for ann in data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        segmentation = ann['segmentation']
        height, width = segmentation['size']
        class_name = category_map.get(category_id, f"category_{category_id}")
        key = (image_id, class_name)

        # RLEデコード
        mask = maskUtils.decode(segmentation)
        if len(mask.shape) == 3:
            mask = np.any(mask, axis=2).astype(np.uint8)

        # 既存マスクと統合
        if key in combined_masks:
            combined_masks[key] = np.logical_or(combined_masks[key], mask).astype(np.uint8)
        else:
            combined_masks[key] = mask

    # ===== マスク画像の保存 =====
    for (image_id, class_name), mask in combined_masks.items():
        file_name = id_to_filename[image_id]
        manga_title = file_name.split("/")[0]
        img_name = os.path.splitext(os.path.basename(file_name))[0]

        out_dir = os.path.join(output_root, manga_title, class_name)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"{img_name}_mask.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)
        print(f"Saved: {out_path}")
