# インスタンスマスクについて

## インスタンスマスクが「真っ黒」に見える理由

インスタンスマスクは**正常に生成されています**が、通常の画像ビューアで開くと真っ黒に見えます。

### 理由

インスタンスマスクのピクセル値は以下の範囲です：
- **0** = 背景
- **1, 2, 3, ..., N** = 各吹き出しインスタンス（NはTotal balloons数）

例: 20個の吹き出しがある場合、ピクセル値は 0〜20 の範囲

通常の画像ビューアは 0〜255 のグレースケールを表示するため、0〜20程度の値は**ほぼ黒**として表示されます。

### 確認方法

#### 1. Pythonで数値を確認

```python
import numpy as np
from PIL import Image

# インスタンスマスクを読み込み
mask = np.array(Image.open('evaluation_dataset/gt_instances/ARMS_017.png'))

print(f"Shape: {mask.shape}")
print(f"Min: {mask.min()}, Max: {mask.max()}")
print(f"Unique IDs: {np.unique(mask)}")  # 0, 1, 2, 3, ... と表示されるはず
```

#### 2. 可視化ツールを使用

各インスタンスに異なる色を割り当てて可視化します：

```bash
# 単一ファイル
python visualize_instance_masks.py \
    evaluation_dataset/gt_instances/ARMS_017.png \
    output_vis.png

# ディレクトリ全体
python visualize_instance_masks.py \
    evaluation_dataset/gt_instances/ \
    output_vis_dir/
```

### 正しく生成されていることの確認

以下を確認してください：

1. **アノテーションファイルに吹き出しがある**
   ```bash
   cat evaluation_dataset/gt_annotations/ARMS_017.json | grep "\"balloons\""
   ```

2. **インスタンスマスクにユニークな値がある**
   ```bash
   python3 -c "import numpy as np; from PIL import Image; \
   mask = np.array(Image.open('evaluation_dataset/gt_instances/ARMS_017.png')); \
   print(f'Unique values: {len(np.unique(mask)) - 1} instances')"
   ```

3. **二値マスクに白い領域がある**
   ```bash
   python3 -c "import numpy as np; from PIL import Image; \
   mask = np.array(Image.open('evaluation_dataset/gt_masks/ARMS_017.png')); \
   print(f'White pixels: {np.sum(mask > 0)} / {mask.size} pixels')"
   ```

## 評価での使用

Track B（インスタンスレベル評価）では、このインスタンスマスクを使用して：
- 各吹き出しを個別に識別
- 予測インスタンスとGTインスタンスをマッチング（Hungarian algorithm）
- AP, PQ, F1などの指標を計算

インスタンスマスクのピクセル値が小さいことは、評価には全く影響しません。

## まとめ

✅ インスタンスマスクは正しく生成されています  
✅ 「真っ黒」に見えるのは、ピクセル値が0〜N（小さい値）のため  
✅ 評価には問題なく使用できます  
✅ 可視化する場合は `visualize_instance_masks.py` を使用してください
