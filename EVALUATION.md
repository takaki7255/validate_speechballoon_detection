## 吹き出し抽出手法の評価システム

### 概要

このディレクトリには、吹き出し抽出手法の比較評価を行うためのスクリプトが含まれています。

**2つの評価トラック:**

- **Track A (画素レベル)**: セマンティックセグメンテーションとしての評価
- **Track B (インスタンスレベル)**: 吹き出し個体ごとの検出・分割精度の評価

### 評価フロー

```
0. 評価用データセット作成（画像・アノテーション・マスク一括生成）
   ↓
1. Proxy-Instance生成 (Track B用)
   ↓
2. 予測実行 (ルールベース or U-Net)
   ↓
3. 評価
   ├─ Track A: 画素レベル指標
   └─ Track B: インスタンスレベル指標
```

---

## 必要なパッケージ

```bash
pip install opencv-python numpy scipy scikit-image scikit-learn matplotlib pycocotools
```

---

## 0. 評価用データセット作成

Manga109から吹き出しを含む画像をランダムサンプリングし、アノテーション（RLEセグメンテーション付き）と
評価用マスク（Track A二値マスク、Track Bインスタンスマスク）を一括生成します。

### 使い方

```bash
python create_evaluation_dataset_with_masks.py \
    ../Manga109_released_2023_12_07/images \
    ../Manga109_released_2023_12_07/manga_seg_jsons \
    ./evaluation_dataset \
    100
```

### 出力

```
evaluation_dataset/
├── sampled_images/      # サンプリングされた画像（100枚）
├── gt_annotations/      # アノテーションJSON（RLEセグメンテーション付き）
├── gt_masks/           # 二値マスク（Track A用、自動生成）
├── gt_instances/       # インスタンスマスク（Track B用、自動生成）
└── dataset_summary.json # サマリー情報
```

**特徴:**
- 吹き出しが1つ以上含まれる画像のみを選択
- COCO形式のRLEセグメンテーションをアノテーションに含める
- GT二値マスクとインスタンスマスクを同時に生成
- 1回のコマンドで評価準備が完了

---

## 1. Proxy-Instance生成

GTの二値マスクから疑似的なインスタンスGTを生成します。

### 使い方

```bash
python proxy_instance_generator.py <gt_mask.png> <output_dir>
```

### 例

```bash
# 単一画像
python proxy_instance_generator.py ./gt_masks/image001.png ./proxy_instances/

# バッチ処理用スクリプト例
for f in gt_masks/*.png; do
    name=$(basename "$f" .png)
    python proxy_instance_generator.py "$f" "./proxy_instances/$name/"
done
```

### 出力

- `proxy_instances.png`: インスタンスラベルマップ (0=背景, 1,2,3...=各インスタンス)
- `proxy_instances.json`: インスタンス情報 (bbox, area等)
- `proxy_generator_params.json`: 使用したパラメータ（再現性のため）

### パラメータ（固定・公開）

```json
{
  "min_area_threshold": 100,
  "hole_fill_area_threshold": 500,
  "distance_sigma": 1.5,
  "min_distance": 10,
  "watershed_compactness": 0.001,
  "merge_iou_threshold": 0.3,
  "merge_area_ratio_threshold": 0.1,
  "elongation_threshold": 5.0
}
```

---

## 2. Track A: 画素レベル評価

### 指標

| 指標 | 説明 |
|------|------|
| **IoU** | Jaccard係数 (Intersection over Union) |
| **Dice** | Dice係数 (F1 score of pixels) |
| **Pixel Precision** | 正解率（予測した吹き出し画素のうち正解の割合） |
| **Pixel Recall** | 再現率（GT吹き出し画素のうち予測できた割合） |
| **Boundary F-score** | 境界精度（半径2pxの許容範囲） |

### 使い方

```bash
python evaluate_track_a.py <pred_masks_dir> <gt_masks_dir> <output_dir>
```

### 例

```bash
# ルールベースの評価
python evaluate_track_a.py \
    ./rulebased_predictions/eval_masks \
    ./evaluation_dataset/gt_masks \
    ./results/rulebased_track_a

# U-Netの評価
python evaluate_track_a.py \
    ./unet_predictions/masks \
    ./evaluation_dataset/gt_masks \
    ./results/unet_track_a
```

### 入力フォーマット

- **予測マスク**: 二値画像 (0=背景, 255=吹き出し)
- **GTマスク**: 二値画像 (0=背景, 255=吹き出し) - `create_evaluation_dataset_with_masks.py`で自動生成済み
- ファイル名は対応している必要があります

### 出力

- `track_a_per_image.csv`: 画像ごとの詳細結果
- `track_a_statistics.json`: 全体統計（平均、標準偏差等）

---

## 3. Track B: インスタンスレベル評価
| **Pixel Precision/Recall** | 画素レベルの適合率・再現率 |
| **Boundary F-score** | 境界の忠実度（推奨） |

### 使い方

```bash
python evaluate_track_a.py <pred_masks_dir> <gt_masks_dir> <output_dir>
```

### 例

```bash
# ルールベースの評価
python evaluate_track_a.py \
    ./rulebased_predictions/masks \
    ./gt_masks \
    ./results/rulebased_track_a

# U-Netの評価
python evaluate_track_a.py \
    ./unet_predictions/masks \
    ./gt_masks \
    ./results/unet_track_a
```

### 入力フォーマット

- **予測マスク**: 二値画像 (0=背景, 255=吹き出し)
- **GTマスク**: 二値画像 (0=背景, 255=吹き出し)
- ファイル名は対応している必要があります

### 出力

- `track_a_per_image.csv`: 画像ごとの詳細結果
- `track_a_statistics.json`: 全体統計（平均、標準偏差等）

---

## 3. Track B: インスタンスレベル評価

### 指標

| 指標 | 説明 |
|------|------|
| **AP@[.50:.95]** | Average Precision (IoU閾値0.50〜0.95の平均) |
| **F1@0.50** | IoU閾値0.50でのF1スコア |
| **F1@0.75** | IoU閾値0.75でのF1スコア |
| **PQ** | Panoptic Quality (SQ × RQ) |
| **Count MAE** | 個数誤差の平均絶対誤差 |

### 使い方

```bash
python evaluate_track_b.py <pred_instances_dir> <gt_instances_dir> <output_dir>
```

### 例

```bash
# ルールベースの評価
python evaluate_track_b.py \
    ./rulebased_predictions/instances \
    ./proxy_instances \
    ./results/rulebased_track_b

# U-Netの評価（事前にインスタンス化が必要）
python evaluate_track_b.py \
    ./unet_predictions/instances \
    ./proxy_instances \
    ./results/unet_track_b
```

### 入力フォーマット

- **インスタンスラベルマップ**: グレースケール画像
  - 0 = 背景
  - 1, 2, 3, ... = 各インスタンスID
- ファイル名は対応している必要があります

### 出力

- `track_b_per_image.csv`: 画像ごとの詳細結果
- `track_b_statistics.json`: 全体統計

---

## 4. 完全な評価パイプライン例

### クイックスタート（evaluation_datasetを使用）

**ワンコマンドで完全評価:**

```bash
bash complete_evaluation_pipeline.sh
```

このスクリプトは以下を自動実行します：
1. GTマスク生成
2. Proxy-Instance生成
3. ルールベース手法で予測
4. Track A評価
5. Track B評価

**または、手動で段階的に実行:**

```bash
#!/bin/bash

# ステップ1: GTマスク生成（Track A用）
echo "Step 1: Generating GT masks for Track A..."
python generate_gt_masks.py \
    ./evaluation_dataset/gt_annotations \
    ./evaluation_dataset/gt_masks

# ステップ2: Proxy-Instance生成（Track B用）
echo "Step 2: Generating proxy instances for Track B..."
mkdir -p evaluation_dataset/proxy_instances
for f in evaluation_dataset/gt_masks/*.png; do
    name=$(basename "$f" .png)
    python proxy_instance_generator.py "$f" "evaluation_dataset/proxy_instances/" 2>/dev/null
    mv "evaluation_dataset/proxy_instances/proxy_instances.png" \
       "evaluation_dataset/proxy_instances/${name}.png"
done

# ステップ3: ルールベース手法で予測（評価モード）
echo "Step 3: Running rule-based predictions..."
python manga_processor.py \
    ./evaluation_dataset/sampled_images \
    ./rulebased_predictions \
    --eval

# ステップ4: Track A評価
echo "Step 4: Evaluating Track A (pixel-level)..."
python evaluate_track_a.py \
    ./rulebased_predictions/eval_masks \
    ./evaluation_dataset/gt_masks \
    ./results/rulebased_track_a

# ステップ5: Track B評価
echo "Step 5: Evaluating Track B (instance-level)..."
python evaluate_track_b.py \
    ./rulebased_predictions/eval_instances \
    ./evaluation_dataset/proxy_instances \
    ./results/rulebased_track_b

echo "Evaluation complete!"
echo "Results saved to ./results/"
```

### ルールベース手法（manga_processor.py）の使い方

**通常モード（パネル分割あり）:**
```bash
python manga_processor.py <input_folder> <output_folder>
```

**評価モード（--eval フラグ）:**
```bash
python manga_processor.py <input_folder> <output_folder> --eval
```

評価モードでは以下が出力されます：
- `eval_masks/`: Track A用の二値マスク（全吹き出しを統合）
- `eval_instances/`: Track B用のインスタンスラベルマップ（各吹き出しに異なるID）

### フル評価パイプライン（カスタムデータ用）

```bash
#!/bin/bash
# complete_evaluation.sh

# ステップ1: Proxy-Instance生成
echo "Generating proxy instances from GT masks..."
mkdir -p proxy_instances
for f in evaluation_dataset/gt_masks/*.png; do
    name=$(basename "$f" .png)
    python proxy_instance_generator.py "$f" "proxy_instances/$name/"
done

# ステップ2: ルールベース手法で予測（manga_processor.py使用）
echo "Running rule-based method..."
# (manga_processor.pyを使って予測を生成)

# ステップ3: U-Net手法で予測
echo "Running U-Net method..."
# (U-Netモデルで予測を生成)

# ステップ4: Track A評価
echo "Evaluating Track A..."
python evaluate_track_a.py \
    ./rulebased_predictions/masks \
    ./evaluation_dataset/gt_masks \
    ./results/rulebased_track_a

python evaluate_track_a.py \
    ./unet_predictions/masks \
    ./evaluation_dataset/gt_masks \
    ./results/unet_track_a

# ステップ5: Track B評価
echo "Evaluating Track B..."
python evaluate_track_b.py \
    ./rulebased_predictions/instances \
    ./proxy_instances \
    ./results/rulebased_track_b

python evaluate_track_b.py \
    ./unet_predictions/instances \
    ./proxy_instances \
    ./results/unet_track_b

echo "Evaluation complete!"
```

---

## 5. 結果の解釈

### Track A (画素レベル)

- **IoU / Dice**: 全体的なピクセル一致度
  - 0.7以上: 良好
  - 0.5〜0.7: 中程度
  - 0.5未満: 要改善

- **Boundary F-score**: 境界の正確さ（細いテール等に敏感）
  - ルールベース手法は境界が正確な傾向
  - U-Netは滑らかだが境界がぼやける可能性

### Track B (インスタンスレベル)

- **PQ (Panoptic Quality)**:
  - SQ: 検出した吹き出しのセグメンテーション品質
  - RQ: 吹き出しの検出率

- **F1@0.50 vs F1@0.75**:
  - 0.50: 緩い基準（大まかに合っていればOK）
  - 0.75: 厳しい基準（正確な領域一致が必要）

- **Count MAE**: 個数の正確さ
  - ルールベースは過検出しやすい
  - U-Netは見逃しやすい

### 比較のポイント

1. **Track A (画素) vs Track B (インスタンス)**:
   - Track Aが高くTrack Bが低い → 領域は合っているが、個体分離が不正確
   - Track Bが高くTrack Aが低い → 個体は検出できているが、領域が不正確

2. **Proxy-Instanceの限界**:
   - 接触・合体した吹き出しの分離精度に依存
   - Track Aも併記して総合判断

3. **速度・メモリ**:
   - 実用性の観点で重要
   - 別途計測が必要

---

## 6. 論文での報告例

```
### 実験結果

| Method | IoU↑ | Dice↑ | BF-score↑ | PQ↑ | F1@0.50↑ | Count MAE↓ |
|--------|------|-------|-----------|-----|----------|------------|
| Rule-based | 0.72±0.15 | 0.81±0.12 | 0.76±0.14 | 0.68±0.18 | 0.74±0.16 | 2.3±1.8 |
| U-Net | 0.78±0.12 | 0.85±0.09 | 0.72±0.13 | 0.73±0.15 | 0.79±0.13 | 1.5±1.2 |

**Track A (画素レベル)**: U-Netが全体的に高いIoU/Diceを達成。
**Track B (インスタンス)**: U-NetがPQ・F1で優位。個数誤差も少ない。
**境界品質**: ルールベースがやや高いBF-scoreだが、U-Netも競合。

**Proxy-GTの妥当性**: 開発セット10枚の人手GTとの相関 r=0.89 (PQ)、
パラメータ±20%での感度分析で変動<5%を確認。
```

---

## 7. トラブルシューティング

### エラー: "No prediction files found"

- ディレクトリパスとファイル拡張子を確認
- `.png` ファイルが存在するか確認

### 警告: "GT not found for ..."

- 予測とGTのファイル名が一致しているか確認
- 両方のディレクトリに同じファイル名が存在するか確認

### メモリエラー

- 大きな画像の場合、バッチ処理を小分けにする
- 一度に評価する画像数を減らす

---

## 8. 参考文献

- **Boundary F-score**: Csurka et al., "What is a good evaluation measure for semantic segmentation?" BMVC 2013
- **Panoptic Quality**: Kirillov et al., "Panoptic Segmentation," CVPR 2019
- **Watershed**: Beucher & Meyer, "The morphological approach to segmentation: the watershed transformation," 1992

---

## ファイル一覧

```
validate_speechballoon_detection/
├── proxy_instance_generator.py    # Proxy-Instance生成器
├── evaluate_track_a.py            # Track A評価スクリプト
├── evaluate_track_b.py            # Track B評価スクリプト
├── generate_gt_masks.py           # GTアノテーションJSON→二値マスク変換
├── create_evaluation_dataset.py   # 評価データセット作成
├── manga_processor.py             # ルールベース手法
├── requirements.txt               # 必要パッケージ
├── README.md                      # データセット作成ドキュメント
└── EVALUATION.md                  # このファイル（評価手順）
```
