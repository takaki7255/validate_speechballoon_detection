# 漫画吹き出し抽出手法の比較評価システム

> **プロジェクト目的**: Manga109データセットを使用して、吹き出し抽出手法（ルールベース vs U-Net等の機械学習）を精度比較するための統合評価フレームワーク

---

## 📋 プロジェクト概要

このプロジェクトは、漫画の吹き出し（speech balloon）を抽出する複数の手法を、統一タフレームワークで比較評価します。

### 🎯 主な特徴

- **統合評価フレームワーク**: 複数の手法を同じデータセットで公平に評価可能
- **2つの評価トラック**:
  - **Track A（画素レベル）**: セマンティックセグメンテーション指標（IoU, Dice, Boundary F-score等）
  - **Track B（インスタンスレベル）**: 個体検出指標（AP@[.50:.95], F1, PQ等）
- **完全な自動パイプライン**: データセット作成から評価まで一括実行可能
- **COCO形式対応**: Manga109のRLEセグメンテーション形式に完全対応

### 📊 対応する評価手法

現在実装済み：
- ✅ **ルールベース手法**（C++ 実装、研究室の既存手法）
- 🔄 **U-Net等の機械学習手法**（フレームワーク構築済み、`test_unet.py` 参照）

---

## 🏗️ システムアーキテクチャ

```
データセット準備
    ↓
  Manga109 (images + COCOアノテーション)
    ↓
[create_evaluation_dataset_with_masks.py]
    ↓
evaluation_dataset/
├── sampled_images/         ← ランダムサンプリング画像
├── gt_annotations/         ← RLEセグメンテーション付きGT
├── gt_masks/              ← Track A用二値マスク
└── gt_instances/          ← Track B用インスタンスマスク（GT）
    ↓
┌─────────────────────────────────────────────────────────┐
│         [検出手法の予測実行]                              │
├─────────────────────────────────────────────────────────┤
│  ルールベース                        │ U-Net等
│  [manga_processor.py]               │ [カスタム実装]
│  ↓                                  │ ↓
│  predictions/eval_masks/           │ predictions_ml/
│  predictions/eval_instances/       │ predictions_ml/instances/
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│           [評価指標計算]                                  │
├─────────────────────────────────────────────────────────┤
│  Track A                │  Track B
│  (画素レベル)            │ (インスタンスレベル)
│  [evaluate_track_a.py]  │ [evaluate_track_b.py]
└─────────────────────────────────────────────────────────┘
    ↓
results/
├── track_a_per_image.csv       ← 画像ごとの詳細結果
├── track_a_statistics.json     ← 全体統計（IoU, Dice等）
├── track_b_per_image.csv       ← インスタンスマッチング結果
└── track_b_statistics.json     ← 全体統計（AP, F1等）
```

---

## 🚀 クイックスタート

### 前提条件

```bash
# Python 3.8以上が必要
python3 --version

# 必要なパッケージをインストール
pip install -r requirements.txt
```

### ステップ1: 評価用データセットの作成

Manga109データセットからランダムに100枚の吹き出し含有画像をサンプリングし、 GT（Ground Truth）マスクとアノテーションを生成します。

```bash
python create_evaluation_dataset_with_masks.py \
    ../Manga109_released_2023_12_07/images \
    ../Manga109_released_2023_12_07/manga_seg_jsons \
    ./evaluation_dataset \
    100  # サンプリング数
```

**出力結果:**
```
evaluation_dataset/
├── sampled_images/      (100枚の画像)
├── gt_annotations/      (RLEセグメンテーション付き COCO JSON)
├── gt_masks/           (Track A用の二値マスク)
├── gt_instances/       (Track B用のインスタンスマスク)
└── dataset_summary.json (統計情報)
```

### ステップ2: 評価パイプラインの実行

ルールベース手法の評価（データセット作成済みの場合）：

```bash
bash complete_evaluation_pipeline.sh
```

このスクリプトが実行する内容：
1. **Proxy-Instance生成**: GT二値マスクから疑似インスタンスを生成
2. **予測実行**: `manga_processor.py` でルールベース手法を実行
3. **Track A評価**: 画素レベルの指標を計算
4. **Track B評価**: インスタンスレベルの指標を計算

### ステップ3: 結果の確認

```bash
# Track A（画素レベル）の結果
cat results/track_a_statistics.json | python -m json.tool

# Track B（インスタンスレベル）の結果
cat results/track_b_statistics.json | python -m json.tool

# 画像ごとの詳細
cat results/track_a_per_image.csv
cat results/track_b_per_image.csv
```

---

## 📁 ファイル構成

### 🔧 コアスクリプト（評価フレームワーク）

| ファイル | 役割 | 説明 |
|---------|------|------|
| **create_evaluation_dataset_with_masks.py** | データ準備 | Manga109からサンプリング→GT生成（マスク＋アノテーション） |
| **proxy_instance_generator.py** | GT生成補助 | 二値マスク → 疑似インスタンスマスク変換（Track B用） |
| **evaluate_track_a.py** | Track A評価 | 画素レベル指標（IoU, Dice, Boundary F-score等） |
| **evaluate_track_b.py** | Track B評価 | インスタンスレベル指標（AP@[.50:.95], F1, PQ等） |
| **complete_evaluation_pipeline.sh** | 統合スクリプト | 一括評価パイプライン |

### 🔬 検出手法実装

| ファイル | 説明 |
|---------|------|
| **manga_processor.py** | **ルールベース手法（Python実装）**<br>C++実装をPythonで再現。ページ分割→ページ分類→フレーム検出→吹き出し検出 |
| **src/** | **ルールベース手法（C++元実装）**<br>研究室サーバの既存コード。複数ファイル分割型。 |
| **test_unet.py** | **U-Net等のテンプレート**<br>新しい機械学習手法を追加する際のスケルトン |

### 🛠️ ユーティリティ

| ファイル | 説明 |
|---------|------|
| **generate_mask.py** | 汎用マスク生成ツール |
| **visualize_instance_masks.py** | インスタンスマスク可視化ツール |

### 📚 ドキュメント

| ファイル | 内容 |
|---------|------|
| **EVALUATION.md** | 評価手法の詳細説明（指標定義、アルゴリズム） |
| **INSTANCE_MASKS_README.md** | インスタンスマスク生成・確認方法 |
| **INSTANCE_MASK_OVERLAP.md** | インスタンスマスク重複の扱い方 |

### 📂 出力ディレクトリ構造

```
evaluation_dataset/          ← ステップ1の出力（GT）
├── sampled_images/         （元画像）
├── gt_annotations/         （RLE形式の吹き出しアノテーション）
├── gt_masks/              （Track A用二値マスク）
├── gt_instances/          （Track B用疑似インスタンス）
└── dataset_summary.json

rulebased_predictions/       ← ルールベース手法の予測結果
├── eval_masks/            （検出結果の二値マスク）
└── eval_instances/        （検出インスタンスマスク）

predictions_python/          ← Python実装版の結果
predictions_origin/          ← オリジナル版の結果  
predictions_cpp/             ← C++版の結果

results/                      ← 最終評価結果
├── track_a_per_image.csv
├── track_a_statistics.json
├── track_b_per_image.csv
└── track_b_statistics.json

results_python_track/         ← Python実装の評価結果
results_cpp_track/            ← C++版の評価結果
results_origin_track/         ← オリジナル版の評価結果
```

---

## 📊 評価トラックの詳細

### Track A: 画素レベル評価
セマンティックセグメンテーションとして評価

**指標:**
- IoU (Intersection over Union)
- Dice係数
- Pixel Precision/Recall
- Boundary F-score

### Track B: インスタンスレベル評価
個々の吹き出し検出精度を評価

**指標:**
- AP@[.50:.95] (COCO形式)
- F1@0.50, F1@0.75
- PQ (Panoptic Quality)
- Count MAE（個数誤差）

## 評価結果の比較

### 原手法（ルールベース）の評価結果

**Track A（ピクセルレベル）:**
- IoU: **51.57%** ± 27.08%
- Dice係数: **63.37%** ± 26.69%
- Boundary F-score: **57.89%** ± 24.82%

**Track B（インスタンスレベル）:**
- AP (IoU 0.50:0.95): **52.69%**
- AP@0.50: **55.97%**
- AP@0.75: **52.86%**
- F1-score@0.50: **55.97%** ± 23.76%
- PQ (Panoptic Quality): **53.03%** ± 23.10%
- Count MAE: **4.67個** ± 4.09

評価データ: Manga109から100枚のサンプル画像

**結果ファイル:**
- `results_origin_track/track_a_statistics.json`
- `results_origin_track/track_b_statistics.json`
- `results_origin_track/track_a_per_image.csv`
- `results_origin_track/track_b_per_image.csv`

### U-Net手法との比較（予定）

U-Netベースの深層学習手法との比較評価を実施予定。以下の指標を用いて比較：

**主要比較指標:**
- AP@0.50, AP@0.75（インスタンス検出精度）
- Dice係数（ピクセルレベル精度）
- F1-score@0.50（検出・認識の総合精度）
- Count MAE（検出個数の精度）

**比較の観点:**
1. **精度**: 各評価指標での性能比較
2. **ロバスト性**: 標準偏差の比較（様々な漫画スタイルへの対応力）
3. **推論速度**: 処理時間の比較
4. **誤検出パターン**: 失敗ケースの分析

U-Net評価結果は`results_unet_track/`ディレクトリに保存予定。

## アノテーションJSONフォーマット

```json
{
  "title": "漫画タイトル",
  "page": "001.jpg",
  "image_width": 1654,
  "image_height": 1170,
  "balloons": [
    {
      "id": "12345",
      "xmin": 123,
      "ymin": 456,
      "xmax": 789,
      "ymax": 1011,
      "width": 666,
      "height": 555,
      "area": 369630,
      "segmentation": {
        "size": [1170, 1654],
        "counts": "RLE_encoded_string..."
      }
    }
  ]
}
```

**新機能**: `segmentation`フィールドにRLE形式のセグメンテーション情報が含まれるため、
元のCOCO JSONファイルを参照せずに正確なマスクを生成可能。

---

## 🔧 実装ノート

### ルールベース手法の実装

**C++実装（元実装）**: `src/` ディレクトリ
- `main.cpp`: メインパイプライン
- `frame_separation.hpp/cpp`: ページからコマフレームを抽出
- `speechballoon_separation.hpp/cpp`: フレーム内の吹き出しを分割
- `blackpage_framedetect.hpp`: 黒いページ判定
- `page_classification.hpp`: ページ分類
- `twopage_to_onepage.hpp`: 見開きを1ページに変換

**Python再実装**：`manga_processor.py`
- C++ の実装をPythonで再実装
- 同じアルゴリズムロジック、同じパラメータを使用
- OpenCV, NumPy, SciPyで実装

**構築方法:**
```bash
# C++版の構築
bash build_origin.sh      # オリジナル版
bash build_cpp.sh         # C++版（同等品）
bash build_origin_eval.sh # 評価用特殊版
```

### Proxy-Instance生成アルゴリズム

二値マスクから疑似インスタンスマスクへの変換：

1. **前処理**
   - 穴埋め（面積閾値: 500px以下の穴を埋める）
   - ノイズ除去（100px未満の連結成分を削除）

2. **分割（Watershed + 距離変換）**
   - 距離変換を計算
   - 局所極大を検出（最小距離: 10px）
   - Watershedセグメンテーション

3. **後処理**
   - 過分割の統合（IoU > 0.3で統合）
   - 極小片の統合（面積比 < 0.1で統合）
   - 細長い領域の統合（細長さ > 5.0で統合）

**パラメータ公開**で公平性を保証：
```python
PARAMS = {
    'min_area_threshold': 100,
    'hole_fill_area_threshold': 500,
    'distance_sigma': 1.5,
    'min_distance': 10,
    'watershed_compactness': 0.001,
    'merge_iou_threshold': 0.3,
    'merge_area_ratio_threshold': 0.1,
    'elongation_threshold': 5.0,
}
```

### 評価指標の計算

**Track A（画素レベル）**:
```
IoU = |予測 ∩ GT| / |予測 ∪ GT|
Dice = 2|予測 ∩ GT| / (|予測| + |GT|)
Boundary F-score: 境界画素のみで計算（距離: ±2px）
```

詳細は [EVALUATION.md](EVALUATION.md) を参照。

**Track B（インスタンスレベル）**:
- Hungarian Matching: 予測インスタンスとGTインスタンスを1対1で対応
- IoU閾値で判定（AP計算時は0.50:0.05:0.95）
- Panoptic Quality: セグメンテーション品質の総合指標

---

## 📈 今後の拡張・改善ポイント

### ✅ 完了済み
- ✅ ルールベース手法による評価フレームワーク構築
- ✅ Track A, Track B の両評価トラック実装
- ✅ COCO形式対応とRLEセグメンテーション対応
- ✅ Manga109での実装と検証

### 🔄 進行中・検討中
- 🔄 **U-Net手法の実装と評価**
  - `test_unet.py` をテンプレートとして使用
  - 同じ評価フレームワークで比較可能
  - 予定: PyTorch or TensorFlow ベース実装

### 📋 推奨される次のステップ

1. **機械学習手法の追加**
   - U-Net, Mask R-CNN, SegFormer など
   - `test_unet.py` から `YourMLMethod.predict()` を実装
   - `evaluate_track_a.py`, `evaluate_track_b.py` はそのまま使用可

2. **精度向上の検討**
   - 誤検出パターン分析（`results/track_b_per_image.csv` で失敗ケースを特定）
   - パラメータチューニング
   - 異なるプリ/ポストプロセッシング

3. **拡張評価**
   - 複数のデータセット（不同テスト）での評価
   - マンガスタイル別の詳細分析
   - 推論速度ベンチマーク

4. **可視化・分析ツール**
   - ` visualize_instance_masks.py` の拡張
   - 失敗例の可視化ツール
   - 精度比較チャート生成スクリプト

5. **ドキュメント化**
   - 各手法の詳細なパラメータドキュメント
   - 結果の論文化

---

## 🔍 トラブルシューティング

### エラー: "images_root does not exist"
```
解決: Manga109の正しいパスを確認してください
例: /path/to/Manga109_released_2023_12_07/images
```

### 警告: "Bbox fallback"
```
原因: RLEセグメンテーションが見つからない
解決: 新しい create_evaluation_dataset_with_masks.py で
     データセットを再生成してください
```

### インスタンスマスクが「真っ黒」に見える
```
原因: インスタンスID (0-20など) が低値なため、
     通常の画像ビューアで黒く見える（仕様）
確認方法: INSTANCE_MASKS_README.md を参照
```

### Track B評価でInstanceMatchingエラー
```
確認: evaluation_dataset/proxy_instances/ が
     正しく生成されているか確認
     python INSTANCE_MASKS_README.md の方法で検証
```

### メモリ不足
```
大量のインスタンスを扱う場合、バッチ処理を検討してください
スクリプトに --batch_size オプションを追加する方法もあります
```

---

## 📚 参考資料

### 関連ドキュメント
- [EVALUATION.md](EVALUATION.md) - 評価手法の詳細説明
- [INSTANCE_MASKS_README.md](INSTANCE_MASKS_README.md) - インスタンスマスクの生成・確認
- [INSTANCE_MASK_OVERLAP.md](INSTANCE_MASK_OVERLAP.md) - インスタンス重複の扱い方

### 外部参考資料
- **Manga109**: https://www.manga109.org/
- **COCO形式**: https://cocodataset.org/#format-data
- **panopticapi**: https://github.com/cocodataset/panopticApi
- **pycocotools**: https://github.com/cocodataset/cocoapi

---

## 👥 プロジェクト情報

**実装期間**: 2023年12月 ～ 2024年
**対応データセット**: Manga109 (2023年12月版)
**評価フレームワーク**: COCO形式の指標に準拠
**主な参考手法**: Mask R-CNN, Panoptic Segmentation

---

## 📝 ライセンス

このプロジェクトはManga109データセットの利用規約に従って使用してください。
詳細は [Manga109公式サイト](https://www.manga109.org/) を参照してください。
