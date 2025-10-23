# 吹き出し抽出手法の評価システム

Manga109データセットを使用して、吹き出し抽出手法（ルールベース vs U-Net）を比較評価するためのツール群です。

## 概要

このシステムは以下を行います：
1. **評価用データセット作成**: Manga109から吹き出しを含む画像をランダムサンプリング（画像・アノテーション・マスクを一括生成）
2. **2トラック評価**: 画素レベル（Track A）とインスタンスレベル（Track B）で精度を測定
3. **自動評価パイプライン**: Proxy-Instance生成から評価まで一括実行

**対応フォーマット**: COCO形式のManga109アノテーション（category_id=5がballoon、RLEセグメンテーション付き）

## クイックスタート

### 1. 評価用データセット作成（全て一括）

```bash
python create_evaluation_dataset_with_masks.py \
    ../Manga109_released_2023_12_07/images \
    ../Manga109_released_2023_12_07/manga_seg_jsons \
    ./evaluation_dataset \
    100
```

**出力:**
```
evaluation_dataset/
├── sampled_images/      # サンプリングされた画像（吹き出し付きのみ）
├── gt_annotations/      # アノテーションJSON（RLEセグメンテーション付き）
├── gt_masks/           # 二値マスク（Track A用、自動生成）
├── gt_instances/       # インスタンスマスク（Track B用、自動生成）
└── dataset_summary.json # サマリー情報
```

### 2. 評価パイプライン実行

```bash
bash complete_evaluation_pipeline.sh
```

このスクリプトは以下を自動実行します：
1. Proxy-Instance生成（Track B用GT）
2. ルールベース手法で予測
3. Track A評価（画素レベル）
4. Track B評価（インスタンスレベル）

### 3. 結果確認

```
results/
├── rulebased_track_a/
│   ├── track_a_per_image.csv      # 画像ごとの詳細結果
│   └── track_a_statistics.json    # 全体統計
└── rulebased_track_b/
    ├── track_b_per_image.csv
    └── track_b_statistics.json
```

## 前提条件

- Python 3.6以上
- Manga109データセット（2023年12月版）

### 必要なパッケージ

```bash
pip install opencv-python numpy scipy scikit-image scikit-learn matplotlib pycocotools
```

または：

```bash
pip install -r requirements.txt
```

## 詳細ドキュメント

評価手法の詳細は[EVALUATION.md](EVALUATION.md)を参照してください。

## ファイル構成

### 主要スクリプト

| ファイル | 説明 |
|---------|------|
| `create_evaluation_dataset_with_masks.py` | 評価データセット一括作成（画像・アノテーション・マスク） |
| `proxy_instance_generator.py` | GT二値マスクから疑似インスタンスGTを生成 |
| `manga_processor.py` | ルールベース吹き出し抽出（`--eval`フラグで評価用出力） |
| `evaluate_track_a.py` | 画素レベル評価（IoU, Dice, Boundary F-score等） |
| `evaluate_track_b.py` | インスタンスレベル評価（AP, PQ, F1等） |
| `complete_evaluation_pipeline.sh` | 評価パイプライン一括実行スクリプト |

### その他

- `generate_mask.py`: 汎用的なマスク生成ユーティリティ（評価システムとは独立）
- `test_unet.py`: U-Net手法のテスト用スクリプト

## 評価トラック

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

## トラブルシューティング

### エラー: "images_root does not exist"
指定したパスが正しいか確認してください。

### 警告: "Bbox fallback"
RLEセグメンテーション情報がない場合、バウンディングボックスから矩形マスクを生成します。
新しい`create_evaluation_dataset_with_masks.py`を使用すれば、この警告は出ません。

### データセットの再生成
異なるランダムシードで再生成したい場合は、スクリプト内で`random.seed()`を設定してください。

## ライセンス

このツールはManga109データセットの利用規約に従って使用してください。
