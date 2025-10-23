#!/bin/bash
# complete_evaluation_pipeline.sh
# 
# ルールベース手法（manga_processor.py）の完全評価パイプライン
# evaluation_datasetを使用

set -e  # エラーで停止

echo "=========================================="
echo "Balloon Detection Evaluation Pipeline"
echo "=========================================="
echo ""
echo "Note: Run create_evaluation_dataset_with_masks.py first to generate dataset"
echo ""

# ステップ1: Proxy-Instance生成（Track B用）
# Note: GT masksは既にcreate_evaluation_dataset_with_masks.pyで生成済み
echo "Step 1/4: Generating proxy instances for Track B..."
mkdir -p evaluation_dataset/proxy_instances

# 各GTマスクに対してproxy instanceを生成
count=0
total=$(ls evaluation_dataset/gt_masks/*.png | wc -l | tr -d ' ')

for f in evaluation_dataset/gt_masks/*.png; do
    name=$(basename "$f" .png)
    count=$((count + 1))
    
    # 進捗表示（10枚ごと）
    if [ $((count % 10)) -eq 0 ]; then
        echo "  Processed $count/$total proxy instances..."
    fi
    
    # Proxy instance生成（エラー出力を抑制）
    python proxy_instance_generator.py "$f" "evaluation_dataset/proxy_instances/" > /dev/null 2>&1
    
    # 生成されたインスタンスマスクをリネーム
    if [ -f "evaluation_dataset/proxy_instances/proxy_instances.png" ]; then
        mv "evaluation_dataset/proxy_instances/proxy_instances.png" \
           "evaluation_dataset/proxy_instances/${name}.png"
    fi
done

echo "  Completed $total proxy instances."
echo ""

# ステップ2: ルールベース手法で予測
echo "Step 2/4: Running rule-based balloon detection..."
python manga_processor.py \
    ./evaluation_dataset/sampled_images \
    ./rulebased_predictions \
    --eval

echo ""

# ステップ3: Track A評価（画素レベル）
echo "Step 3/4: Evaluating Track A (pixel-level metrics)..."
python evaluate_track_a.py \
    ./rulebased_predictions/eval_masks \
    ./evaluation_dataset/gt_masks \
    ./results/rulebased_track_a

echo ""

# ステップ4: Track B評価（インスタンスレベル）
echo "Step 4/4: Evaluating Track B (instance-level metrics)..."
python evaluate_track_b.py \
    ./rulebased_predictions/eval_instances \
    ./evaluation_dataset/proxy_instances \
    ./results/rulebased_track_b

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Track A (pixel-level):    ./results/rulebased_track_a/"
echo "  Track B (instance-level): ./results/rulebased_track_b/"
echo ""
echo "Key files:"
echo "  - track_a_per_image.csv      (per-image results)"
echo "  - track_a_statistics.json    (overall statistics)"
echo "  - track_b_per_image.csv      (per-image results)"
echo "  - track_b_statistics.json    (overall statistics)"
echo ""
