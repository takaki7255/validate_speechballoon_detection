# インスタンスマスク生成の問題と解決策

## 問題: 重複するインスタンス

### 状況
COCO形式のアノテーションで、複数の吹き出しセグメンテーションが重複する場合があります。

例:
- アノテーション: 27個の吹き出し
- インスタンスマスク: 25個のインスタンス
- **2個のインスタンスが失われている**

### 原因

#### 従来の方法（上書き方式）
```python
for idx, balloon in enumerate(balloons):
    mask = decode_rle(balloon['segmentation'])
    instance_mask[mask > 0] = idx  # 重複部分は上書きされる
```

**問題点**: 重複する領域は後のIDで上書きされ、前のインスタンスが失われる

## 解決策の比較

### 1. 面積優先方式（現在の実装） ⭐

```python
# 面積が大きい順にソート
balloon_data.sort(key=lambda x: x['area'], reverse=True)

# 未割り当て領域にのみIDを設定
for data in balloon_data:
    unassigned = (instance_mask == 0) & (mask > 0)
    instance_mask[unassigned] = data['id']
```

**利点**:
- 面積が大きいインスタンスを優先的に保持
- すべてのインスタンスが何らかの領域を持つ

**欠点**:
- 重複部分は1つのインスタンスにしか割り当てられない
- 小さいインスタンスが大きく侵食される可能性

### 2. 輪郭収縮方式

```python
# 各インスタンスの輪郭を検出し、重複部分を収縮
for mask in individual_masks:
    # 重複領域を検出
    overlap = (instance_mask > 0) & (mask > 0)
    # 重複部分を収縮（erosion）
    if overlap.any():
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel)
```

**利点**:
- すべてのインスタンスを保持
- 重複を最小限に

**欠点**:
- 形状が変わる
- GTとしての正確性が低下

### 3. 重複領域を最近傍インスタンスに割り当て

```python
# 距離変換を使用
for overlap_pixel in overlap_region:
    # 各インスタンスの中心からの距離を計算
    # 最も近いインスタンスに割り当て
```

**利点**:
- より自然な分割
- 形状の歪みが少ない

**欠点**:
- 計算コストが高い
- 実装が複雑

### 4. 重複を許容（マルチラベル）

インスタンスマスクを3次元配列にする：
```python
instance_masks = np.zeros((height, width, max_instances), dtype=bool)
for idx, balloon in enumerate(balloons):
    mask = decode_rle(balloon['segmentation'])
    instance_masks[:, :, idx] = mask
```

**利点**:
- すべての情報を完全に保持
- GTとして最も正確

**欠点**:
- ファイルサイズが大きい
- 既存の評価ツールとの互換性がない
- Track Bの評価方法を大幅に変更する必要がある

## 推奨アプローチ

### 現状の実装: 面積優先方式

COCOアノテーション自体が「重複がある場合は曖昧」という性質を持っているため、
**面積優先方式**が最も実用的です。

#### 理由:

1. **公平性**: 大きいインスタンスほど重要と仮定するのは合理的
2. **単純性**: 実装がシンプルで理解しやすい
3. **再現性**: 同じアノテーションから常に同じ結果が得られる
4. **互換性**: 既存のTrack B評価ツールとそのまま使用可能

#### 注意点:

- **評価時に考慮**: 重複が多い画像では評価が難しくなる可能性
- **統計を記録**: どのくらいの重複が発生しているか記録する

## 重複の分析

評価データセット作成時に、重複統計を出力することを推奨：

```python
# アノテーション数とインスタンスマスク数の比較
num_balloons = len(balloons)
num_instances = len(np.unique(instance_mask)) - 1
overlap_count = num_balloons - num_instances

if overlap_count > 0:
    print(f"Warning: {overlap_count} overlapping instances detected")
```

## 結論

**現在の実装（面積優先）は適切です。**

ただし、以下を追加することを推奨：
1. 重複統計の記録
2. ドキュメントに重複の可能性を明記
3. 評価時に重複が多い画像を特定できるようにする
