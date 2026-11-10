# photo2stamp

現地で撮った写真を、少ない色数のフラットなスタンプ風イラストに変換するCLIです。
画像生成AIは使わず、OpenCVの古典的な画像処理だけで動きます。GPUは不要で、800pxの写真1枚を数秒で処理します。

現在はPOC(概念実証)の段階です。写真の構図をそのまま残し、色と形を単純化します。
写真から要素を選んで配置し直すような意味的なデフォルメはしません。

## 変換例

左から順に、元写真、`teal`パレット(既定)、`warm`パレット、`mono`パレットを円形に切り抜いたものです。

| 元写真 | teal | warm | mono + circle |
|---|---|---|---|
| ![元写真](examples/photo.jpg) | ![teal](examples/photo_teal.png) | ![warm](examples/photo_warm.png) | ![mono circle](examples/photo_mono_circle.png) |

SVG出力の例は[examples/photo_teal.svg](examples/photo_teal.svg)にあります。

## インストール

```bash
git clone https://github.com/shiyow5/photo2stamp.git
cd photo2stamp
uv sync            # または: pip install -e .
```

## 使い方

```bash
uv run photo2stamp photo.jpg -o out --svg
# out/photo.png と out/photo.svg ができる

uv run photo2stamp *.jpg -o out --palette warm --shape circle   # 丸いスタンプ形・暖色
uv run photo2stamp photo.jpg -o out --palette original          # 写真の色を平坦化して使う
```

Pythonから呼ぶ場合の例です。

```python
import cv2
from photo2stamp import StampConfig, stylize

result = stylize(cv2.imread("photo.jpg"), StampConfig(palette="teal", colors=8))
cv2.imwrite("stamp.png", result.image)
open("stamp.svg", "w").write(result.svg)
```

## 処理の流れ

| 段階 | 処理 | モジュール |
|---|---|---|
| 1 | 長辺を 800px に縮小 | `pipeline.py` |
| 2 | バイラテラルフィルタと mean-shift で、輪郭を残したまま細部を消す | `segment.smooth` |
| 3 | Lab 色空間の k-means で色を 8 クラスタに分ける | `segment.quantize` |
| 4 | 多数決フィルタでラベルの境界をならす | `segment.majority_filter` |
| 5 | 彩度の高い小領域(看板や照明)を独立したクラスタとして拾う。文字のすき間は閉じて1枚の面にする | `segment.extract_accents` |
| 6 | 上端に接する明るい領域を空として扱う | `segment.extract_sky` |
| 7 | 明度の順に、面積で重み付けしてパレットの濃淡に割り当てる。空は空色、彩度の高い領域は色相が近いアクセント色にする | `palette.assign_colors` |
| 8 | 同じ色になったクラスタを統合し、小さな領域を隣の領域に吸収させる。アクセント色の領域は小さくても残す | `segment.merge_small_regions` |
| 9 | 輪郭を追跡し、Douglas-Peucker 法で頂点を減らし、垂直に近い辺を垂直にそろえる | `vectorize.labels_to_shapes` |
| 10 | 面積の大きい順に塗って PNG と SVG に書き出す | `vectorize.render_png` / `shapes_to_svg` |

9の「垂直にそろえる」処理は、建物の壁を直線にして建築物らしく見せるために入れています。

## 主なオプション

| オプション | 既定値 | 内容 |
|---|---|---|
| `--palette` | `teal` | `teal` / `warm` / `mono` / `original` |
| `--colors` | 8 | k-means のクラスタ数。増やすと細部が残る |
| `--majority` | 17 | 多数決フィルタの窓幅(px)。大きいほど形が大づかみになる |
| `--min-region` | 0.003 | 画像面積に対するこの割合より小さい領域を消す |
| `--simplify` | 0.005 | 多角形の簡略化の許容誤差(対角線長に対する割合) |
| `--snap-deg` | 12 | 垂直からこの角度以内の辺を垂直にする。0 で無効 |
| `--accent-chroma` | 28 | この彩度以上の領域をアクセント色にする |
| `--no-sky` | なし | 空の検出を切る |
| `--tone-mapping` | `area` | `area`は面積で重み付けして濃淡を割り当てる(暗い写真でも明暗が出る)。`rank`はクラスタごとに順に1段ずつ割り当てる |
| `--shape` | `rect` | `circle` にすると円形に切り抜いた透過 PNG を出す |
| `--svg` | なし | PNG に加えて SVG も出力する |

## 制約

- 構図は写真のままなので、写っているものが多い写真ほど図形が細かく残ります。被写体を絞って撮ると結果が良くなります
- 色の割り当ては明度と面積だけで決めます。道路を明るい色にするといった、物の意味に応じた塗り分けはしません
- 窓の縦縞やガラスの格子のような、写真に細かく写っている模様は平滑化の段階で消えます
- 空の検出は「上端に接していて明るい」という条件だけで判定します。白い建物が上端に接していると空と判定されることがあります
- 地平線付近のかすんだ街並みは明るく写るため、白やクリーム色で塗られます。遠景を青いシルエットにする処理はまだありません
- SVGは多角形を重ねて描くので、隣り合う図形のすき間に下の図形の色が細く見えることがあります。PNGはラベル画像を下地に塗ってからこの多角形を重ねるので、すき間は出ません

## 実験用スクリプト

`scripts/experiment.py`は、設定を変えた結果を横に並べた比較画像を作るスクリプトです。各変種は`名前:項目=値,項目=値`の形で指定し、項目には`StampConfig`のフィールド名を使います。

```bash
uv run python scripts/experiment.py examples/photo.jpg -o sheet.png \
    "rank:tone_mapping=rank" "area:tone_mapping=area" "blocky:majority=25,min_region=0.006"
```

## 開発

```bash
uv sync
uv run pytest            # 単体・結合・CLIのE2Eテストとカバレッジ
```

## ライセンス

MIT License。詳細は[LICENSE](LICENSE)を参照してください。
