# photo2stamp

現地で撮った写真を、少ない色数のフラットなスタンプ風イラストに変換するCLIです。
画像生成AIは使わず、OpenCVの古典的な画像処理だけで動きます。GPUは不要で、800pxの写真1枚を数秒で処理します。

現在はPOC(概念実証)の段階です。写真を260〜400px程度まで縮めてから色分けし、境界をなめらかに拡大し直して、細部を捨てた大きな形にします。
色は写真の色相(空は青、木は緑)を残したまま、少数の基準色にそろえます。被写体を見分ける手がかりは色相が担っているので、これを捨てると何の絵か分からなくなるためです。
円形や角丸の枠を付けた場合は、縁取りの付いたスタンプの形で出力されます。
写真から要素を選んで配置し直すような意味的なデフォルメはしません。

## 変換例

左から順に、元写真、`natural`パレット(既定)、`teal`パレット、`natural`の円形スタンプです。

| 元写真 | natural | teal | natural + circle |
|---|---|---|---|
| ![元写真](examples/photo.jpg) | ![natural](examples/photo_natural.png) | ![teal](examples/photo_teal.png) | ![circle](examples/photo_circle.png) |

SVG出力の例は[examples/photo_natural.svg](examples/photo_natural.svg)にあります。

## 複数の写真での結果

性質の違う写真8枚を、既定の設定のまま円形スタンプにした結果です。左が元写真、右が変換結果です。

![円形スタンプの一覧](examples/gallery_circle.jpg)

枠なし(`--shape rect`)の一覧は[examples/gallery.jpg](examples/gallery.jpg)、`teal`パレットの一覧は[examples/gallery_teal.jpg](examples/gallery_teal.jpg)にあります。

出力を1枚ずつ元写真と並べ、何が写っているか分かるかを目で確認した結果です(枠なしの場合)。

| 写真 | 判定 | 見た目 |
|---|---|---|
| 灯台 | 分かる | 空・海・堤防・白い灯台と赤い先端 |
| 城 | 分かる | 白い壁と青い屋根、湖、森 |
| 合成画像 | 分かる | ビル・看板・木・道 |
| 木陰の道 | 分かる | 青空・緑の並木・道 |
| 夜景 | 分かる | 暗い空にビル群のシルエット。ただし円形に切り抜くとビルの輪郭が外れて読めなくなる |
| 街の俯瞰(作者撮影) | ほぼ分かる | 上半分のビル群は読めるが、下半分は塊になる |
| 夕景の橋 | ほぼ分かる | 空・灯りの帯・川には分かれるが、橋の形は出ない |
| 寺院 | ほぼ分かる | 塔の輪郭は出るが、極彩色の彫刻は平均されて灰色になる |

合成画像は`scripts/make_synthetic.py`で生成しました。一覧は`scripts/gallery.py`で作り直せます(`--shape`と`--palette`を指定できます)。

### 写真の出典

`examples/free/`の写真はWikimedia CommonsのCC0の画像を、長辺800pxに縮小して使っています。

| ファイル | 作者 | 元画像 |
|---|---|---|
| castle.jpg | Wilfredor | [Neuschwanstein Castle 2024-02](https://commons.wikimedia.org/wiki/File:Neuschwanstein_Castle_2024-02.jpg) |
| harbour.jpg | Wilfredor | [Victoria Harbour skyscrapers](https://commons.wikimedia.org/wiki/File:Victoria_Harbour_skyscrapers.jpg) |
| lighthouse.jpg | Jebulon | [Lighthouse, harbor Saint-Martin-de-Ré](https://commons.wikimedia.org/wiki/File:Lighthouse,_harbor_Saint-Martin-de-R%C3%A9,_R%C3%A9_island,_Charente-Maritime,_France.jpg) |
| bridge.jpg | Pierre Blaché | [Bir-Hakeim bridge, Paris](https://commons.wikimedia.org/wiki/File:Bir-Hakeim_bridge,_Paris_24_June_2021.jpg) |
| street.jpg | Omina006 | [Street city](https://commons.wikimedia.org/wiki/File:Street_city.jpg) |
| temple.jpg | Anonymous0006 | [Thillai Nataraja Temple](https://commons.wikimedia.org/wiki/File:Thillai_Nataraja_Temple_(Chidambaram_Nataraja_Temple).jpg) |

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

uv run photo2stamp *.jpg -o out --shape circle                  # 縁取り付きの丸いスタンプ
uv run photo2stamp photo.jpg -o out --work-size 150 --colors 5     # もっと大胆に抽象化する
uv run photo2stamp photo.jpg -o out --palette original          # 写真の色を平坦化して使う
```

Pythonから呼ぶ場合の例です。

```python
import cv2
from photo2stamp import StampConfig, stylize

result = stylize(cv2.imread("photo.jpg"), StampConfig(palette="natural", shape="circle"))
cv2.imwrite("stamp.png", result.image)
open("stamp.svg", "w").write(result.svg)
```

## 処理の流れ

| 段階 | 処理 | モジュール |
|---|---|---|
| 1 | 円形・角丸のときは正方形に切り抜く。位置はエッジの最も多い範囲(主役の被写体)に合わせる | `pipeline.crop_square` |
| 2 | 長辺を800pxに縮小し、エッジ密度(細部の量)を測る | `segment.edge_density` |
| 3 | 細部の量から作業解像度(400〜260px)を決め、作業用の小さな画像を作る | `pipeline.auto_work_size` |
| 4 | バイラテラルフィルタとmean-shiftで平滑化し、Lab色空間のk-meansで色分けし、多数決フィルタで境界をならす | `segment.smooth` / `quantize` / `majority_filter` |
| 5 | 彩度の高い小領域(看板や照明)を独立したクラスタとして拾う | `segment.extract_accents` |
| 6 | 上端に接していて、エッジの少ないなめらかな領域を空とする。色は問わないので夕焼けや曇り空も拾える。空の代表色から大きく外れる画素(白い塔など)は空から外す | `segment.extract_sky` |
| 7 | 空の直下にあり、すぐ上の空と色が近い画素を遠景とみなし、1色のシルエットにする | `segment.extract_far` |
| 8 | 色を決める。`natural`では色相を8つの基準色にそろえ、彩度を2段階、明度を一定の段差にする。灰色は灰色のまま残す。同じ色になったクラスタは統合し、小さな領域は隣に吸収させる | `palette.natural_colors` / `segment.merge_small_regions` |
| 9 | ラベルを元の大きさになめらかに拡大する。最近傍補間のような階段状の縁にならない | `segment.upscale_labels` |
| 10 | 輪郭を追跡し、Douglas-Peucker法で頂点を減らし、垂直に近い辺を垂直にそろえる | `vectorize.labels_to_shapes` |
| 11 | PNGとSVGに書き出す。円形・角丸のときは外周にリングと紙色のすき間を描く | `vectorize.render_png` / `frame_stamp` / `shapes_to_svg` |

抽象化の度合いは、主に3の作業解像度で決まります。小さく縮めるほど細部が消え、図形が大きくなります。

## 主なオプション

| オプション | 既定値 | 内容 |
|---|---|---|
| `--shape` | `rect` | `circle`/`rounded`にすると正方形に切り抜き、縁取り付きのスタンプ(透過PNG)にする |
| `--palette` | `natural` | `natural`(写真の色相を残して整える) / `teal` / `warm` / `mono`(固定パレット、明度だけで塗り分ける) / `original`(写真の色そのまま) |
| `--work-size` | 0(自動) | 色分けに使う画像の長辺(px)。小さいほど抽象的になる |
| `--colors` | 8 | k-meansのクラスタ数 |
| `--ring` / `--ring-gap` | 0.035 / 0.02 | 縁取りリングの太さと、リング内側のすき間(スタンプの大きさに対する割合)。`--ring 0`で縁取りなし |
| `--majority` | 3 | 多数決フィルタの窓幅(作業画像のpx)。大きいほど形が大づかみになる |
| `--min-region` | 0.004 | 画像面積に対するこの割合より小さい領域を消す |
| `--simplify` | 0.006 | 多角形の簡略化の許容誤差(対角線長に対する割合) |
| `--snap-deg` | 12 | 垂直からこの角度以内の辺を垂直にする。0で無効 |
| `--accent-chroma` | 28 | この彩度以上の領域をアクセント色にする |
| `--no-sky` / `--no-far` | なし | 空の検出 / 遠景のシルエット化を切る |
| `--far-delta` | 20 | すぐ上の空との色差(Lab)がこの値以下の画素を遠景候補にする |
| `--tone-mapping` | `area` | `area`は面積で重み付けして濃淡を割り当てる。`rank`はクラスタごとに1段ずつ割り当てる |
| `--svg` | なし | PNGに加えてSVGも出力する |

## 制約

- 固定パレット(`teal`など)は明度だけで塗り分けるので、青空が白、木が黄色のように物の色が入れ替わり、被写体が分かりにくくなります。被写体を見せたいときは`natural`を使ってください
- 多色の模様(寺院の彫刻など)は、クラスタ内で色が平均されて灰色になります
- 正方形の切り抜き位置は「エッジが最も多い範囲」で決めます。木の葉のように細かいものが主役でない写真では、主役が外れることがあります(木陰の道の例)
- 空は「上端に接していて、なめらか」という条件で判定します。上端に接する無地の壁も空と判定されることがあります
- 遠景の判定は「空の直下にあって空と色が近い」という条件だけで行います。淡い灰色の建物が空の近くにあると遠景と区別できません
- 細い被写体(電柱、塔の先端など)は、作業解像度で数px以下になると消えます
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
