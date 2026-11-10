# photo2stamp

現地で撮った写真を、少ない色数のフラットなスタンプ風イラストに変換するCLIです。
画像生成AIは使わず、OpenCVの古典的な画像処理だけで動きます。GPUは不要で、800pxの写真1枚を数秒で処理します。

現在はPOC(概念実証)の段階です。写真をL0平滑化で平らな面に分け、色を少数の基準色にそろえ、その上に輪郭線を描きます。
色は写真の色相(空は青、木は緑)を残します。被写体を見分ける手がかりは、色相と輪郭線が担っているためです。夜景は輪郭線の代わりに、窓の灯りを点で描きます。
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
| 城 | 分かる | 白い壁、青い屋根、尖塔、湖、森 |
| 灯台 | 分かる | 空、海、堤防、白い灯台と赤い先端 |
| 合成画像 | 分かる | ビル、看板、緑の木、道 |
| 街の俯瞰(作者撮影) | 分かる | 輪郭線の入った高層ビル群と空 |
| 夜景 | 分かる | 暗いビル群に灯りの点が並ぶ |
| 木陰の道 | 分かる | 青空、枝、道、緑と青緑の葉 |
| 夕景の橋 | ほぼ分かる | 灯りの付いた橋の帯、川、夕焼け。橋脚のアーチは出ない |
| 寺院 | ほぼ分かる | 段になった塔の形。極彩色の彫刻は灰色と青の面になる |

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
| 2 | 長辺を800pxに縮め、細部の量(エッジ密度)から作業用の大きさ(500〜400px)を決める | `pipeline.auto_work_size` |
| 3 | L0勾配最小化で、重要な輪郭を残したまま面を平らにする | `pipeline._smooth` |
| 4 | Lab色空間のk-meansで8色に分ける。色み(a, b)を明るさの2倍に重み付けし、同じ明るさの緑の木と灰色の道を別の色にする | `segment.quantize` |
| 5 | 彩度の高い小領域(看板や照明)を拾い、上端に接するなめらかな領域を空とする | `segment.extract_accents` / `extract_sky` |
| 6 | 色を決める。`natural`では色相を8つの基準色に、彩度を2段階に、明るさを一定の段差にそろえる。色のある面は明るさを78までに抑え(白に近づけるとsRGBの範囲を外れて淡くなるため)、白に近い明るさは灰色にだけ使う | `palette.natural_colors` |
| 7 | 小さな領域を隣に吸収させ、ラベルを元の大きさになめらかに拡大し、多角形にする | `segment.merge_small_regions` / `upscale_labels` / `vectorize.labels_to_shapes` |
| 8 | 輪郭線を描く。600pxの写真にDoG(ガウシアン差分)をかけ、短い線を捨てて太さをそろえる。暗い面の上には描かない | `lines.dog_lines` / `clean_lines` |
| 9 | 夜景(明るさの中央値が45未満)では面を暗く保ち、輪郭線の代わりに明るい点(窓の灯り)を灯り色の点で描く | `pipeline.is_night` / `lines.light_spots` |
| 10 | PNGとSVGに書き出す。円形・角丸のときは外周にリングと紙色のすき間を描く | `vectorize.render_png` / `frame_stamp` / `shapes_to_svg` |

### 参考にした研究

| 手法 | 出典 | 使い方 |
|---|---|---|
| 平滑化・量子化・輪郭線の組み合わせ | Winnemöller, Olsen, Gooch. Real-Time Video Abstraction. SIGGRAPH 2006 | 全体の構成。抽象化しても輪郭線があれば被写体が読めるという知見 |
| L0勾配最小化 | Xu, Lu, Xu, Jia. Image Smoothing via L0 Gradient Minimization. SIGGRAPH Asia 2011 | 段階3。OpenCV contribの`ximgproc.l0Smooth`を使う |
| XDoG | Winnemöller, Kyprianidis, Olsen. XDoG. Computers & Graphics 2012 | 段階8。XDoG(τ<1)は暗い面を黒く塗りつぶして夜景を壊したので、τ=1の素のDoGにした |

## 主なオプション

| オプション | 既定値 | 内容 |
|---|---|---|
| `--shape` | `rect` | `circle`/`rounded`にすると正方形に切り抜き、縁取り付きのスタンプ(透過PNG)にする |
| `--palette` | `natural` | `natural`(写真の色相を残して整える) / `teal` / `warm` / `mono`(固定パレット、明度だけで塗り分ける) / `original`(写真の色そのまま) |
| `--no-lines` | なし | 輪郭線を描かない(色の面だけ) |
| `--night` | `auto` | 夜モード。`auto`は写真の明るさで判定。`on`/`off`で強制 |
| `--work-size` | 0(自動) | 色分けに使う画像の長辺(px)。小さいほど抽象的になる |
| `--colors` | 8 | k-meansのクラスタ数 |
| `--ring` / `--ring-gap` | 0.035 / 0.02 | 縁取りリングの太さと、リング内側のすき間(スタンプの大きさに対する割合)。`--ring 0`で縁取りなし |
| `--min-region` | 0.002 | 画像面積に対するこの割合より小さい領域を消す |
| `--simplify` | 0.006 | 多角形の簡略化の許容誤差(対角線長に対する割合) |
| `--snap-deg` | 12 | 垂直からこの角度以内の辺を垂直にする。0で無効 |
| `--no-sky` / `--no-far` | なし | 空の検出 / 遠景のシルエット化(固定パレットのときだけ働く)を切る |
| `--svg` | なし | PNGに加えてSVGも出力する |

線の太さ・量・長さは`StampConfig`の`line_width`、`line_threshold`、`line_min_length`で変えられます。

## 制約

- 固定パレット(`teal`など)は明度だけで塗り分けるので、青空が白、木が黄色のように物の色が入れ替わり、被写体が分かりにくくなります。被写体を見せたいときは`natural`を使ってください
- 影の中の葉は写真では少し青みがかっているため、青緑で塗られます
- 多色の模様(寺院の彫刻など)は、色が平均されて灰色や青の面になります
- 夜モードの判定は写真全体の明るさだけで行います。夕景のように半分だけ暗い写真は昼の描き方になります
- 正方形の切り抜き位置は「エッジが最も多い範囲」で決めます。細部の多いものが主役でない写真では、主役が外れることがあります
- 空は「上端に接していて、なめらか」という条件で判定します。上端に接する無地の壁も空と判定されることがあります
- 雲の縁も輪郭線として描かれます
- SVGは多角形を重ねて描くので、隣り合う図形のすき間に下の図形の色が細く見えることがあります

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
