# photo2stamp

[![tests](https://github.com/shiyow5/photo2stamp/actions/workflows/test.yml/badge.svg)](https://github.com/shiyow5/photo2stamp/actions/workflows/test.yml)

写真を、少ない色数のフラットなスタンプ風イラストに変換するツールです。CLIとPythonライブラリの両方で使えます。
画像生成AIは使わず、OpenCVの古典的な画像処理だけで動かしています。GPUは不要で、800pxの写真1枚なら数秒です。

写真を平らな面に分けて色を少数の基準色にそろえ、その上に輪郭線を描きます。
色は写真の色相(空は青、木は緑)を残すので、抽象化しても何が写っているか分かります。
室内の照明による色かぶりは先に取り除くため、白い皿は白いままです。
細かい色の模様(彫刻や花、料理の具)は小さな色の点として残します。点は大きく目立つものから60個までにして、草や落ち葉の粒は描きません。
夜景では輪郭線の代わりに、窓の灯りを点にして描くのが特徴です。
円形や角丸を選ぶと、縁取りの付いたスタンプの形(透過PNG)で出力します。

## 変換例

左から順に、元写真、`natural`配色(既定)、`teal`配色、`natural`の円形スタンプです。

| 元写真 | natural | teal | natural + circle |
|---|---|---|---|
| ![元写真](examples/photo.jpg) | ![natural](examples/photo_natural.png) | ![teal](examples/photo_teal.png) | ![circle](examples/photo_circle.png) |

SVG出力の例は[examples/photo_natural.svg](examples/photo_natural.svg)にあります。

性質の違う写真8枚を、既定の設定のまま円形スタンプにした結果です。左が元写真、右が変換結果です。

![円形スタンプの一覧](examples/gallery_circle.jpg)

ほかの一覧もあります。

| 一覧 | 設定 |
|---|---|
| [gallery.jpg](examples/gallery.jpg) | 枠なし(`--shape rect`) |
| [gallery_circle_after.jpg](examples/gallery_circle_after.jpg) | 枠なしの絵をそのまま丸く切り抜く(`--shape circle --crop after`) |
| [gallery_indoor.jpg](examples/gallery_indoor.jpg) | 料理・室内・動物・花の8枚(枠なし) |
| [gallery_teal.jpg](examples/gallery_teal.jpg) | `teal`配色 |

## インストール

Python 3.10以上が必要です。

```bash
git clone https://github.com/shiyow5/photo2stamp.git
cd photo2stamp
uv sync            # または: pip install .
```

## 使い方

```bash
uv run photo2stamp photo.jpg -o out --svg
# out/photo.png と out/photo.svg ができる

uv run photo2stamp *.jpg -o out --shape circle                  # 縁取り付きの丸いスタンプ
uv run photo2stamp *.jpg -o out --shape circle --crop after      # 枠なしの絵を丸く切り抜く
uv run photo2stamp photo.jpg -o out --work-size 150 --colors 5   # もっと大胆に抽象化する
uv run photo2stamp photo.jpg -o out --line-width 3               # 輪郭線を太くする
uv run photo2stamp photo.jpg -o out --palette original          # 写真の色を平坦化して使う
```

出力ファイル名は入力と同じ名前で、拡張子が`.png`(と`.svg`)になります。
名前が重なる入力は`photo_1.png`のように番号を付けて書き分けます。
読めないファイルがあっても残りの処理は続け、その場合の終了コードは1です。

Pythonから呼ぶ場合の例です。

```python
import cv2
from photo2stamp import StampConfig, stylize

result = stylize(cv2.imread("photo.jpg"), StampConfig(shape="circle"))
cv2.imwrite("stamp.png", result.image)   # BGR、円形・角丸のときはBGRA
with open("stamp.svg", "w", encoding="utf-8") as f:
    f.write(result.svg)
```

`StampConfig`のフィールド名には、下のオプション名のハイフンをアンダースコアに替えたものを使います。

## オプション

| オプション | 既定値 | 内容 |
|---|---|---|
| `-o`, `--out-dir` | `out` | 出力先のフォルダ |
| `--svg` | なし | PNGに加えてSVGも出力する |
| `--shape` | `rect` | `circle`/`rounded`にすると正方形に切り抜き、縁取り付きのスタンプにする |
| `--crop` | `before` | `circle`/`rounded`の切り抜き方。`before`は写真を先に正方形に切り、その範囲だけで色と線を決める。`after`は写真全体で絵を作ってから切り抜くので、`rect`の絵と同じ色・線になる(夜モードの判定も写真全体で行う) |
| `--ring` / `--ring-gap` | 0.035 / 0.02 | 縁取りリングの太さと、リング内側のすき間(スタンプの大きさに対する割合)。`--ring 0`で縁取りなし |
| `--palette` | `natural` | `natural`(写真の色相を残して整える) / `teal` / `warm` / `mono`(固定の配色を明るさの順に割り当てる) / `original`(写真の色そのまま) |
| `--colors` | 8 | 色分けの数(k-meansのクラスタ数) |
| `--work-size` | 0(自動) | 色分けに使う画像の長辺(px)。小さいほど抽象的になる。0なら細部の量から500〜400pxを選ぶ |
| `--max-size` | 800 | 出力の長辺(px)。0で元の大きさのまま |
| `--no-lines` | なし | 輪郭線を描かない(色の面だけ) |
| `--no-details` | なし | 細かい色の模様を点として残す処理を切る(`natural`のときだけ働く) |
| `--no-white-balance` | なし | 照明の色かぶりを取り除かず、写真の色のまま使う |
| `--line-width` | 1.5 | 輪郭線の太さ(600pxの画像に対するpx) |
| `--line-threshold` | 0.012 | 線を引くのに必要なエッジの強さ。大きいほど線が減る |
| `--line-min-length` | 0.07 | 長辺に対するこの割合より短い線を捨てる |
| `--line-sigma` | 2.0 | 線を拾う大きさ。大きいほど大きな構造だけになる |
| `--night` | `auto` | 夜モード(暗い面に灯りの点)。`auto`は写真の明るさと灯りの散らばり方で判定する。`on`/`off`で強制 |
| `--smoother` | `l0` | 面を平らにする方法。`l0`(L0勾配最小化)か`meanshift` |
| `--l0-lambda` | 0.04 | L0平滑化の強さ。大きいほど面が平らになる |
| `--smooth` | 1 | `--smoother meanshift`のときの平滑化の回数 |
| `--majority` | 3 | 多数決フィルタの窓(作業画像のpx)。大きいほど形が角張る |
| `--min-region` | 0.002 | 画像面積に対するこの割合より小さい領域を消す |
| `--simplify` | 0.006 | 多角形の簡略化の許容誤差(対角線長に対する割合) |
| `--snap-deg` | 12 | 垂直からこの角度以内の辺を垂直にする。0で無効 |
| `--accent-chroma` | 28 | この彩度より鮮やかな小領域(看板や照明)を残す |
| `--no-sky` | なし | 空の検出を切る |
| `--no-far` / `--far-delta` | なし / 20 | 遠景のシルエット化(固定配色のときだけ働く)を切る / 空の色にどこまで近ければ遠景とみなすか |
| `--tone-mapping` | `area` | 明るさの割り振り方。`area`は面積で均す、`rank`は色ごとに1段ずつ |
| `--seed` | 0 | k-meansの乱数の種 |
| `--version` | | 版を表示する |

## 処理の流れ

| 段階 | 処理 | モジュール |
|---|---|---|
| 1 | 照明の色かぶりを取り除く。彩度の低い画素を「光に染まった灰色の物」とみなして光の色を推定する。上半分の青みは空とみなして使わず、上端が明るい暖色(夕焼け)のときは補正しない | `whitebalance.white_balance` |
| 2 | 円形・角丸のときは正方形に切り抜く。位置はエッジの量と色の目立ち度が最も大きい範囲(主役の被写体)に合わせる(`--crop after`では最後に切り抜く) | `pipeline.square_window` |
| 3 | 長辺を800pxに縮め、細部の量(エッジ密度)から作業用の大きさ(500〜400px)を決める | `pipeline.auto_work_size` |
| 4 | L0勾配最小化で、重要な輪郭を残したまま面を平らにする | `pipeline._smooth` |
| 5 | Lab色空間のk-meansで8色に分ける。色み(a, b)を明るさの2倍に重み付けし、同じ明るさの緑の木と灰色の道を別の色にする | `segment.quantize` |
| 6 | 彩度の高い小領域(看板や照明、ご飯の上の赤ピーマン)を色相ごとに拾い、上端に接するなめらかな領域を空とする | `segment.extract_accents` / `extract_sky` |
| 7 | 色を決める。`natural`では色相を8方向に、明るさを一定の段差にそろえ、彩度は写真の彩度に比例させる。うっすらした色味は光の色とみなして灰色に残す。葉に囲まれた暗い面(日陰の葉)は暗い緑にする | `palette.natural_colors` / `segment.tint_shaded_foliage` |
| 8 | 小さな領域を隣に吸収させ、ラベルを元の大きさになめらかに拡大し、多角形にする | `segment.merge_small_regions` / `upscale_labels` / `vectorize.labels_to_shapes` |
| 9 | 細かい色の模様を、600pxの写真で周りと色が違う小さな点として拾う。40px未満の粒と下の面と色の近い点は捨て、大きさと色の差の順に60個まで塗る。粒が400個を超える写真は質感とみなして塗らない | `details.find_details` / `details.select_details` |
| 10 | 輪郭線を描く。600pxの写真にDoG(ガウシアン差分)をかけ、短い線を捨てて太さをそろえる。暗い面の上と、昼の空の雲の縁(なだらかな境目)には描かない | `lines.dog_lines` / `pipeline._soft_sky_edges` |
| 11 | 夜景(暗く、明るい部分が小さく散らばっている写真)では面を暗く保ち、輪郭線の代わりに窓の灯りを灯り色の点で描く。点どうしは長辺の1.5%以上離す | `pipeline.is_night` / `lines.light_spots` |
| 12 | PNGとSVGに書き出す。SVGは輪郭どおりの下地の上に滑らかな多角形を重ね、すき間に背景色が出ないようにする。円形・角丸のときは外周にリングと紙色のすき間を描く | `vectorize.render_png` / `frame_stamp` / `shapes_to_svg` |

それぞれの設定値を選んだ理由、試してやめた方法、参考にした研究は[docs/development-notes.md](docs/development-notes.md)にまとめています。
設定を変える前に読んでください。

## 制約

- 固定配色(`teal`など)は色を明るさの順に割り当てるので、青空が白、木が黄色のように物の色が入れ替わります。被写体を見せたいときは`natural`を使ってください
- 色かぶりの補正は、画面の上半分の青みを空とみなして使いません。上の方にある青っぽい映り込みや壁は、補正の手がかりになりません
- 上端が明るい暖色の写真は夕焼けとみなし、暖色の照明を補正しません
- 日陰の葉が暗い緑になるのは、周りを葉に囲まれているときだけです。空を背にした日陰の葉は暗い灰色になります
- 細かい色の模様は点として残しますが、600pxの写真で40pxより小さい模様は描きません。点は60個までなので、模様の多い彫刻では一部だけが残ります
- 落ち葉や草むらのように細かい模様が画面いっぱいにある写真では、輪郭線が短く曲がった線になって混み合います
- 日陰で青みを帯びた灰色の壁は、自然配色で彩度を強めるため鮮やかな水色になります(壁画の写真)
- 空は「上端に接していて、なめらかで、画面の下端までは届かない」という条件で判定します。上端に接する無地の壁も空と判定されることがあります
- SVGとPNGは、境目のアンチエイリアスの違いで1ピクセルほど食い違います
- 夜の判定は「明るい部分が小さく散らばっている」ことを条件にしています。照らされた大きな建物や月が大きく写った夜景は、昼の描き方になります
## 開発

```bash
uv sync
uv run pytest            # 単体・結合・CLIのE2Eテストとカバレッジ
```

GitHub Actionsで、pushとプルリクエストのたびにPython 3.10と3.13でテストを実行します。CIではカバレッジが80%を下回ると失敗にしています。

例の画像を作り直すときは、次のスクリプトを使ってください。

```bash
uv run python scripts/make_synthetic.py                      # examples/synthetic_city.png を作る
uv run python scripts/gallery.py -o examples/gallery_circle.jpg --shape circle \
    examples/photo.jpg examples/synthetic_city.png examples/free/*.jpg
```

`gallery.py`で指定できる項目は`--shape`、`--palette`、`--crop`、`--columns`です。

## 写真の出典

`examples/photo.jpg`は作者の撮影です。`examples/free/`の写真はWikimedia CommonsのCC0の画像を、長辺800pxに縮小して使っています。

| ファイル | 作者 | 元画像 |
|---|---|---|
| castle.jpg | Wilfredor | [Neuschwanstein Castle 2024-02](https://commons.wikimedia.org/wiki/File:Neuschwanstein_Castle_2024-02.jpg) |
| harbour.jpg | Wilfredor | [Victoria Harbour skyscrapers](https://commons.wikimedia.org/wiki/File:Victoria_Harbour_skyscrapers.jpg) |
| lighthouse.jpg | Jebulon | [Lighthouse, harbor Saint-Martin-de-Ré](https://commons.wikimedia.org/wiki/File:Lighthouse,_harbor_Saint-Martin-de-R%C3%A9,_R%C3%A9_island,_Charente-Maritime,_France.jpg) |
| bridge.jpg | Pierre Blaché | [Bir-Hakeim bridge, Paris](https://commons.wikimedia.org/wiki/File:Bir-Hakeim_bridge,_Paris_24_June_2021.jpg) |
| street.jpg | Omina006 | [Street city](https://commons.wikimedia.org/wiki/File:Street_city.jpg) |
| temple.jpg | Anonymous0006 | [Thillai Nataraja Temple](https://commons.wikimedia.org/wiki/File:Thillai_Nataraja_Temple_(Chidambaram_Nataraja_Temple).jpg) |
| paella.jpg | Wilfredor | [Paella de fruit de mer](https://commons.wikimedia.org/wiki/File:Paella_de_fruit_de_mer.jpg) |
| steak.jpg | Syced | [Dish at Italian restaurant in Tokyo](https://commons.wikimedia.org/wiki/File:Dish_at_Italian_restaurant_in_Tokyo.jpg) |
| ramen.jpg | Andy Li | [Soy Milk Ramen and Tonkotsu Miso Ramen by Goemon Ramen Bar](https://commons.wikimedia.org/wiki/File:Soy_Milk_Ramen_and_Tonkotsu_Miso_Ramen_by_Goemon_Ramen_Bar.jpg) |
| sushi.jpg | Mojmir Churavy | [Sushi Plate in Organic Sushi Prague 4](https://commons.wikimedia.org/wiki/File:Sushi_Plate_in_Organic_Sushi_Prague_4.jpg) |
| cafe.jpg | Andy Li | [Interior of 1UP Cafe 2024-02-18](https://commons.wikimedia.org/wiki/File:Interior_of_1UP_Cafe_2024-02-18.jpg) |
| cat.jpg | TheOmniCommie | [Orange tabby persian cat sleeping on a couch](https://commons.wikimedia.org/wiki/File:Orange_tabby_persian_cat_sleeping_on_a_couch.png) |
| tulip.jpg | Alabama Extension | [Close up of red-yellow tulip in bloom](https://commons.wikimedia.org/wiki/File:Close_up_of_red-yellow_tulip_in_bloom.jpg) |
| autumn.jpg | Martin Vorel | [Autumn Forest Wet Bark](https://commons.wikimedia.org/wiki/File:Autumn_Forest_Wet_Bark.jpg) |

## ライセンス

MIT License。詳細は[LICENSE](LICENSE)を参照してください。
