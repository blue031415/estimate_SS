# 実験ガイド

## 基本的な実行方法

実行可能な実験は `src/ex*.py` にあります。YAMLは処理を切り替えるためのものではなく、
対応するスクリプトへ設定値を渡すために使用します。

```powershell
uv run python src/ex3.py configs/ex3.yaml
```

YAML内の相対パスは、そのYAMLファイルが置かれたディレクトリを基準に解決されます。
YAMLを省略した場合は、各 `ex*.py` の `DEFAULT_CONFIG` に定義された既定値を使用します。

```powershell
uv run python src/ex3.py
```

## スクリプトと設定ファイルの対応

```text
src/ex1.py                   <-> configs/ex1.yaml
src/ex2_first.py            <-> configs/ex2_first.yaml
src/ex2_first_origin.py     <-> configs/ex2_first_origin.yaml
src/ex2_first_overlay.py    <-> configs/ex2_first_overlay.yaml
src/ex2_second.py           <-> configs/ex2_second.yaml
src/ex2_second_origin.py    <-> configs/ex2_second_origin.yaml
src/ex2_second_overlay.py   <-> configs/ex2_second_overlay.yaml
src/ex2_geodestic.py        <-> configs/ex2_geodestic.yaml
src/ex2_geodestic_origin.py <-> configs/ex2_geodestic_origin.yaml
src/ex3.py                   <-> configs/ex3.yaml
src/ex3_second.py            <-> configs/ex3_second.yaml
src/ex3_geodestic.py         <-> configs/ex3_geodestic.yaml
src/ex4.py                   <-> configs/ex4.yaml
src/ex5.py                   <-> configs/ex5.yaml
src/ex6.py                   <-> configs/ex6.yaml
src/ex6_analysis.py          <-> configs/ex6_analysis.yaml
src/ex6_pose.py              <-> configs/ex6_pose.yaml
src/ex7.py                   <-> configs/ex7.yaml
src/ex8.py                   <-> configs/ex8.yaml
src/ex9.py                   <-> configs/ex9.yaml
src/ex10.py                  <-> configs/ex10.yaml
src/ex11.py                  <-> configs/ex11.yaml
src/ex12.py                  <-> configs/ex12.yaml
```

## 実験1 共通部分空間と射影誤差

`src/ex1.py` はC3Dの各フレームからShape Subspaceを構築し、その平均射影行列を
固有値分解します。`contribution_threshold` を満たす固有ベクトルから共通部分空間を作り、
固有値の累積寄与率とフレームごとの射影誤差を `out/ex1/eigen_val.png` に保存します。

```powershell
uv run python src/ex1.py configs/ex1.yaml
```

## 実験2 時間差分と測地線分解

実験2は、元のShape Subspaceと共通部分空間へ射影したShape Subspaceを対象に、
時間方向の変化を評価します。

- `ex2_first.py`: 射影後の1階差分
- `ex2_first_origin.py`: 射影前の1階差分
- `ex2_first_overlay.py`: 複数dimensionの1階差分比較
- `ex2_second.py`: 射影後の2階差分
- `ex2_second_origin.py`: 射影前の2階差分
- `ex2_second_overlay.py`: 複数dimensionの2階差分比較
- `ex2_geodestic.py`: 射影後の測地線方向・直交方向への分解
- `ex2_geodestic_origin.py`: 射影前の測地線方向・直交方向への分解

各スクリプトでは `frame_interval` が比較するフレーム間隔を表します。

```powershell
uv run python src/ex2_first.py configs/ex2_first.yaml
uv run python src/ex2_second.py configs/ex2_second.yaml
uv run python src/ex2_geodestic.py configs/ex2_geodestic.yaml
```

## 実験3 ノイズ付きC3Dの解析

実験3は、C3Dの指定した身体部位とフレーム範囲へガウスノイズを加え、
共通部分空間と時間差分への影響を評価します。ノイズの対象、平均、共分散、scaleは
各YAMLの `noise` で指定します。

- `ex3.py`: 固有値の累積寄与率と射影誤差
- `ex3_second.py`: 2階差分
- `ex3_geodestic.py`: 測地線方向・直交方向への分解

```powershell
uv run python src/ex3.py configs/ex3.yaml
uv run python src/ex3_second.py configs/ex3_second.yaml
uv run python src/ex3_geodestic.py configs/ex3_geodestic.yaml
```

## 実験4 MPI-INF-3DHP動画の姿勢推定

`src/ex4.py` は `configs/ex4.yaml` の `input_videos` に指定したMPI-INF-3DHP動画を読み込み、
カメラ相対の3次元骨格を推定します。数値結果はNPZ、入力動画と3次元骨格の左右比較は
GIFとして保存します。複数動画を指定した場合は、設定順に同じ処理を行います。

MediaPipeのlite modelを使う場合は、実行前に公式modelを配置します。

```powershell
New-Item -ItemType Directory -Force data/models
Invoke-WebRequest `
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task `
  -OutFile data/models/pose_landmarker_lite.task
```

```powershell
uv run python src/ex4.py configs/ex4.yaml
```

既定の出力は `out/ex4/<video-name>.npz` と `out/ex4/<video-name>.gif` です。NPZには
world/image landmarks、visibility、presence、入力frame index、timestamp、landmark名、
骨格接続、入力FPS、入力path、estimator名を保存します。検出に失敗したフレームはNaN行として
残すため、入力動画とのフレーム対応は失われません。

推定器固有の処理は `experiments.pose_estimation` にあります。MediaPipeは33 landmarks、
MMPose Human3.6Mは17 landmarksを出力するため、後続処理では固定の関節数を仮定せず、
NPZの `landmark_names` と `connections` を使用します。

MMPoseを使う場合は追加依存関係を導入し、`estimator: mmpose` を指定します。

```powershell
uv sync --extra mmpose
uv run --extra mmpose python src/ex5.py configs/ex5.yaml
```

MMPoseは既定で公式の `human3d` aliasとCPUを使用し、初回実行時に公式checkpointを
PyTorch cacheへdownloadします。対応するCUDA環境では `mmpose_device: cuda:0` を指定できます。
MMPoseの出力では `world_landmarks` に3次元結果を保存し、取得できない
`image_landmarks` はNaNになります。MediaPipeの結果を上書きしないよう、出力名には
`_mmpose` suffixを付けます。

## 実験5 GaHu-Videoの姿勢推定

`src/ex5.py` は `data/GaHu-Video/S001.avi` に実験4と同じ3次元姿勢推定を適用します。
既定では動画全体を3フレームごとに処理し、左側のGIF panelでは元動画の16:9比率を維持します。

```powershell
uv run python src/ex5.py configs/ex5.yaml
```

既定の出力は `out/ex5/S001.npz` と `out/ex5/S001.gif` です。短い試行では
`max_frames` に整数を指定し、全体を処理する場合は `null` を指定します。

## 実験6 自撮り動画の姿勢推定と解析

`src/ex6.py` は `data/self_videos` の `video_pattern` に一致する動画をfilename順に処理します。
3次元姿勢推定のフレーム対応を保った結果を
`out/ex6/<video-name>/<video-name>.npz`、確認用の比較GIFを
`out/ex6/<video-name>/<video-name>_pose.gif` に保存します。検出失敗フレームはNPZ内で
NaNのまま保持します。

完全に検出できたフレームについて、実験2と同じ1階差分、2階差分、測地線分解を行います。
出力は次のようにframe intervalを先、dimensionを後に配置します。

```text
out/ex6/
  <video-name>/
    <video-name>.npz
    <video-name>_pose.gif
    fiN/origin/*.gif
    fiN/dN/*.gif
```

`origin` は共通部分空間へ射影しない条件です。dimensionとframe intervalはdirectoryで区別し、
filenameには重複して含めません。titleと凡例には解析条件を表示します。

```powershell
uv run python src/ex6.py configs/ex6.yaml
```

`frame_stride` は入力動画のsampling間隔、`frame_interval` はsampling後かつ検出成功した
フレーム列での比較間隔です。`first_dimension`、`second_dimension`、
`geodestic_dimension`、`frame_interval` は整数または整数のlistを受け付けます。

3D姿勢panelの視点だけを方位角方向へ90度回転する場合は
`rotate_pose_90_degrees: true` を指定します。通常の出力を上書きしないよう、実験6では
`<video-name>_pose_rotated_90.gif` に保存します。保存済みNPZから姿勢推定と数値解析を
繰り返さずに比較GIFだけを再描画する場合は、`ex6_pose.py` を使用します。

```powershell
uv run python src/ex6_pose.py configs/ex6_pose.yaml
```

保存済みNPZだけを再解析する場合は `ex6_analysis.py` を使います。動画decodeと姿勢推定は
行わず、既存のNPZも変更しません。

```powershell
uv run python src/ex6_analysis.py configs/ex6_analysis.yaml
```

## 実験7 姿勢推定結果の固有値解析

`src/ex7.py` は実験6の各NPZを読み込み、不完全な姿勢検出を含むフレームだけを除外します。
その後、実験1と同じ共通部分空間の固有値解析と射影誤差評価を行い、
`out/ex7/<video-name>_eigen_val.png` に保存します。入力NPZは変更しません。

```powershell
uv run python src/ex7.py configs/ex7.yaml
```

## 実験8〜10 座標射影の比較

実験8は、実験3と同じ規則でC3Dへノイズを加え、共通部分空間を構築します。中心化した
marker座標を `U U^T` で射影してから重心を戻し、指定フレームの射影前後を比較します。

```powershell
uv run python src/ex8.py configs/ex8.yaml
```

実験9は、実験6の各姿勢NPZについて、ノイズを加えずに同じ座標射影を行います。
動画ごとに1枚の比較PNGを保存します。

```powershell
uv run python src/ex9.py configs/ex9.yaml
```

実験10は、C3DとMediaPipeから肩、肘、手首、腰、膝、足首の左右12点を対応する順序で抽出します。
C3Dの手首はpaired marker、腰は前後markerの平均を使用します。共通部分空間は
ノイズ付きC3Dだけから構築し、動画姿勢座標をその空間へ射影します。射影前の動画姿勢も
`out/ex10/video_original.png` に保存します。

```powershell
uv run python src/ex10.py configs/ex10.yaml
```

実験8〜10は、既定では指定した1フレームの3次元座標を静止画で比較します。
`dimension` と `frame_index` は対応するYAMLで変更できます。

## 実験11 クリーンなC3D由来共通部分空間での差分解析

`src/ex11.py` は実験10と同じ12関節を抽出しますが、ノイズを加えていないC3Dから
共通部分空間を構築します。動画姿勢のShape Subspaceをこの空間へ射影し、実験6と同じ
1階差分、2階差分、測地線分解を行います。

```powershell
uv run python src/ex11.py configs/ex11.yaml
```

出力は次のdirectoryへ保存します。

```text
out/ex11/
  <video-name>/
    fiN/origin/*.gif
    fiN/dN/*.gif
```

`origin` は動画姿勢のShape Subspaceをそのまま評価する条件、`dN` はC3D由来の
dimension Nの共通部分空間へ射影する条件です。`dimension` と `frame_interval` は
整数または整数のlistを受け付けます。条件はdirectoryで区別し、filenameには含めません。

## 実験12 実験6と実験11の評価値比較

`src/ex12.py` は、実験6の姿勢推定データ由来の共通部分空間と、実験11のクリーンな
C3D由来の共通部分空間について、同じdimensionとframe intervalの評価値を重ねて表示します。
比較するのは1階差分、2階差分、測地線方向、測地線直交方向、および両方向をまとめたgraphです。
姿勢推定は再実行せず、実験6・11の既存出力も変更しません。

```powershell
uv run python src/ex12.py configs/ex12.yaml
```

PNGは `out/ex12/fiN/dN` に保存します。dimensionとframe intervalはdirectoryで区別し、
filenameには含めません。`dimensions` と `frame_interval` は整数のlistを指定できます。

## 共通module

- `experiments/common_subspace.py`: Shape Subspace、平均射影行列、固有値分解、共通部分空間への射影
- `experiments/dissimilarity.py`: 1階差分、2階差分、測地線分解、marker contribution
- `experiments/noise.py`: 身体部位とmarkerの対応、乱数生成、ノイズ付加
- `experiments/config.py`: YAML読込、既定値とのmerge、path解決
- `experiments/pose_estimation.py`: MediaPipeとMMPoseに共通する3次元姿勢推定interface
- `utils/c3d.py`: C3D読込とpoint dataの整形
- `utils/display.py`: motion、評価値、contributionの可視化
- `utils/video.py`: 動画情報の取得とRGB frameの読込
- `utils/pose_display.py`: 入力動画と3次元骨格の比較GIF生成
- `utils/plot_style.py`: 利用可能な日本語対応fontの選択とMatplotlibへの適用

## subspace package

Shape Subspaceの基礎計算は `subspace` package内で完結します。

- `subspace/decomposition.py`: SVD、Gram-Schmidt、正規直交基底
- `subspace/shape.py`: Shape Subspace、差分部分空間、主成分部分空間、距離
- `subspace/geodesic.py`: 測地線方向と直交方向への分解
- `subspace/__init__.py`: 外部へ公開するAPI

利用側は内部moduleではなく、公開APIからimportします。

```python
from subspace import cal_magnitude, gen_shape_subspace
```
