# 実験設定

実験は必ず対応する `src/ex*.py` から実行します。YAML は処理を選択するものではなく、
そのスクリプトに変数を渡すためだけのファイルです。

たとえば実験3は次のように実行します。

```powershell
uv run python src/ex3.py configs/ex3.yaml
```

`src/ex3.py` には、以下の実験フローが明示されています。

1. C3D データの読込と対象マーカーの選択
2. 指定フレームへのノイズ付加
3. フレームごとの形状部分空間と共通部分空間の構築
4. 射影残差の集計と固有値の出力

数値やパスは `configs/ex3.yaml` で変更します。YAML 内の相対パスは、その YAML
ファイルが置かれたディレクトリを基準に解決されます。

YAML を省略した場合は、各 `ex*.py` の `DEFAULT_CONFIG` に定義された従来値で実行します。

```powershell
uv run python src/ex3.py
```

## 対応関係

各スクリプトには同名の設定ファイルがあります。

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
```

## Experiment 4: MPI-INF-3DHP video pose estimation

`src/ex4.py` decodes configured MPI-INF-3DHP videos, estimates a camera-relative
3D skeleton with MediaPipe Pose Landmarker, saves the numerical result as NPZ,
and creates a side-by-side GIF of the source video and 3D skeleton. The first
configured video is the primary trial; subsequent entries repeat the same flow
for other subjects or motions.

Download the official lite model once before running the experiment:

```powershell
New-Item -ItemType Directory -Force data/models
Invoke-WebRequest `
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task `
  -OutFile data/models/pose_landmarker_lite.task
```

Then run:

```powershell
uv run python src/ex4.py configs/ex4.yaml
```

The default outputs are `out/ex4/<video-name>.gif` and
`out/ex4/<video-name>.npz`. NPZ contains world/image landmarks, visibility,
presence, source frame indices and timestamps, landmark names, skeleton edges,
source FPS, source path, and estimator name. Missing detections are retained as
NaN rows so frame correspondence is never lost.

Model-specific inference is isolated in `experiments.pose_estimation`. Both
`MediaPipePoseEstimator` and `MMPosePoseEstimator` implement the same
`PoseEstimator` protocol, so decode, NPZ, and GIF processing remain common.

MediaPipe is the default. To add the optional MMPose environment and switch
either experiment, run:

```powershell
uv sync --extra mmpose
# Set `estimator: mmpose` in configs/ex4.yaml or configs/ex5.yaml.
uv run --extra mmpose python src/ex5.py configs/ex5.yaml
```

The MMPose setting uses the official `human3d` alias (RTMDet, RTMPose, and
MotionBERT) and CPU by default. On first use, MMPose downloads its official
checkpoints into the PyTorch cache. The Windows/Python 3.12 optional environment
uses `mmcv-lite`; the adapter supplies RTMDet's NMS through TorchVision while
keeping unsupported native MMCV operations fail-fast. Set
`mmpose_device: cuda:0` when a compatible
CUDA-enabled PyTorch environment is available. `mmpose_max_input_size` limits
the longer image side before MMPose inference; the default 1024 avoids feeding
the original 2048-pixel MPI-INF-3DHP frames into the CPU pipeline.

MediaPipe produces 33 landmarks and MMPose Human3.6M produces 17, so consumers
must use the `landmark_names` and `connections` arrays saved in each NPZ rather
than assuming a fixed joint count. MMPose does not expose the intermediate 2D
pose through this inferencer result; therefore its `image_landmarks` array is
NaN while its `world_landmarks` contains the 3D result. MMPose outputs use an
`_mmpose` filename suffix so they do not overwrite MediaPipe results.

## Experiment 5: GaHu-Video S001 pose estimation

`src/ex5.py` applies the same MediaPipe 3D pose workflow to
`data/GaHu-Video/S001.avi`. The default configuration samples every third frame
across the complete video and preserves its 16:9 aspect ratio in the left GIF
panel.

```powershell
uv run python src/ex5.py configs/ex5.yaml
```

The outputs are `out/ex5/S001.npz` and `out/ex5/S001.gif`. Set `max_frames` in
`configs/ex5.yaml` to an integer for a shorter trial, or leave it as `null` to
process through the end of the source video.

共通モジュールは次の役割に限定しています。

- `experiments/common_subspace.py`: 形状部分空間の構築、固有値分解、共通部分空間への射影
- `experiments/dissimilarity.py`: 1階・2階差分と測地線分解による評価
- `experiments/noise.py`: マーカー選択、乱数生成、ノイズ付加
- `experiments/config.py`: YAML の読込、既定値とのマージ、パス解決
- `experiments/pose_estimation.py`: 差し替え可能な3次元骨格推定インターフェースとMediaPipe実装
- `utils/c3d.py`: C3D の読込と点群データの NaN 除去
- `utils/display.py`: モーションと評価値の可視化
- `utils/video.py`: 動画情報の取得とRGBフレームの読込
- `utils/pose_display.py`: 元動画と3次元骨格の左右比較GIF生成

実験の順序や、どの計算を組み合わせるかは各 `ex*.py` に残しています。

## subspace パッケージ

形状部分空間の基礎計算は `subspace` パッケージ内で完結しています。

- `subspace/decomposition.py`: SVD、Gram-Schmidt、正規直交基底
- `subspace/shape.py`: 形状・差分・主成分部分空間と距離
- `subspace/geodesic.py`: 測地線方向と直交方向への分解
- `subspace/__init__.py`: 外部へ公開するAPI

利用側は内部構成に依存せず、次のようにインポートします。

```python
from subspace import cal_magnitude, gen_shape_subspace
```
