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
src/ex6.py                   <-> configs/ex6.yaml
src/ex6_analysis.py          <-> configs/ex6_analysis.yaml
src/ex7.py                   <-> configs/ex7.yaml
src/ex8.py                   <-> configs/ex8.yaml
src/ex9.py                   <-> configs/ex9.yaml
src/ex10.py                  <-> configs/ex10.yaml
src/ex11.py                  <-> configs/ex11.yaml
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

## Experiment 6: self-video pose estimation and Experiment 2 analyses

`src/ex6.py` processes every video matching `video_pattern` under
`data/self_videos` in filename order. For each video it performs 3D skeleton
estimation, saves the frame-aligned estimate to `out/ex6/<video-name>.npz`, and
creates `out/ex6/<video-name>_pose.gif` for visual confirmation. Failed pose
detections remain as NaN frames in the NPZ.

The fully detected frames from that NPZ are then passed through the same common
subspace projection, first-order difference, second-order difference, and
geodestic decomposition functions used by the Experiment 2 scripts. Their GIFs
are written under `out/ex6/origin/fiN` and `out/ex6/dN/fiN` using `_first_origin`, `_second_origin`, and
`_geodestic_origin` variants for analyses without common-subspace projection,
plus `_first_d4`, `_second_d5`,
`_along_geodestic_d5`, `_orth_geodestic_d5`, and `_geodestic_d5` suffixes.

```powershell
uv run python src/ex6.py configs/ex6.yaml
```

The default samples every third source frame. `frame_interval` is measured in
these sampled, successfully detected frames. Set `max_frames` for a short trial;
leave it as `null` to process all four videos. Existing videos and Experiment 2
outputs are only read and are never overwritten.
Analysis GIF filenames and titles include `_fi<frame_interval>` so temporal
interval trials can coexist. The current default interval is 3 sampled frames.

To repeat only the numerical analysis with different common-subspace
dimensions, edit `first_dimension`, `second_dimension`, and
`geodestic_dimension` in `configs/ex6_analysis.yaml`. Each accepts either one
integer or a list of integers. Then run:

```powershell
uv run python src/ex6_analysis.py configs/ex6_analysis.yaml
```

This command reads the existing `out/ex6/*.npz` files and does not decode video
or run pose estimation. Output filenames include each configured dimension, so
results from different dimensions can coexist. Reusing the same dimensions
replaces only the corresponding analysis GIFs.

## Experiment 7: eigenvalues from Experiment 6 estimates

`src/ex7.py` loads every Experiment 6 NPZ, removes only frames containing an
incomplete pose detection, and applies the same common-subspace eigenvalue and
projection-error evaluation used by Experiment 1. It writes one graph per input
to `out/ex7/<video-name>_eigen_val.png` without modifying the NPZ files.

```powershell
uv run python src/ex7.py configs/ex7.yaml
```

## Experiments 8-10: projected coordinate plots

Experiment 8 repeats Experiment 3's noise injection and common-subspace
estimation, then projects the centered noisy marker coordinates with `U U^T`,
restores the frame centroid, and plots the configured frame before and after
projection.

```powershell
uv run python src/ex8.py configs/ex8.yaml
```

Experiment 9 performs the same coordinate projection independently for every
Experiment 6 pose NPZ, without noise. One comparison PNG is written per video.

```powershell
uv run python src/ex9.py configs/ex9.yaml
```

Experiment 10 extracts 12 corresponding body points from C3D and MediaPipe:
shoulders, elbows, wrists, hips, knees, and ankles. Paired C3D wrist markers and
front/back waist markers are averaged for the wrist and hip positions. The
common subspace is estimated only from the noisy C3D points, and the video pose
coordinates are projected into that C3D-derived subspace.
The unprojected 12 video joints are also saved as
`out/ex10/video_original.png`, using the same configured frame as the projected
comparison.

```powershell
uv run python src/ex10.py configs/ex10.yaml
```

Experiments 8-10 initially create static side-by-side 3D plots for one
configured frame. `dimension` and `frame_index` can be changed in each matching
YAML file. Existing C3D, NPZ, and prior experiment outputs are not modified.

## Experiment 11: video differences in a clean-C3D subspace

`src/ex11.py` extracts the same 12 corresponding joints used by Experiment 10,
but estimates the common subspace from the original C3D without adding noise.
Each video pose shape subspace is projected into that clean-C3D subspace before
the Experiment 6 first-order, second-order, and geodestic analyses are run.

```powershell
uv run python src/ex11.py configs/ex11.yaml
```

GIFs are grouped under `out/ex11/origin/fiN` and
`out/ex11/c3d/dN/fiN`. Five `_origin.gif` variants use
the video shape subspaces without dimensional reduction. Five filenames contain
`_c3d` and the configured dimension and use the clean-C3D projection, allowing
the two conditions to be compared without overwriting Experiment 6 outputs.
New outputs include `_fi<frame_interval>` in their filenames, titles, and
geodestic legends so results from different temporal intervals can coexist.

## Experiment 12: Experiment 6 and 11 score comparison

`src/ex12.py` reuses the Experiment 6 pose-derived projection and the
Experiment 11 clean-C3D-derived projection for matching dimensions and frame
intervals. It recalculates the existing first-order, second-order, along-
geodestic, and orthogonal-geodestic scores, then overlays only the two score
graphs. It does not run pose estimation or modify prior outputs.

```powershell
uv run python src/ex12.py configs/ex12.yaml
```

PNG files are grouped under `out/ex12/dN/fiN` and retain `_dN_fiN` in their
filenames.

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
