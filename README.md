# Shape Subspaceによるモーション解析

モーションキャプチャデータや動画から推定した3次元姿勢を形状部分空間として表現し、
共通部分空間への射影、時間差分、測地線分解、ノイズの影響を評価する実験用リポジトリです。

## 実行方法

依存関係を準備します。

```powershell
uv sync
```

各実験は、対応するYAML設定ファイルを指定して実行します。

```powershell
uv run python src/ex3.py configs/ex3.yaml
```

YAMLを省略した場合は、各スクリプトの `DEFAULT_CONFIG` に定義された既定値を使用します。
YAML内の相対パスは、そのYAMLファイルが置かれたディレクトリを基準に解決されます。
詳しい設定と出力については [実験ガイド](docs/experiments.md) を参照してください。

## 実験1 共通部分空間への射影

### 目的

C3Dモーションから共通部分空間を構築し、固有値の累積寄与率と各フレームの射影誤差を確認する。

### データ

- `data/07_01.c3d`

### 設定

- 設定ファイル: `configs/ex1.yaml`
- 実行スクリプト: `src/ex1.py`
- 主な設定: `contribution_threshold`

### 実行

```powershell
uv run python src/ex1.py configs/ex1.yaml
```

### 結果

## 実験2 部分空間の時間差分と測地線分解

### 目的

元の形状部分空間と共通部分空間へ射影した形状部分空間について、1階差分、2階差分、
測地線方向成分、測地線直交成分を比較する。

### データ

- `data/07_01.c3d`

### 設定

- 1階差分: `src/ex2_first.py`、`src/ex2_first_origin.py`
- 1階差分の次元比較: `src/ex2_first_overlay.py`
- 2階差分: `src/ex2_second.py`、`src/ex2_second_origin.py`
- 2階差分の次元比較: `src/ex2_second_overlay.py`
- 測地線分解: `src/ex2_geodestic.py`、`src/ex2_geodestic_origin.py`
- 主な設定: `dimension`、`dimensions`、`frame_interval`

### 実行

```powershell
uv run python src/ex2_first.py configs/ex2_first.yaml
```

### 結果

## 実験3 ノイズを加えたC3Dの部分空間解析

### 目的

指定した身体部位とフレーム範囲にガウスノイズを加え、共通部分空間の固有値、
2階差分、測地線方向・直交方向の変化を評価する。

### データ

- `data/07_01.c3d`

### 設定

- 固有値と射影誤差: `src/ex3.py`、`configs/ex3.yaml`
- 2階差分: `src/ex3_second.py`、`configs/ex3_second.yaml`
- 測地線分解: `src/ex3_geodestic.py`、`configs/ex3_geodestic.yaml`
- 主な設定: `noise.start_frame`、`noise.end_frame`、`noise.scale`、`noise.body_parts`

### 実行

```powershell
uv run python src/ex3.py configs/ex3.yaml
```

### 結果

## 実験4 MPI-INF-3DHP動画の3次元姿勢推定

### 目的

MPI-INF-3DHPの動画から3次元骨格を推定し、数値データと入力動画・推定骨格の比較GIFを保存する。

### データ

- `data/mpi_inf_3dhp/`
- `data/models/pose_landmarker_lite.task`

### 設定

- 設定ファイル: `configs/ex4.yaml`
- 実行スクリプト: `src/ex4.py`
- 推定器: MediaPipeまたはMMPose
- 主な設定: `input_videos`、`estimator`、`frame_stride`、`max_frames`

### 実行

```powershell
uv run python src/ex4.py configs/ex4.yaml
```

### 結果

## 実験5 GaHu-Videoの3次元姿勢推定

### 目的

GaHu-VideoのS001動画に実験4と同じ姿勢推定処理を適用し、NPZと比較GIFを保存する。

### データ

- `data/GaHu-Video/S001.avi`
- `data/models/pose_landmarker_lite.task`

### 設定

- 設定ファイル: `configs/ex5.yaml`
- 実行スクリプト: `src/ex5.py`
- 主な設定: `estimator`、`frame_stride`、`max_frames`

### 実行

```powershell
uv run python src/ex5.py configs/ex5.yaml
```

### 結果

## 実験6 自撮り動画の姿勢推定と部分空間解析

### 目的

自撮り動画から3次元姿勢を推定し、元の形状部分空間と姿勢データから構築した共通部分空間について、
1階差分、2階差分、測地線方向・直交方向の評価値を算出する。

### データ

- `data/self_videos/*.mp4`
- `data/models/pose_landmarker_lite.task`

### 設定

- 姿勢推定と解析: `src/ex6.py`、`configs/ex6.yaml`
- 保存済みNPZの再解析: `src/ex6_analysis.py`、`configs/ex6_analysis.yaml`
- 主な設定: `first_dimension`、`second_dimension`、`geodestic_dimension`、`frame_interval`
- 現在の比較条件: `frame_interval = [2, 3]`、`dimension = [4, 5, 6]`

### 実行

```powershell
uv run python src/ex6.py configs/ex6.yaml
```

### 結果

## 実験7 姿勢推定結果の固有値解析

### 目的

実験6で保存した姿勢推定結果から共通部分空間を構築し、固有値の累積寄与率と射影誤差を確認する。

### データ

- `out/ex6/*.npz`

### 設定

- 設定ファイル: `configs/ex7.yaml`
- 実行スクリプト: `src/ex7.py`
- 主な設定: `contribution_threshold`

### 実行

```powershell
uv run python src/ex7.py configs/ex7.yaml
```

### 結果

## 実験8 ノイズ付きC3D座標の射影

### 目的

ノイズを加えたC3Dマーカー座標を共通部分空間へ射影し、指定フレームの射影前後を比較する。

### データ

- `data/07_01.c3d`

### 設定

- 設定ファイル: `configs/ex8.yaml`
- 実行スクリプト: `src/ex8.py`
- 主な設定: `dimension`、`frame_index`、`noise`

### 実行

```powershell
uv run python src/ex8.py configs/ex8.yaml
```

### 結果

## 実験9 動画姿勢座標の共通部分空間への射影

### 目的

実験6の姿勢推定座標から共通部分空間を構築し、指定フレームの射影前後を比較する。

### データ

- `out/ex6/*.npz`

### 設定

- 設定ファイル: `configs/ex9.yaml`
- 実行スクリプト: `src/ex9.py`
- 主な設定: `dimension`、`frame_index`

### 実行

```powershell
uv run python src/ex9.py configs/ex9.yaml
```

### 結果

## 実験10 C3D由来共通部分空間への動画姿勢の射影

### 目的

C3Dと動画姿勢から対応する12関節を抽出し、動画姿勢座標をノイズ付きC3Dから構築した
共通部分空間へ射影して、射影前後を比較する。

### データ

- `data/07_01.c3d`
- `out/ex6/歩行動作3.npz`

### 設定

- 設定ファイル: `configs/ex10.yaml`
- 実行スクリプト: `src/ex10.py`
- 主な設定: `dimension`、`frame_index`、`noise`

### 実行

```powershell
uv run python src/ex10.py configs/ex10.yaml
```

### 結果

## 実験11 クリーンなC3D由来共通部分空間での動画差分解析

### 目的

ノイズを加えていないC3Dから共通部分空間を構築し、動画姿勢の形状部分空間を射影する。
射影前後について、1階差分、2階差分、測地線方向・直交方向の評価値を比較する。

### データ

- `data/07_01.c3d`
- `out/ex6/*.npz`

### 設定

- 設定ファイル: `configs/ex11.yaml`
- 実行スクリプト: `src/ex11.py`
- 現在の比較条件: `frame_interval = [2, 3]`、`dimension = [4, 5, 6]`
- 出力構造: `out/ex11/fiN/origin`、`out/ex11/fiN/c3d/dN`

### 実行

```powershell
uv run python src/ex11.py configs/ex11.yaml
```

### 結果

## 実験12 実験6と実験11の評価値比較

### 目的

姿勢推定データから構築した共通部分空間を使う実験6と、クリーンなC3Dから構築した
共通部分空間を使う実験11について、同じ次元・フレーム間隔の評価値を重ねて比較する。

### データ

- `data/07_01.c3d`
- `out/ex6/*.npz`

### 設定

- 設定ファイル: `configs/ex12.yaml`
- 実行スクリプト: `src/ex12.py`
- 現在の比較条件: `frame_interval = [2, 3]`、`dimension = [4, 5, 6]`
- 出力構造: `out/ex12/fiN/dN`

### 実行

```powershell
uv run python src/ex12.py configs/ex12.yaml
```

### 結果
