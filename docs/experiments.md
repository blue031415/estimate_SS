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
```

共通モジュールは次の役割に限定しています。

- `experiments/analysis.py`: 部分空間に対する再利用可能な数値計算
- `experiments/noise.py`: マーカー選択とノイズ付加
- `utils/experiment_config.py`: YAML の読込、既定値とのマージ、パス解決

実験の順序や、どの計算を組み合わせるかは各 `ex*.py` に残しています。

