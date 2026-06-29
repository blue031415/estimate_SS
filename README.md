# Shape Subspaceのノイズ耐性の検証
## ex. 1 固有空間への射影
### 目的
歩行空間の固有空間の次元を調べる。
### データ
- data : `data/07_01.c3d`
### 設定
- 全フレームの形状部分空間にたいして固有空間を生成。寄与率99%を初めて超えるのは5次元  
→5次元部分空間へ射影
### 結果
![Eigenvalue](out\ex1\eigen_val.png)
```
frame num: 316
selected indices: [0 1 2 3 4]
cumulative contribution: 0.9924324637361394
average magnitude: 0.022802185180121167 ↓
max magnitude: 0.06027836904286654 ↓ at frame 219
min magnitude: 0.00451280091887174 ↓ at frame 161
```

## ex. 2 固有空間内での差分部分空間と2階差分部分空間の波形生成
### データ
- data : `data/07_01.c3d`
### 設定
- 全フレームの形状部分空間にたいして固有空間を生成。寄与率99%を初めて超えるのは5次元  
- 固有空間の次元を変えて実験してみる。
- contributionの計算は各フレームごとに正規化