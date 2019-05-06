# PFN2019インターン選考課題 機械学習・数理分野
Preferred Networksの2019年のインターン選考課題(機械学習・数理分野)です。  
Graph Neural Networkをフレームワークを用いずに実装しました。

## はじめに
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ディレクトリ説明
- [datasets/](/datasets): 与えられたデータセット
- [log/](/log): 学習中の出力ログ(loss, accuracyの変化)
- [model/](/model): 学習済みモデル(pickle)
- [src/](/src): ソースコード
- [prediction.txt](/prediction.txt): 課題4でテストデータに対して予測したラベル
- [README-ja(en).pdf](/README-ja.pdf): 課題の説明pdf
- [report.pdf](/report.pdf): 課題レポート

## 各課題の実行方法
- 課題1  
    `python src/test.py`
- 課題2  
    `python src/kadai/kadai2.py`
- 課題3  
    `python src/kadai/kadai3.py`
- 課題4  
    - Adamによる学習  
        `python src/kadai/kadai4-train.py`
    - 隣接行列の正規化を加えたモデルでの学習  
        `python src/kadai/kadai4-plus.py`
    - taskに対するラベルの予測  
        `python src/kadai/kadai4.py`

## ハイパーパラメータについて
課題pdf及びChainerのデフォルト値を参考にしました。  
学習率については、収束に時間がかかったためpdfの値より大きい0.001に設定しました。