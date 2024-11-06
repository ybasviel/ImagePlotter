# Image Plotter
## 概要

画像からエッジを検出してベクトル化して、gcodeでプロットするためのプログラム

CR10Sで動作確認済み


## 使い方


- 入力オプション
    - url
        ```shell
        -u https://example.com/xxx.jpg
        ```
    - ファイル
        ```shell
        -i ./xxx.jpg
        ```
    - カメラ
        オプション不要

- 出力オプション
    - ファイルに出力
        ```shell
        -o ./xxx.gcode
        ```
    - シリアルポート転送
        ```shell
        -s /dev/ttyUSB0
        ```

例: url入力、シリアルポート出力
```shell
python main.py -u https://pbs.twimg.com/profile_images/1560525787270631424/NLQvt7JG_400x400.jpg -s /dev/ttyUSB0
```

## 参考

- [ShyBoy233/PyGcodeSender: A simple python script to send gcode file using serial to various machines.](https://github.com/ShyBoy233/PyGcodeSender/tree/main)
- [@s-col(スコル) python+OpenCVで検出した画像のエッジを曲線(折れ線)に変換する[Python3] #画像処理 - Qiita](https://qiita.com/s-col/items/115b7f7d80133f89359d)