# facial_expression_recognition_mobilefacenet_example

## Overview
facial_expression_recognitionのPythonでのTFLITE推論サンプルです。

（OXXNモデルが追加されたらONNX版も作るかも）

## Environment
- Windows 10 Home
- Poetry

## Usage
### 環境構築
```
$poetry install --no-root
```

Poetry環境がない場合は以下を実行する。

```
$pip install -r requirements.txt
```

### 実行方法

PINTOさんの[PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)から以下のTfliteモデルをダウンロードする。

- [YuNet](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/144_YuNet)
- [facial_expression_recognition_mobilefacenet](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/346_facial_expression_recognition_mobilefacenet)

ダウンロードが完了したら以下コマンドを実行する。

```shell
$poetry run python -m emotion_recog --face_model {YuNetのモデルパス} --emo_model {facial_expression_recognition_mobilefacenetのモデルパス}

# 以下コマンドオプション一覧
$poetry run python -m emotion_recog -h
usage: __main__.py [-h] [--device DEVICE] [--video VIDEO] [--face_model FACE_MODEL] [--emo_model EMO_MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE       カメラデバイスID
  --video VIDEO         動画ファイルのパス（指定された場合これが優先される）
  --face_model FACE_MODEL
                        顔検出yunetモデルファイルのパス
  --emo_model EMO_MODEL
                        感情認識モデルファイルのパス
```

## Author
T-Sumida

## Reference
- [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
- [YuNet-ONNX-TFLite-Sample](https://github.com/Kazuhito00/YuNet-ONNX-TFLite-Sample)


## License
[Apache-2.0 license](https://github.com/T-Sumida/e2pose_example/blob/main/LICENSE)
