from typing import List, Dict

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

class EmotionRecognizer:
    EMOTION_LIST = [
        "angry",
        "disgust",
        "fearful",
        "happy",
        "neutral",
        "sad",
        "surprised"
    ]


    def __init__(self, model_path: str, input_shape: List = [112, 112], num_threads: int = 1) -> None:
        """Initialize

        Args:
            model_path (str): モデルパス
            input_shape (List, optional): 入力サイズ. Defaults to [112, 112].
            num_threads (int, optional): 推論スレッド数. Defaults to 1.
        """
        self._input_shape = input_shape

        self.interpreter = tflite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self._mean = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, :]
        self._std = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, :]

    def process(self, bgr_image: np.ndarray, bboxes: List) -> List:
        """感情認識

        Args:
            bgr_image (np.ndarray): 入力画像
            bboxes (List): 顔BBOXのリスト

        Returns:
            List: [{感情ラベル: 推論値, ... }, ...]
        """
        ret_list = []

        for bbox in bboxes:
            image = self._preprocess(bgr_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :])
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                image,
            )
            self.interpreter.invoke()

            result = self.interpreter.get_tensor(
                self.output_details[0]['index'])
            ret = self._postprocess(result)
            ret_list.append(ret)
        return ret_list

    def _preprocess(self, bgr_image: np.ndarray) -> np.ndarray:
        """前処理

        Args:
            bgr_image (np.ndarray): 入力画像
        Returns:
            np.ndarray: 前処理後画像
        """
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self._input_shape)
        image = image.astype(np.float32, copy=False) / 255.0
        image -= self._mean
        image /= self._std
        return np.expand_dims(image, axis=0)

    def _postprocess(self, results: List) -> Dict:
        """後処理

        Args:
            results (List): 推論結果

        Returns:
            Dict: {感情ラベル: 推論値, ... }
        """
        ret = {}
        prob = np.exp(results[0]) / np.sum(np.exp(results[0]))
        for emo_label, prob in zip(self.EMOTION_LIST, prob):
            ret[emo_label] = prob
        return ret