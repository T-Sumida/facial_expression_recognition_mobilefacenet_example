from typing import List, Tuple
from itertools import product

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


class YuNet:
    # Feature map用定義
    MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    STEPS = [8, 16, 32, 64]
    VARIANCE = [0.1, 0.2]


    def __init__(
        self,
        model_path: str,
        input_shape: List = [160, 120],
        conf_th: float = 0.6,
        nms_th: float = 0.3,
        topk: int = 5000,
        keep_topk: int = 750,
        num_threads: int= 1
    ) -> None:
        """Initialize

        Args:
            model_path (str): モデルパス
            input_shape (List, optional): 入力画像のサイズ. Defaults to [160, 120].
            conf_th (float, optional): 検出閾値 Defaults to 0.6.
            nms_th (float, optional): NMS閾値. Defaults to 0.3.
            topk (int, optional): _description_. Defaults to 5000.
            keep_topk (int, optional): _description_. Defaults to 750.
            num_threads (int, optional): _description_. Defaults to 1.
        """
        self._input_shape = input_shape
        self._conf_th = conf_th
        self._nms_th = nms_th
        self._topk = topk
        self._keep_topk = keep_topk

        self.interpreter = tflite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self._generate_priors()

    def process(self, bgr_image: np.ndarray) -> Tuple[List, List, List]:
        """顔検出を行う

        Args:
            bgr_image (np.ndarray): 入力画像

        Returns:
            Tuple[List, List, List]: ([[x, y, width, height], ...], [[x, y], ...], [score1, ...])
        """
        h, w, _ = bgr_image.shape
        image = self._preprocess(bgr_image)
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            image,
        )
        self.interpreter.invoke()

        result_01 = self.interpreter.get_tensor(
            self.output_details[2]['index'])
        result_02 = self.interpreter.get_tensor(
            self.output_details[0]['index'])
        result_03 = self.interpreter.get_tensor(
            self.output_details[1]['index'])
        result = [
            np.array(result_01),
            np.array(result_02),
            np.array(result_03)
        ]

        bboxes, landmarks, scores = self._postprocess(result, w, h)

        return bboxes, landmarks, scores

    def _preprocess(self, bgr_image: np.ndarray) -> np.ndarray:
        """前処理

        Args:
            bgr_image (np.ndarray): 入力画像
        Returns:
            np.ndarray: 前処理後画像
        """
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # リサイズ
        image = cv2.resize(
            image,
            (self._input_shape[0], self._input_shape[1]),
            interpolation=cv2.INTER_LINEAR,
        )

        # リシェイプ
        image = image.astype(np.float32)
        image = image.reshape(1, self._input_shape[1], self._input_shape[0], 3)

        return image

    def _postprocess(self, results: List, original_width: int, original_height: int) -> Tuple[List, List, List]:
        """後処理

        Args:
            results (List): 推論結果
            original_width (int): 入力画像の幅
            original_height (int): 入力画像の高さ
        Returns:
            Tuple[List, List, List]: ([[x, y, width, height], ...], [[x, y], ...], [score1, ...])
        """
        dets = self._decode(results)

        # NMS
        keepIdx = cv2.dnn.NMSBoxes(
            bboxes=dets[:, 0:4].tolist(),
            scores=dets[:, -1].tolist(),
            score_threshold=self._conf_th,
            nms_threshold=self._nms_th,
            top_k=self._topk,
        )

        # bboxes, landmarks, scores へ成形
        scores = []
        bboxes = []
        landmarks = []
        if len(keepIdx) > 0:
            dets = dets[keepIdx]
            if len(dets.shape) == 3:
                dets = np.squeeze(dets, axis=1)
            for det in dets[:self._keep_topk]:
                tmp_bbox = det[0:4].astype(np.int32)
                tmp_landmarks = det[4:14].astype(np.int32).reshape((5, 2))

                x1 = int(original_width * (tmp_bbox[0] / self._input_shape[0]))
                y1 = int(original_height * (tmp_bbox[1] / self._input_shape[1]))
                width = int(original_width * (tmp_bbox[2] / self._input_shape[0]))
                height = int(original_height * (tmp_bbox[3] / self._input_shape[1]))

                for i in range(tmp_landmarks.shape[0]):
                    tmp_landmarks[i][0] = int(original_width * (tmp_landmarks[i][0] / self._input_shape[0]))
                    tmp_landmarks[i][1] = int(original_height * (tmp_landmarks[i][1] / self._input_shape[1]))

                scores.append(det[-1])
                bboxes.append([x1, y1, width, height])
                landmarks.append(tmp_landmarks)

        return bboxes, landmarks, scores

    def _decode(self, result: List) -> np.ndarray:
        loc, conf, iou = result

        # スコア取得
        cls_scores = conf[:, 1]
        iou_scores = iou[:, 0]

        _idx = np.where(iou_scores < 0.)
        iou_scores[_idx] = 0.
        _idx = np.where(iou_scores > 1.)
        iou_scores[_idx] = 1.
        scores = np.sqrt(cls_scores * iou_scores)
        scores = scores[:, np.newaxis]

        scale = np.array(self._input_shape)

        # バウンディングボックス取得
        bboxes = np.hstack(
            ((self._priors[:, 0:2] +
              loc[:, 0:2] * self.VARIANCE[0] * self._priors[:, 2:4]) * scale,
             (self._priors[:, 2:4] * np.exp(loc[:, 2:4] * self.VARIANCE)) *
             scale))
        bboxes[:, 0:2] -= bboxes[:, 2:4] / 2

        # ランドマーク取得
        landmarks = np.hstack(
            ((self._priors[:, 0:2] +
              loc[:, 4:6] * self.VARIANCE[0] * self._priors[:, 2:4]) * scale,
             (self._priors[:, 0:2] +
              loc[:, 6:8] * self.VARIANCE[0] * self._priors[:, 2:4]) * scale,
             (self._priors[:, 0:2] +
              loc[:, 8:10] * self.VARIANCE[0] * self._priors[:, 2:4]) * scale,
             (self._priors[:, 0:2] +
              loc[:, 10:12] * self.VARIANCE[0] * self._priors[:, 2:4]) * scale,
             (self._priors[:, 0:2] +
              loc[:, 12:14] * self.VARIANCE[0] * self._priors[:, 2:4]) * scale))

        dets = np.hstack((bboxes, landmarks, scores))

        return dets

    def _generate_priors(self):
        w, h = self._input_shape

        feature_map_2th = [
            int(int((h + 1) / 2) / 2),
            int(int((w + 1) / 2) / 2)
        ]
        feature_map_3th = [
            int(feature_map_2th[0] / 2),
            int(feature_map_2th[1] / 2)
        ]
        feature_map_4th = [
            int(feature_map_3th[0] / 2),
            int(feature_map_3th[1] / 2)
        ]
        feature_map_5th = [
            int(feature_map_4th[0] / 2),
            int(feature_map_4th[1] / 2)
        ]
        feature_map_6th = [
            int(feature_map_5th[0] / 2),
            int(feature_map_5th[1] / 2)
        ]

        feature_maps = [
            feature_map_3th, feature_map_4th, feature_map_5th, feature_map_6th
        ]

        priors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self.MIN_SIZES[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / w
                    s_ky = min_size / h

                    cx = (j + 0.5) * self.STEPS[k] / w
                    cy = (i + 0.5) * self.STEPS[k] / h

                    priors.append([cx, cy, s_kx, s_ky])

        self._priors = np.array(priors, dtype=np.float32)
