import copy
import argparse
import logging
import time
from typing import List

import cv2
import numpy as np

from emotion_recog.yunet import YuNet
from emotion_recog.emotion_recognizer import EmotionRecognizer


def get_args() -> argparse.Namespace:
    """引数情報を取得
    Returns:
        argparse.Namespace: 引数情報
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0, help="カメラデバイスID")
    parser.add_argument("--video", type=str, default=None, help="動画ファイルのパス（指定された場合これが優先される）")
    parser.add_argument("--face_model", type=str, default="saved_model/model_float32.tflite", help="顔検出yunetモデルファイルのパス")
    parser.add_argument("--emo_model", type=str, default="facial_expression_recognition_mobilefacenet_2022july_float32.tflite", help="感情認識モデルファイルのパス")

    args = parser.parse_args()

    return args


def main(args: argparse.Namespace) -> None:
    """推論

    Args:
        args (argparse.Namespace): 引数情報
    """
    face_detector = YuNet(args.face_model)
    emo_recog = EmotionRecognizer(args.emo_model)

    if args.video is not None:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.device)

    try:
        while True:
            ret, image = cap.read()
            if not ret:
                break
            debug_image = copy.deepcopy(image)

            start_t = time.time()
            bboxes, landmarks, scores = face_detector.process(image)
            emotions = emo_recog.process(image, bboxes)
            elapsed_time = time.time() - start_t
            
            debug_image = draw_debug(debug_image, bboxes, scores, landmarks, emotions, elapsed_time)

            cv2.imshow("facial expression example", debug_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        logging.info("press ctrl+c")
    except Exception as e:
        logging.error(e)
    finally:
        cap.release()


def draw_debug(image: np.ndarray, bboxes: List, scores: List, landmarks: List, emotions: List, elapsed_time: float) -> np.ndarray:
    """デバッグ情報を描画する

    Args:
        image (np.ndarray): 画像
        bboxes (List): 顔BBOXのリスト
        scores (List): 顔BBOXの検出スコアのリスト
        landmarks (List): 顔ランドマークのリスト
        emotions (List): 感情認識結果のリスト
        elapsed_time (float): 推論時間[sec]

    Returns:
        np.ndarray: 描画済み画像
    """
    
    for bbox, score, landmark, emotion in zip(bboxes, scores, landmarks, emotions):
        # draw face bbox
        cv2.rectangle(
            image,
            (bbox[0], bbox[1], bbox[2], bbox[3]),
            (139, 70, 195),
            2,
        )

        # draw face detect score
        cv2.putText(
            image,
            f"score: {float(score):.2f}",
            (bbox[0], bbox[1]-10),
            0,
            0.5,
            (139, 70, 195),
            2,
        )

        # draw face landmark
        for land_x, land_y in landmark:
            cv2.circle(
                image, (land_x, land_y),
                1,
                (139, 70, 195),
                2
            )

        # draw emotin
        dy = 15
        for emo_label, prob in emotion.items():
            cv2.putText(
                image,
                f"{emo_label}: ",
                (bbox[0], int(bbox[1] + bbox[3] + dy)),
                0,
                0.5,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                image,
                f":{prob:.2f}",
                (bbox[0]+80, int(bbox[1] + bbox[3] + dy)),
                0,
                0.5,
                (255, 255, 255),
                2,
            )
            dy += 15

        text = f"Elapsed Time: {(elapsed_time * 1000):.1f}[ms]"
        image = cv2.putText(
            image,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            thickness=2,
        )
    return image


if __name__ == "__main__":
    main(get_args())
