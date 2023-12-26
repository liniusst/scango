from ultralytics import YOLO
import os
import cv2
import easyocr
from typing import List
from itertools import chain
import numpy as np
import torch
import time
import re
import matplotlib.pyplot as plt

torch.device("cpu")
model_path = os.path.join(".", "runs", "detect", "train3", "weights", "best.pt")
model = YOLO(model_path)


class detect_license_plate:
    def __init__(self, img_path, lang=["en"]) -> None:
        self.allow_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.reader = easyocr.Reader(lang, gpu=False, verbose=False)
        self.img_path = img_path
        self.result = None

    def _read_image_to_tresh(self) -> List[dict]:
        cap = cv2.VideoCapture(self.img_path)
        _, image = cap.read()
        cap.release()
        with torch.no_grad():
            detections = model(image)[0]

        img_thresholded = [
            self._process_detection(image, box)
            for box in detections.boxes.data.tolist()
        ]

        return img_thresholded

    def _process_detection(self, image, box):
        x1, y1, x2, y2, _, _ = box

        region_of_interest = image[int(y1) : int(y2) + 5, int(x1) : int(x2) + 5]
        gray_image = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray_image, 11, 17, 17)  # Noise reduction
        _, thresholded = cv2.threshold(
            bfilter, 127, 255, cv2.THRESH_BINARY_INV
        )  # 64, 255

        # plt.subplot(2, 2, 1)
        # plt.imshow(cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2RGB))
        # plt.title("Original Region")

        # plt.subplot(2, 2, 2)
        # plt.imshow(gray_image)
        # plt.title("Gray Image")

        # plt.subplot(1, 2, 1)
        # plt.imshow(bfilter)
        # plt.title("Gray after filter Image")

        # plt.subplot(1, 2, 2)
        # plt.imshow(thresholded)
        # plt.title("Thresholded Image")

        # plt.show()

        return thresholded

    # ideti filtra

    def final_ocr(self):
        img_tresh = self._read_image_to_tresh()
        cnts, _ = cv2.findContours(
            np.vstack(img_tresh).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        # cnts, new = cv2.findContours(
        #     img_tresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        # )
        cnt = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        output = self.reader.readtext(
            np.vstack(img_tresh), allowlist=self.allow_list, paragraph=False
        )
        return output

    def final_dict(self):
        start_time = time.time()
        plate_number = self.final_ocr()
        text = [x[1] for x in plate_number]
        confid = [x[2] for x in plate_number]
        confid = np.round(np.mean(confid), 2) if len(confid) > 0 else None
        plate_number = "".join(chain.from_iterable(text)) if text else None
        end_time = time.time()
        ocr_time = f"{(end_time - start_time):.3f}s"
        result = {
            "ocr_time": ocr_time,
            "plate_number": plate_number,
            "conf_level": confid,
        }
        return result


# image_path = "images/car4.jpeg"
# detection = detect_license_plate(image_path)
# result = detection.final_dict()
# print(result)
