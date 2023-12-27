from ultralytics import YOLO
import os
import cv2
import easyocr

import numpy as np
import torch
import time
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

    def _read_image_to_tresh(self):
        cap = cv2.VideoCapture(self.img_path)
        _, image = cap.read()
        cap.release()
        with torch.no_grad():
            detections = model(image)[0]

        self.img_tresh = [
            self._process_detection(image, box)
            for box in detections.boxes.data.tolist()
        ]

    def _process_detection(self, image, box):
        x1, y1, x2, y2, _, _ = box

        region_of_interest = image[int(y1) : int(y2), int(x1) : int(x2)]
        gray_image = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2GRAY)
        enhanced_image = cv2.convertScaleAbs(gray_image, alpha=1.5, beta=0)
        # enhance = self._img_enhance(region_of_interest)

        # _, thresholded = cv2.threshold(gray_image, 64, 255, cv2.THRESH_BINARY_INV)
        _, tresh = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2RGB))
        plt.title("Original Region")

        plt.subplot(2, 2, 2)
        plt.imshow(gray_image)
        plt.title("Gray Image")

        plt.subplot(1, 2, 1)
        plt.imshow(enhanced_image)
        plt.title("Gray after filter Image")

        plt.subplot(1, 2, 2)
        plt.imshow(tresh)
        plt.title("Thresholded Image")

        plt.show()

        return tresh

    # ideti filtra

    def _img_enhance(self, img_cnts):
        gray_image = cv2.cvtColor(img_cnts, cv2.COLOR_BGR2GRAY)
        enhanced_image = cv2.convertScaleAbs(gray_image, alpha=1.5, beta=0)
        return enhanced_image

    def img_cnts(self):
        cnts, _ = cv2.findContours(
            np.vstack(self.img_tresh).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cnt = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        output = self.reader.readtext(
            np.vstack(self.img_tresh), allowlist=self.allow_list, paragraph=False
        )
        return output

    def return_txt(self):
        t0 = time.time()
        self._read_image_to_tresh()
        output = self.img_cnts()
        text = [x[1] for x in output]
        confid = [x[2] for x in output]
        text = "".join(text) if len(text) > 0 else None
        confid = np.round(np.mean(confid), 2) if len(confid) > 0 else None
        t1 = time.time()
        print(
            f"Recognized number: {text}, conf.:{confid}.\nOCR total time: {(t1 - t0):.3f}s"
        )


image_path = "images/car5.jpeg"
detection = detect_license_plate(image_path)
result = detection.return_txt()

# print(dir(cv2))
