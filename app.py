from ultralytics import YOLO
import os
import cv2
import easyocr
import re
import numpy as np
import torch

torch.device("cpu")
model_path = os.path.join(".", "runs", "detect", "train3", "weights", "best.pt")
model = YOLO(model_path)


class detect_license_plate:
    def __init__(self, img_path) -> None:
        self.img_path = img_path
        self.result = None

    def _read_image(self):
        try:
            cap = cv2.VideoCapture(self.img_path)
            ret, image = cap.read()
            cap.release()
            if not ret:
                print("Error reading image.")
                return None
            return image
        except Exception as e:
            print(f"Error during image reading: {e}")
            return None

    def _img_to_tresh(self):
        image = self._read_image()
        if image is None:
            self.img_tresh = []
            return

        try:
            detections = model(image)[0]
            self.img_tresh = [
                cv2.threshold(
                    cv2.cvtColor(
                        image[int(y1) : int(y2), int(x1) : int(x2)], cv2.COLOR_BGR2GRAY
                    ),
                    64,
                    255,
                    cv2.THRESH_BINARY_INV,
                )[1]
                for x1, y1, x2, y2, _, _ in detections.boxes.data.tolist()
            ]
        except Exception as e:
            print(f"Error during image processing: {e}")
            self.img_tresh = []

    def img_cnts(self):
        if not hasattr(self, "img_tresh") or not self.img_tresh:
            return []

        try:
            reader = easyocr.Reader(["en"], gpu=False)
            cnts, _ = cv2.findContours(
                np.vstack(self.img_tresh).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cnt = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            output = reader.readtext(np.vstack(self.img_tresh), paragraph=False)
        except Exception as e:
            print(f"Error during contour extraction or OCR: {e}")
            output = []

        return output

    def return_txt(self):
        try:
            self._img_to_tresh()
            output = self.img_cnts()
            license_plate_list = [
                re.sub(r"[^A-Z0-9]", "", text)
                for _, text, text_score in output
                if text_score > 0.25
            ]
        except Exception as e:
            print(f"Error during text processing: {e}")
            license_plate_list = []

        license_plate = "".join(license_plate_list)
        self.result = {"plate": license_plate}
        return self.result


# Usage
try:
    image_path = "images/car6.jpeg"
    detection = detect_license_plate(image_path)
    result = detection.return_txt()
    print(result)
except Exception as e:
    print(f"Error during license plate detection: {e}")
    # Handle the error or exit the program
