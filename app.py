from ultralytics import YOLO
import os
import cv2
import easyocr
import re
from typing import Dict


def detect_on_img(img_path: os.path) -> Dict:
    model_path = os.path.join(".", "runs", "detect", "train3", "weights", "best.pt")
    model = YOLO(model_path)

    image = cv2.imread(img_path)

    detections = model(image)[0]
    reader = easyocr.Reader(["en"], gpu=False)
    license_plate_list = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        cropped_plate = image[int(y1) : int(y2), int(x1) : int(x2)]
        gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)
        _, license_plate_thresh = cv2.threshold(
            gray_plate, 64, 255, cv2.THRESH_BINARY_INV
        )

    output = reader.readtext(license_plate_thresh, paragraph=False)

    for out in output:
        text_bbox, text, text_score = out
        if text_score > 0.25:
            text = re.sub(r"[^A-Z0-9]", "", text)
            license_plate_list.append(text)

    license_plate = "".join(license_plate_list)
    result = {"plate": license_plate}
    return result


image_path = "images/car6.jpeg"
detect = detect_on_img(image_path)
print(detect)


# cv2.imshow("License Plate Threshold", license_plate_thresh)
# key = cv2.waitKey(0) & 0xFF
# if key == 27 or key == ord("q"):
#     cv2.destroyAllWindows()
