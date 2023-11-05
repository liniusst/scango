from ultralytics import YOLO
import os
import cv2
import easyocr
import re


def detect_on_img(img_path) -> None:
    image = cv2.imread(img_path)

    model_path = os.path.join(".", "runs", "detect", "train2", "weights", "best.pt")
    model = YOLO(model_path)

    threshold = 0.7

    detections = model(image)[0]

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        cropped_plate = image[int(y1) : int(y2), int(x1) : int(x2)]
        gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        gray_plate = cv2.bilateralFilter(gray_plate, 11, 17, 17)

        thresh = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[
            1
        ]
        cv2.imshow("Rotated", thresh)
        cv2.waitKey(0)

        if score >= threshold:
            reader = easyocr.Reader(["en"], gpu=False)
            ocr_result = reader.readtext(thresh, paragraph=False)
            license_plate_text = ocr_result[0][1]
            conf_score = ocr_result[0][-1]
            license_plate_text = re.sub(r"[^A-Z0-9]", "", license_plate_text)
            result = {
                "plate": license_plate_text,
                "score": conf_score,
            }
            return result
        else:
            return None


image_path = "images/image7.jpg"
new = detect_on_img(image_path)
print(new)
