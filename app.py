from ultralytics import YOLO
import os
import cv2
import easyocr
import re
import numpy as np


def preprocess_image_for_ocr(img_path) -> None:
    # Load image from path
    image = cv2.imread(img_path)

    # Image enhancement and noise reduction (add these steps)

    # Set trained model path from repo
    model_path = os.path.join(".", "runs", "detect", "train2", "weights", "best.pt")

    # Load trained model with YOLO
    model = YOLO(model_path)

    # Find all detected license plates from image
    detections = model(image)[0]

    # Loop all detections and convert image to gray, return crooped and gray license plate after filter
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        cropped_plate = image[int(y1) : int(y2), int(x1) : int(x2)]
        gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        gray_plate = cv2.bilateralFilter(gray_plate, 11, 17, 17)

    return gray_plate


def enhance_image_contrast(image):
    alpha = 1.5
    beta = 0
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced_image


def enhance_image_brightness(image):
    alpha = 1.0
    beta = 50
    enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced_image


def enhance_image_sharpening(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    enhanced_image = cv2.filter2D(image, -1, kernel)
    return enhanced_image


def detect_on_img(image) -> None:
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    reader = easyocr.Reader(["en"], gpu=False)
    ocr_result = reader.readtext(thresh, paragraph=False)

    if ocr_result:
        license_plate_text = ocr_result[0][1]
        conf_score = ocr_result[0][-1]
        license_plate_text = re.sub(r"[^A-Z0-9]", "", license_plate_text)
        result = {
            "plate": license_plate_text,
            "score": conf_score,
        }
    else:
        result = {
            "plate": "",
            "score": 0.0,
        }
    return result


image_path = "images/image9.jpg"
image = preprocess_image_for_ocr(image_path)
image = enhance_image_contrast(image)
image = enhance_image_brightness(image)
image = enhance_image_sharpening(image)

text = detect_on_img(image)
print(text)
