from ultralytics import YOLO
import os
import cv2
import easyocr
import re
import numpy as np
import torch
from torchvision.transforms import functional as F

torch.device("cpu")
model_path = os.path.join(".", "runs", "detect", "train3", "weights", "best.pt")
model = YOLO(model_path)
chrs_model_path = os.path.join(".", "runs", "detect", "train5", "weights", "best.pt")
chrs_model = YOLO(chrs_model_path)

image_path = "images/car1.jpg"


import os
import cv2
import torch
from torchvision.transforms import functional as F

# Assuming YOLO class is implemented and available
# from your_yolo_module import YOLO

image_path = "images/car1.jpg"
model_path = os.path.join(".", "runs", "detect", "train3", "weights", "best.pt")
chrs_model_path = os.path.join(".", "runs", "detect", "train5", "weights", "best.pt")

model = YOLO(model_path)
chrs_model = YOLO(chrs_model_path)


def read_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print("Error reading image.")
            return None
        return img
    except Exception as e:
        print(f"Error during image reading: {e}")
        return None


# Resize the input image to a size compatible with YOLO model
def resize_image(img, target_size):
    img = cv2.resize(img, target_size)
    return img


# Convert YOLO detections to bounding box coordinates
def yolo_to_bbox(yolo_detections, img_width, img_height):
    bboxes = []
    for detection in yolo_detections:
        x, y, w, h = detection["bbox"]
        x1 = max(0, int((x - w / 2) * img_width))
        y1 = max(0, int((y - h / 2) * img_height))
        x2 = min(int((x + w / 2) * img_width), img_width)
        y2 = min(int((y + h / 2) * img_height), img_height)
        bboxes.append((x1, y1, x2, y2))
    return bboxes


# Make predictions using the YOLO model to detect license plates
input_image = read_image(image_path)

# Resize the input image to a compatible size
target_size = (640, 640)
resized_image = resize_image(input_image, target_size)

# Convert the resized image to a PyTorch tensor
input_tensor = F.to_tensor(resized_image).unsqueeze(0)

# Forward pass for detection using the YOLO model
with torch.no_grad():
    yolo_detections = model(input_tensor)

# Extract bounding boxes from YOLO detections
img_height, img_width, _ = resized_image.shape
bounding_boxes = yolo_to_bbox(yolo_detections, img_width, img_height)

# For each detected license plate, find characters using the chrs_model
for i, box in enumerate(bounding_boxes):
    x1, y1, x2, y2 = box
    license_plate_roi = resized_image[y1:y2, x1:x2]

    # Forward pass for detection using the character YOLO model
    chrs_input_tensor = F.to_tensor(license_plate_roi).unsqueeze(0)
    with torch.no_grad():
        character_detections = chrs_model(chrs_input_tensor)

    # Process the character detections as needed
    # (e.g., draw bounding boxes, extract characters, etc.)
    for char_detection in character_detections:
        char_bbox = char_detection["bbox"]
        char_x1, char_y1, char_x2, char_y2 = char_bbox
        char_roi = license_plate_roi[char_y1:char_y2, char_x1:char_x2]

        # Now you can further process the character region as needed
        # ...

        # For example, display the character region
        cv2.imshow(f"Character {i+1}", char_roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# def read_image(img_path):
#     try:
#         cap = cv2.VideoCapture(img_path)
#         ret, image = cap.read()
#         cap.release()
#         if not ret:
#             print("Error reading image.")
#             return None
#         return image
#     except Exception as e:
#         print(f"Error during image reading: {e}")
#         return None


# def img_to_tresh(good_img):
#     detections = model(good_img)[0]
#     img_tresh = [
#         cv2.threshold(
#             cv2.cvtColor(
#                 image[int(y1) : int(y2), int(x1) : int(x2)], cv2.COLOR_BGR2GRAY
#             ),
#             64,
#             255,
#             cv2.THRESH_BINARY_INV,
#         )[1]
#         for x1, y1, x2, y2, _, _ in detections.boxes.data.tolist()
#     ]
#     return img_tresh


# def img_cnts(tresh_img):
#     reader = easyocr.Reader(["en"], gpu=False, verbose=False)
#     cnts, _ = cv2.findContours(
#         np.vstack(tresh_img).astype(np.uint8),
#         cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_SIMPLE,
#     )
#     cnt = max(cnts, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(cnt)
#     output = reader.readtext(np.vstack(tresh_img), paragraph=False)
#     return output

# def return_txt(cnts_img):
#     license_plate_list = [
#         {"text": re.sub(r"[^A-Z0-9]", "", text), "confidence": text_score}
#         for _, text, text_score in cnts_img
#         if text_score > 0.25
#     ]

#     result = {"plate": license_plate_list}
#     return result


# image_path = "images/car1.jpg"
# image = read_image(image_path)
# img_tresh = img_to_tresh(image)
# img_conts = img_cnts(img_tresh)
# result = return_txt(img_conts)
# print(result)
