# begining
from ultralytics import YOLO
import os
import cv2
import easyocr
import numpy as np
import csv
import uuid


def detect_on_img(img_path) -> None:
    # Specify the path to the custom model.
    model_path = os.path.join(".", "runs", "detect", "train2", "weights", "best.pt")

    # Load the model from the model path
    model = YOLO(model_path)

    # Define the threshold for good detections.
    threshold = 0.7

    # Read the image.
    img = cv2.imread(img_path)

    

    # Make detections on the image.
    results = model(img)[0]

    # Loop through found detections.
    for result in results.boxes.data.tolist():
        # Get the bounding box coordinates, detection scores, and class id's of found detections.
        x1, y1, x2, y2, score, class_id = result

        # Define bounding boxes for detections.
        region = img[int(y1) : int(y2), int(x1) : int(x2)]

        # If the score is better than our threshold for good detections...
        if score > threshold:
            # and only if the detection is a license plate
            if int(class_id) == 0:
                # Create a Reader and have it read the text on the license plate.
                reader = easyocr.Reader(["en"])
                ocr_result = reader.readtext(region)

                # Get the plate number only using the filter_text function.
                text = filter_text(region, ocr_result, 0.5)
                print(text)

    # Show the original image with all detections found using the custom model.
    # while True:
    #     cv2.imshow("out", img)
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break
    # cv2.imwrite(img_path_out, img)
    # cv2.destroyAllWindows()
    # print(text)


def filter_text(region: np.ndarray, ocr_result: list, region_threshold: float) -> list:
    # Calculate the size (area) of the region of interest
    rectangle_size = region.shape[0] * region.shape[1]

    # Initialize an empty list called plate to store filtered plate numbers.
    plate = []
    # Iterate through each OCR result in the list of results.
    for result in ocr_result:
        # Calculate the length and height of the bounding box of the detected text.
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        # Check if the area of the bounding box is greater than the specified threshold.
        if length * height / rectangle_size > region_threshold:
            # If the text meets the criteria, it is added to the list of filtered plate numbers.
            plate.append(result[1])

    # Return the list of filtered plate numbers.
    return plate


def main():
    img_path = "images/car1.png"
    detect_on_img(img_path)


if __name__ == "__main__":
    main()
