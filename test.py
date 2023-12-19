import cv2

# Your image path
image_path = "images/car6.jpeg"

# Load the image
image = cv2.imread(image_path)

# Define the plate holder region coordinates
plate_holder_region = [100, 500, 300, 400]

# Draw a rectangle on the image to visualize the region
cv2.rectangle(
    image,
    (plate_holder_region[0], plate_holder_region[1]),
    (plate_holder_region[2], plate_holder_region[3]),
    (0, 255, 0),
    2,
)

# Display the image with the drawn rectangle
cv2.imshow("Image with Plate Holder Region", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
