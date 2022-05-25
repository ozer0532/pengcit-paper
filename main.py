import argparse
import cv2
import numpy as np

from utils import aruco, transform, matching

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image containing ArUCo tag")
args = vars(ap.parse_args())

print("[INFO] loading image...")
image = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)

# === DETECT ARUCO ===
arucos = aruco.detect_aruco(image)
arucos = list(sorted(arucos, key=lambda arc: arc.id))
image = aruco.draw_aruco(image, arucos)

# === IMAGE WARP/CROP ===
# image = transform.four_point_transform(
#     image, arucos[0].topLeft, arucos[1].topRight, arucos[3].bottomRight, arucos[2].bottomLeft
# )
image = transform.four_point_transform(
    image, arucos[3].bottomRight, arucos[2].bottomLeft, arucos[0].topLeft, arucos[1].topRight
)
image = transform.crop(image, 0, 3 * 30, 17 * 30, 20 * 30)

# === IMAGE ENHANCEMENT ===
lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, 3) * 255.0, 0, 255)
image = cv2.LUT(image, lookUpTable)
_, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# == MATCHING
# print(matching.find_matching_object(transform.crop(image, 0, 0, 30, 30)))
# print(matching.find_matching_object(transform.crop(image, 0, 30, 30, 30)))
# print(matching.find_matching_object(transform.crop(image, 0, 60, 30, 30)))
# print(matching.find_matching_object(transform.crop(image, 0, 90, 30, 30)))
# print(matching.find_matching_object(transform.crop(image, 0, 120, 30, 30)))
# print(matching.find_matching_object(transform.crop(image, 0, 150, 30, 30)))
# print(matching.find_matching_object(transform.crop(image, 0, 180, 30, 30)))
# print(matching.find_matching_object(transform.crop(image, 0, 210, 30, 30)))
# print(matching.find_matching_object(transform.crop(image, 0, 240, 30, 30)))

print(matching.find_matching_object(transform.crop(image, 30 * 2, 30 * 1, 30, 30)))
for i in range(6):
    for j in range(20):
        pass
        # print(transform.crop(image, i * 30, j * 30, 30, 30).shape)
        # print(i, j, matching.find_matching_object(transform.crop(image, i * 30, j * 30, 30, 30)))

# print(matching.find_matching_object(np.ones((30, 30), dtype="uint8") * 255))

cv2.imshow("Image", image)
cv2.waitKey(0)
