import cv2
import numpy as np


def four_point_transform(image: np.ndarray, tl, tr, bl, br):
    maxWidth = 26 * 30
    maxHeight = 17 * 30

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(np.array([tl, tr, bl, br], dtype="float32"), dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def crop(image: np.ndarray, x, y, w, h):
    return image[x : x + w, y : y + h]
