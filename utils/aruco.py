import cv2
import numpy as np
from typing import List

from utils.classes import Aruco


def detect_aruco(image: np.ndarray) -> List[Aruco]:
    print("[INFO] detecting 'DICT_4X4_100' tags...")
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
    ids = ids.flatten()

    arucos = []
    for (markerCorner, markerID) in zip(corners, ids):
        markerCorner = markerCorner.reshape((4, 2))
        arucos.append(Aruco(markerCorner, markerID))

    return arucos


def draw_aruco(image: np.ndarray, arucos: List[Aruco]):
    # loop over the detected ArUCo corners
    for aruco in arucos:
        # extract the marker corners (which are always returned in
        # top-left, top-right, bottom-right, and bottom-left order)
        (topLeft, topRight, bottomRight, bottomLeft) = aruco.corners
        # convert each of the (x, y)-coordinate pairs to integers
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))
        # draw the bounding box of the ArUCo detection
        cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
        # compute and draw the center (x, y)-coordinates of the ArUco
        # marker
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
        # draw the ArUco marker ID on the image
        cv2.putText(image, str(aruco.id), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("[INFO] ArUco marker ID: {}".format(aruco.id))

    # show the output image
    return image
