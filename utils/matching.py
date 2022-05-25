import cv2
# import mahotas
import numpy as np
from scipy.spatial import distance


def load_reference_image(name: str):
    filename = f"./ref/{name}.png"

    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Resize
    image = cv2.resize(image, (30, 30), interpolation=cv2.INTER_LINEAR)

    # Threshold
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    return image


def similarity(cell: np.ndarray, ref: np.ndarray):
    # cell_desc = cv2.HuMoments(cv2.moments(cell)).flatten()
    # ref_desc = cv2.HuMoments(cv2.moments(ref)).flatten()
    # cell_desc = mahotas.features.zernike_moments(cell, 15)
    # ref_desc = mahotas.features.zernike_moments(cell, 15)

    # print(distance.euclidean(cell_desc, ref_desc))
    # print('{:f}'.format(distance.euclidean(cell_desc, ref_desc)))
    # print('{:f}'.format(distance.euclidean(cell.flatten(), ref.flatten())))
    # return distance.euclidean(cell.flatten(), ref.flatten())

    # Naive
    res = np.sum(np.bitwise_not(np.bitwise_or(cell.astype(np.uint8), ref.astype(np.uint8)))) // 255
    res /= (30 * 30) - (np.sum(ref) / 255)
    return res

    # Ccorr normed
    res = cv2.matchTemplate(cell, ref, cv2.TM_CCOEFF_NORMED)

    return res[0][0]


def find_matching_object(cell: np.ndarray):
    objects = ["circle", "cross", "dash", "slash", "strip", "triangle"]
    normalization = [5563.443178, 5925.664520, 4297.336384, 4756.963317, 4297.336384, 5324.556320]

    # match = None
    # max_similarity = 700
    # for obj, norm in zip(objects, normalization):
    #     sim = similarity(cell, load_reference_image(obj))
    #     sim = abs(norm - sim)
    #     # print(sim)
    #     # cv2.imshow(f"{obj}", cell - load_reference_image(obj))

    #     if sim > max_similarity:
    #         max_similarity = sim
    #         match = obj

    # Naive
    match = None
    max_similarity = 0.25
    for obj in objects:
        sim = similarity(cell, load_reference_image(obj))
        # print(sim)
        # cv2.imshow(f"{obj}", cell)
        cv2.imshow(f"{obj}", cv2.resize(np.logical_not(np.logical_or(cell.astype(np.uint8), load_reference_image(obj).astype(np.uint8))).astype(np.uint8) * 255, (240, 240), interpolation=cv2.INTER_NEAREST))
        # cv2.imshow(f"{obj}", cv2.resize((cell // 3 * 2 + load_reference_image(obj) // 3), (240, 240), interpolation=cv2.INTER_NEAREST))

        if sim > max_similarity:
            max_similarity = sim
            match = obj

    # CCorr normed
    # normalization = [424, 360, 616, 552, 616, 464]
    # match = None
    # max_similarity = 0.25
    # for obj, norm in zip(objects, normalization):
    #     sim = similarity(cell, load_reference_image(obj))
    #     # print(sim)
    #     # cv2.imshow(f"{obj}", cell)
    #     # print((cell // 2 + load_reference_image(obj) // 2))
    #     cv2.imshow(f"{obj}", cv2.resize((cell // 3 * 2 + load_reference_image(obj) // 3), (240, 240), interpolation=cv2.INTER_NEAREST))

    #     if sim > max_similarity:
    #         max_similarity = sim
    #         match = obj

    return max_similarity, match
