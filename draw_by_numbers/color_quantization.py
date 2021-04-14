import cv2
import numpy as np


def quatitize(image, bins: int):
    return image


def kmeans(image, bins: int):
    if image.dtype == np.uint8:
        image = image.astype(np.float32)

    # flatten image
    data = image.reshape((-1, 3))
    assert data.dtype == np.float32

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                20, 1.0)

    ret, label, center = cv2.kmeans(
        data, bins, None,
        criteria=criteria,
        attempts=10,
        flags=cv2.KMEANS_PP_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    return res2, (ret, label, center)
