import cv2
import numpy as np
from typing import Tuple, List
ColorTriple = Tuple[int, int, int]


def create_edge_image(image, colors: List[ColorTriple], *, area_threshold: float = 0.0):
    contour_pairs = []

    for color in colors:

        bin_image = np.zeros(
            [image.shape[0], image.shape[1], 1], dtype=np.uint8)
        bin_image[(image == color).all(axis=-1)] = 255

        # TODO open image // add morphological operation to smooth edges
        contours, hierarchy = cv2.findContours(
            bin_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # filter countours that are to small
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= area_threshold:
                m = cv2.moments(contour)
                center_x = int(m["m10"] / (m["m00"] + 1e-12))
                center_y = int(m["m01"] / (m["m00"] + 1e-12))
                contour_pairs.append(
                    (contour, (center_x, center_y), color))

    return contour_pairs


def plot_contour_pairs(contours_pairs, *, image_shape=None, image_height=None, image_width=None):
    if image_shape is not None:
        canvas = np.full([image_shape[0], image_width[1], 3],
                         255, dtype=np.uint8)
    elif image_height is not None and image_width is not None:
        canvas = np.full([image_height, image_width, 3], 255, dtype=np.uint8)
    else:
        raise AttributeError(
            "Provide either image_shape or image_height/image_width argument")
    colors = []
    text_canvas = canvas.copy()
    for contour, center, color in contours_pairs:
        canvas = cv2.drawContours(canvas, [contour], -1, (0, 0, 0), 1)

        if color not in colors:
            colors.append(color)

        # TODO color numbers on center
        font = cv2.FONT_HERSHEY_PLAIN
        font_px_size = 8
        text_canvas = cv2.putText(
            text_canvas, f"{colors.index(color)}", center, font, cv2.getFontScaleFromHeight(font, font_px_size), (0, 0, 0))
    kernel = np.ones([3, 3], dtype=np.uint8)
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_OPEN, kernel=kernel)
    canvas[(text_canvas == (0, 0, 0)).all(axis=-1)] = (255, 0, 0)
    return canvas


if __name__ == "__main__":
    pass
