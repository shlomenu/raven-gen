from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np
from PIL import Image


@dataclass
class Point:
    x: Union[int, float]
    y: Union[int, float]

    def __iter__(self):
        for attr in ["y", "x"]:
            yield getattr(self, attr)


IMAGE_SIZE = 160
CENTER = Point(x=IMAGE_SIZE // 2, y=IMAGE_SIZE // 2)
DEFAULT_RADIUS = IMAGE_SIZE // 4
DEFAULT_WIDTH = 2


def imshow(array):
    Image.fromarray(array).show()


def imsave(array, filepath):
    Image.fromarray(array).save(filepath)


def generate_matrix(panels):
    """
    Merge nine panels into 3x3 grid.
    :param panels: list of nine ndarrays in left-to-right, top-to-bottom order
    :returns: merged ndarray
    """
    img_grid = np.zeros((IMAGE_SIZE * 3, IMAGE_SIZE * 3), np.uint8)
    for pos, panel in enumerate(panels):
        i, j = divmod(pos, 3)
        img_grid[i * IMAGE_SIZE:(i + 1) * IMAGE_SIZE,
                 j * IMAGE_SIZE:(j + 1) * IMAGE_SIZE] = panel
    for x in [0.33, 0.67]:
        band_center = int(x * IMAGE_SIZE * 3)
        img_grid[band_center - 1:band_center + 1, :] = 0
    for y in [0.33, 0.67]:
        band_center = int(y * IMAGE_SIZE * 3)
        img_grid[:, band_center - 1:band_center + 1] = 0
    return img_grid


def generate_answers(panels):
    """
    Merge eight panels into 2x4 grid.
    :param panels: list of eight ndarrays in left-to-right, top-to-bottom order
    :returns: merged ndarray
    """
    assert len(panels) == 8
    img_grid = np.zeros((IMAGE_SIZE * 2, IMAGE_SIZE * 4), np.uint8)
    for pos, panel in enumerate(panels):
        i, j = divmod(pos, 4)
        img_grid[i * IMAGE_SIZE:(i + 1) * IMAGE_SIZE,
                 j * IMAGE_SIZE:(j + 1) * IMAGE_SIZE] = panel
    for x in [0.5]:
        band_center = int(x * IMAGE_SIZE * 2)
        img_grid[band_center - 1:band_center + 1, :] = 0
    for y in [0.25, 0.5, 0.75]:
        band_center = int(y * IMAGE_SIZE * 4)
        img_grid[:, band_center - 1:band_center + 1] = 0
    return img_grid


def generate_matrix_answer(panels):
    """
    Merge question and completed question panels into single 6x3 layout with 
    the unanswered question matrix above the answered question matrix.
    :param panels: list of eighteen ndarrays in left-to-right, top-to-bottom, incomplete-to-complete matrix order
    :returns: merged ndarray
    """
    assert len(panels) == 18
    img_grid = np.zeros((IMAGE_SIZE * 6, IMAGE_SIZE * 3), np.uint8)
    for pos, panel in enumerate(panels):
        i, j = divmod(pos, 3)
        img_grid[i * IMAGE_SIZE:(i + 1) * IMAGE_SIZE,
                 j * IMAGE_SIZE:(j + 1) * IMAGE_SIZE] = panel
    for x in [0.33, 0.67, 1.00, 1.33, 1.67]:
        band_center = int(x * IMAGE_SIZE * 3)
        img_grid[band_center - 1:band_center + 1, :] = 0
    for y in [0.33, 0.67]:
        band_center = int(y * IMAGE_SIZE * 3)
        img_grid[:, band_center - 1:band_center + 1] = 0
    return img_grid


def merge_matrix_answer(matrix_panels, answer_panels):
    """
    Merge question and answer panels into single 5x4 layout with the 3x3 
    question matrix centered above the 2x4 answer matrix.
    :param matrix_panels: list of nine ndarrays in left-to-right, top-to-bottom order
    :param answer_panels: list of eight ndarrays in left-to-right, top-to-bottom order
    :returns: merged ndarray
    """
    matrix_image = generate_matrix(matrix_panels)
    answer_image = generate_answers(answer_panels)
    img_grid = np.ones(
        (IMAGE_SIZE * 5 + 20, IMAGE_SIZE * 4), np.uint8) * 255
    img_grid[:IMAGE_SIZE * 3,
             int(0.5 * IMAGE_SIZE):int(3.5 * IMAGE_SIZE)] = matrix_image
    img_grid[-(IMAGE_SIZE * 2):, :] = answer_image
    return img_grid


def shift(img, dx, dy):
    return cv2.warpAffine(img,
                          np.array([[1, 0, dx], [0, 1, dy]], np.float32),
                          (IMAGE_SIZE, IMAGE_SIZE),
                          flags=cv2.INTER_LINEAR)


def rotate(img, angle, center=CENTER):
    return cv2.warpAffine(img,
                          cv2.getRotationMatrix2D(tuple(center), angle, 1),
                          (IMAGE_SIZE, IMAGE_SIZE),
                          flags=cv2.INTER_LINEAR)


def scale(img, tx, ty, center=CENTER):
    return cv2.warpAffine(img,
                          np.array([[tx, 0, center.y *
                                     (1 - tx)], [0, ty, center.x * (1 - ty)]],
                                   np.float32), (IMAGE_SIZE, IMAGE_SIZE),
                          flags=cv2.INTER_LINEAR)
