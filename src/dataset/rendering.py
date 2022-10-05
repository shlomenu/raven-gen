# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image

from panel import Panel
from configuration import CENTER, DEFAULT_WIDTH, IMAGE_SIZE, Shape, Point, AngularPosition, PlanarPosition, StructureType


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
    assert len(panels) == 9
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
    img_grid = np.ones((IMAGE_SIZE * 5 + 20, IMAGE_SIZE * 4), np.uint8) * 255
    img_grid[:IMAGE_SIZE * 3,
             int(0.5 * IMAGE_SIZE):int(3.5 * IMAGE_SIZE)] = matrix_image
    img_grid[-(IMAGE_SIZE * 2):, :] = answer_image
    return img_grid


def render_panel(panel: Panel):
    # Decompose the panel into a structure and its entities
    canvas = np.ones((IMAGE_SIZE, IMAGE_SIZE), np.uint8) * 255
    structure_name, entities = panel.prepare()
    background = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    # note left components entities are in the lower layer
    for entity in entities:
        entity_img = render_entity(entity)
        background[entity_img > 0] = 0
        background += entity_img
    structure_img = render_structure(structure_name)
    background[structure_img > 0] = 0
    background += structure_img
    return canvas - background


def render_structure(structure_name):
    r = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    if structure_name is StructureType.LEFT_RIGHT:
        r[:, int(0.5 * IMAGE_SIZE)] = 255.0
    elif structure_name is StructureType.UP_DOWN:
        r[int(0.5 * IMAGE_SIZE), :] = 255.0
    return r


def render_entity(entity):
    entity_type = entity.type.value()
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    center = Point(y=int(entity.bbox.y_c * IMAGE_SIZE),
                   x=int(entity.bbox.x_c * IMAGE_SIZE))
    unit = min(entity.bbox.max_w, entity.bbox.max_h) * IMAGE_SIZE // 2
    # minus because of the way we show the image, see render_panel's return
    color = 255 - entity.color.value()
    width = DEFAULT_WIDTH
    if entity_type is Shape.TRIANGLE:
        dl = int(unit * entity.size.value())
        pts = np.array(
            [[center.y, center.x - dl],
             [center.y + int(dl / 2.0 * np.sqrt(3)), center.x + int(dl / 2.0)],
             [center.x - int(dl / 2.0 * np.sqrt(3)), center.x + int(dl / 2.0)]
             ], np.int32).reshape((-1, 1, 2))
        if color != 0:  # filled
            cv2.fillConvexPoly(img, pts, color)  # fill the interior
            cv2.polylines(img, [pts], True, 255, width)  # draw the edge
        else:  # not filled
            cv2.polylines(img, [pts], True, 255, width)
    elif entity_type is Shape.SQUARE:
        dl = int(unit / 2 * np.sqrt(2) * entity.size.value())
        pt1 = (center.y - dl, center.x - dl)
        pt2 = (center.y + dl, center.x + dl)
        if color != 0:
            cv2.rectangle(img, pt1, pt2, color, -1)
            cv2.rectangle(img, pt1, pt2, 255, width)
        else:
            cv2.rectangle(img, pt1, pt2, 255, width)
    elif entity_type is Shape.PENTAGON:
        dl = int(unit * entity.size.value())
        pts = np.array([[center.y, center.x - dl],
                        [
                            center.y - int(dl * np.cos(np.pi / 10)),
                            center.x - int(dl * np.sin(np.pi / 10))
                        ],
                        [
                            center.y - int(dl * np.sin(np.pi / 5)),
                            center.x + int(dl * np.cos(np.pi / 5))
                        ],
                        [
                            center.y + int(dl * np.sin(np.pi / 5)),
                            center.x + int(dl * np.cos(np.pi / 5))
                        ],
                        [
                            center.y + int(dl * np.cos(np.pi / 10)),
                            center.x - int(dl * np.sin(np.pi / 10))
                        ]], np.int32).reshape((-1, 1, 2))
        if color != 0:
            cv2.fillConvexPoly(img, pts, color)
            cv2.polylines(img, [pts], True, 255, width)
        else:
            cv2.polylines(img, [pts], True, 255, width)
    elif entity_type is Shape.HEXAGON:
        dl = int(unit * entity.size.value())
        pts = np.array(
            [[center.y, center.x - dl],
             [center.y - int(dl / 2.0 * np.sqrt(3)), center.x - int(dl / 2.0)],
             [center.y - int(dl / 2.0 * np.sqrt(3)), center.x + int(dl / 2.0)],
             [center.y, center.x + dl],
             [center.y + int(dl / 2.0 * np.sqrt(3)), center.x + int(dl / 2.0)],
             [center.y + int(dl / 2.0 * np.sqrt(3)), center.x - int(dl / 2.0)]
             ], np.int32).reshape((-1, 1, 2))
        if color != 0:
            cv2.fillConvexPoly(img, pts, color)
            cv2.polylines(img, [pts], True, 255, width)
        else:
            cv2.polylines(img, [pts], True, 255, width)
    elif entity_type is Shape.CIRCLE:
        radius = int(unit * entity.size.value())
        if color != 0:
            cv2.circle(img, tuple(center), radius, color, -1)
            cv2.circle(img, tuple(center), radius, 255, width)
        else:
            cv2.circle(img, tuple(center), radius, 255, width)
    elif entity_type is Shape.NONE:
        pass
    if isinstance(entity.bbox, AngularPosition):
        img = rotate(img,
                     entity.bbox.omega,
                     center=Point(x=(entity.bbox.x_r * IMAGE_SIZE),
                                  y=(entity.bbox.y_r * IMAGE_SIZE)))
    elif isinstance(entity.bbox, PlanarPosition):
        img = rotate(img, entity.angle.value(), center=center)
    else:
        raise ValueError("unknown position type: not angular or planar")
    return img


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
