from typing import Union
from dataclasses import dataclass, field

import cv2
import numpy as np

from .attribute import (Type, Size, Color, Angle, Shape, PlanarPosition,
                       AngularPosition)


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


def rotate(img, angle, center=CENTER):
    return cv2.warpAffine(img,
                          cv2.getRotationMatrix2D(tuple(center), angle, 1),
                          (IMAGE_SIZE, IMAGE_SIZE),
                          flags=cv2.INTER_LINEAR)


@dataclass(init=False)
class Entity:
    name: str
    bbox: Union[PlanarPosition, AngularPosition] = field(repr=False)
    type: Type
    size: Size
    color: Color
    angle: Angle

    def __init__(self, name, bbox, constraints):
        self.name = name
        self.bbox = bbox
        self.type = Type(constraints)
        self.size = Size(constraints)
        self.color = Color(constraints)
        self.angle = Angle(constraints)

    def sample(self, constraints):
        self.type.sample(constraints)
        self.size.sample(constraints)
        self.color.sample(constraints)
        self.angle.sample(constraints)

    def render(self):
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
        center = Point(y=int(self.bbox.y_c * IMAGE_SIZE),
                       x=int(self.bbox.x_c * IMAGE_SIZE))
        unit = min(self.bbox.max_w, self.bbox.max_h) * IMAGE_SIZE // 2
        # minus because of the way we show the image, see render_panel's return
        color = 255 - self.color.value
        width = DEFAULT_WIDTH
        if self.type.value is Shape.TRIANGLE:
            dl = int(unit * self.size.value)
            pts = np.array([[
                center.y, center.x - dl
            ], [
                center.y + int(dl / 2.0 * np.sqrt(3)), center.x + int(dl / 2.0)
            ], [
                center.y - int(dl / 2.0 * np.sqrt(3)), center.x + int(dl / 2.0)
            ]], np.int32).reshape((-1, 1, 2))
            if color != 0:  # filled
                cv2.fillConvexPoly(img, pts, color)  # fill the interior
                cv2.polylines(img, [pts], True, 255, width)  # draw the edge
            else:  # not filled
                cv2.polylines(img, [pts], True, 255, width)
        elif self.type.value is Shape.SQUARE:
            dl = int(unit / 2 * np.sqrt(2) * self.size.value)
            pt1 = (center.y - dl, center.x - dl)
            pt2 = (center.y + dl, center.x + dl)
            if color != 0:
                cv2.rectangle(img, pt1, pt2, color, -1)
                cv2.rectangle(img, pt1, pt2, 255, width)
            else:
                cv2.rectangle(img, pt1, pt2, 255, width)
        elif self.type.value is Shape.PENTAGON:
            dl = int(unit * self.size.value)
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
        elif self.type.value is Shape.HEXAGON:
            dl = int(unit * self.size.value)
            pts = np.array([[
                center.y, center.x - dl
            ], [
                center.y - int(dl / 2.0 * np.sqrt(3)), center.x - int(dl / 2.0)
            ], [
                center.y - int(dl / 2.0 * np.sqrt(3)), center.x + int(dl / 2.0)
            ], [
                center.y, center.x + dl
            ], [
                center.y + int(dl / 2.0 * np.sqrt(3)), center.x + int(dl / 2.0)
            ], [
                center.y + int(dl / 2.0 * np.sqrt(3)), center.x - int(dl / 2.0)
            ]], np.int32).reshape((-1, 1, 2))
            if color != 0:
                cv2.fillConvexPoly(img, pts, color)
                cv2.polylines(img, [pts], True, 255, width)
            else:
                cv2.polylines(img, [pts], True, 255, width)
        elif self.type.value is Shape.CIRCLE:
            radius = int(unit * self.size.value)
            if color != 0:
                cv2.circle(img, tuple(center), radius, color, -1)
                cv2.circle(img, tuple(center), radius, 255, width)
            else:
                cv2.circle(img, tuple(center), radius, 255, width)
        elif self.type.value is Shape.NONE:
            pass
        if isinstance(self.bbox, AngularPosition):
            img = rotate(img,
                         self.bbox.omega,
                         center=Point(x=(self.bbox.x_r * IMAGE_SIZE),
                                      y=(self.bbox.y_r * IMAGE_SIZE)))
        elif isinstance(self.bbox, PlanarPosition):
            img = rotate(img, self.angle.value, center=center)
        else:
            raise ValueError("unknown position type: not angular or planar")
        return img
