from typing import Union
from dataclasses import dataclass, field

import cv2
import numpy as np

from .attribute import (Shape, Size, Color, Angle, Shapes, PlanarPosition,
                        AngularPosition)


@dataclass
class Point:
    x: Union[int, float]
    y: Union[int, float]

    def __iter__(self):
        for attr in ["y", "x"]:
            yield getattr(self, attr)


def rotate(img, angle, background_color, panel_size, center=None):
    if center is None:
        center = Point(x=(panel_size // 2), y=(panel_size // 2))
    return cv2.warpAffine(img,
                          cv2.getRotationMatrix2D(tuple(center), angle, 1),
                          (panel_size, panel_size),
                          flags=cv2.INTER_LINEAR,
                          borderValue=background_color)


@dataclass(init=False)
class Entity:
    name: str
    bbox: Union[PlanarPosition, AngularPosition] = field(repr=False)
    shape: Shape
    size: Size
    color: Color
    angle: Angle

    def __init__(self, name, bbox, constraints):
        self.name = name
        self.bbox = bbox
        self.shape = Shape(constraints)
        self.size = Size(constraints)
        self.color = Color(constraints)
        self.angle = Angle(constraints)

    def sample(self, constraints):
        self.shape.sample(constraints)
        self.size.sample(constraints)
        self.color.sample(constraints)
        self.angle.sample(constraints)

    def render(self, background_color, panel_size, shape_border_thickness):
        img = np.ones((panel_size, panel_size), np.uint8) * background_color
        center = Point(y=int(self.bbox.y_c * panel_size),
                       x=int(self.bbox.x_c * panel_size))
        unit = min(self.bbox.max_w, self.bbox.max_h) * panel_size // 2
        # minus because of the way we show the image, see render_panel's return
        if self.shape.value is Shapes.TRIANGLE:
            dl = int(unit * self.size.value)
            pts = np.array([[
                center.y, center.x - dl
            ], [
                center.y + int(dl / 2.0 * np.sqrt(3)), center.x + int(dl / 2.0)
            ], [
                center.y - int(dl / 2.0 * np.sqrt(3)), center.x + int(dl / 2.0)
            ]], np.int32).reshape((-1, 1, 2))
            if self.color.value != background_color:
                cv2.fillConvexPoly(img, pts, self.color.value)
            cv2.polylines(img, [pts], True, 0, shape_border_thickness)
        elif self.shape.value is Shapes.SQUARE:
            dl = int(unit / 2 * np.sqrt(2) * self.size.value)
            pt1 = (center.y - dl, center.x - dl)
            pt2 = (center.y + dl, center.x + dl)
            if self.color.value != background_color:
                cv2.rectangle(img, pt1, pt2, self.color.value, -1)
            cv2.rectangle(img, pt1, pt2, 0, shape_border_thickness)
        elif self.shape.value is Shapes.PENTAGON:
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
            if self.color.value != background_color:
                cv2.fillConvexPoly(img, pts, self.color.value)
            cv2.polylines(img, [pts], True, 0, shape_border_thickness)
        elif self.shape.value is Shapes.HEXAGON:
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
            if self.color.value != background_color:
                cv2.fillConvexPoly(img, pts, self.color.value)
            cv2.polylines(img, [pts], True, 0, shape_border_thickness)
        elif self.shape.value is Shapes.CIRCLE:
            radius = int(unit * self.size.value)
            if self.color.value != background_color:
                cv2.circle(img, tuple(center), radius, self.color.value, -1)
            cv2.circle(img, tuple(center), radius, 0, shape_border_thickness)
        elif self.shape.value is Shapes.NONE:
            pass
        if isinstance(self.bbox, AngularPosition):
            return rotate(img,
                          self.bbox.omega,
                          background_color,
                          panel_size,
                          center=Point(x=(self.bbox.x_r * panel_size),
                                       y=(self.bbox.y_r * panel_size)))
        elif isinstance(self.bbox, PlanarPosition):
            return rotate(img,
                          self.angle.value,
                          background_color,
                          panel_size,
                          center=center)
        else:
            raise ValueError("unknown position type: not angular or planar")
