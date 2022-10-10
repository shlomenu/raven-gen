from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, NamedTuple
from collections import namedtuple
from enum import Enum, auto

import numpy as np
from scipy.special import comb
import cv2

from attribute import (AttributeType, Uniformity, Type,
                       Size, Color, Angle, Configuration, Shape,
                       NUM_MIN, NUM_MAX, TYPE_MIN, TYPE_MAX,
                       SIZE_MIN, SIZE_MAX, COLOR_MIN, COLOR_MAX,
                       ANGLE_MIN, ANGLE_MAX, UNI_MIN, UNI_MAX)
from rendering import rotate, IMAGE_SIZE, Point, DEFAULT_WIDTH
from rule import Rules, RuleType


class PositionType(Enum):
    PLANAR = auto()
    ANGULAR = auto()


@dataclass
class PlanarPosition:
    x_c: Union[int, float]
    y_c: Union[int, float]
    max_w: Union[int, float]
    max_h: Union[int, float]


@dataclass
class AngularPosition:
    x_c: Union[int, float]
    y_c: Union[int, float]
    max_w: Union[int, float]
    max_h: Union[int, float]
    x_r: Union[int, float]
    y_r: Union[int, float]
    omega: Union[int, float]


@dataclass
class Bounds:
    min: int
    max: int


@dataclass
class Constraints:
    number: Bounds
    type: Bounds
    size: Bounds
    color: Bounds
    angle: Bounds
    uniformity: Bounds
    position_type: PositionType
    positions: Union[List[AngularPosition], List[PlanarPosition]]


class Entity:

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

    def render_entity(self):
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
        center = Point(y=int(self.bbox.y_c * IMAGE_SIZE),
                       x=int(self.bbox.x_c * IMAGE_SIZE))
        unit = min(self.bbox.max_w, self.bbox.max_h) * IMAGE_SIZE // 2
        # minus because of the way we show the image, see render_panel's return
        color = 255 - self.color.value
        width = DEFAULT_WIDTH
        if self.type.value is Shape.TRIANGLE:
            dl = int(unit * self.size.value)
            pts = np.array(
                [[center.y, center.x - dl],
                 [center.y + int(dl / 2.0 * np.sqrt(3)),
                  center.x + int(dl / 2.0)],
                 [center.x - int(dl / 2.0 * np.sqrt(3)),
                  center.x + int(dl / 2.0)]
                 ], np.int32).reshape((-1, 1, 2))
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
            pts = np.array(
                [[center.y, center.x - dl],
                 [center.y - int(dl / 2.0 * np.sqrt(3)),
                  center.x - int(dl / 2.0)],
                 [center.y - int(dl / 2.0 * np.sqrt(3)),
                  center.x + int(dl / 2.0)],
                 [center.y, center.x + dl],
                 [center.y + int(dl / 2.0 * np.sqrt(3)),
                  center.x + int(dl / 2.0)],
                 [center.y + int(dl / 2.0 * np.sqrt(3)),
                  center.x - int(dl / 2.0)]
                 ], np.int32).reshape((-1, 1, 2))
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


@dataclass
class AttributeHistory:

    def __init__(self, constraints):
        PositionHistory = namedtuple(
            "PositionHistory", ["available", "sampled"])
        self.number: List[int] = []
        self.position: Dict[int, NamedTuple[int, List[np.ndarray]]] = {}
        self.type: List[int] = []
        self.size: List[int] = []
        self.color: List[int] = []
        self.angle: List[int] = []
        for i in range(constraints.number.min + 1, constraints.number.max + 2):
            self.position[i] = PositionHistory(available=comb(
                constraints.positions.shape[0], i), sampled=[])


@dataclass
class Component:

    def __init__(self, component_type: ComponentType, layout_type: LayoutType, constraints: Constraints):
        self.component_type = component_type
        self.layout_type = layout_type
        self.constraints = constraints
        self.config = Configuration(self.constraints)
        self.uniformity = Uniformity(self.constraints)
        self.history = AttributeHistory(self.constraints)
        self.entities = []
        self.initial_constraints = copy.deepcopy(self.constraints)

    def setting_of(self, attr):
        return self.attr(attr).setting

    def attr(self, attr):
        return getattr(self.entity, attr.name.lower())

    @property
    def entity(self):
        return self.entities[0]

    def make_uniform(self, attr):
        setting = self.setting_of(attr)
        for entity in self.entities[1:]:
            entity_attr = getattr(entity, attr.name.lower())
            entity_attr.setting = setting

    def set_uniform(self, attr, setting):
        for entity in self.entities:
            entity_attr = getattr(entity, attr.name.lower())
            entity_attr.setting = setting

    def set_position(self):
        for entity, bbox in zip(self.entities, self.config.position.value):
            entity.bbox = bbox

    def sample(self, sample_position=False, sample_number=False, carryover=True, uniform=None):
        if sample_position or sample_number:
            if sample_number:
                self.config.number.sample(self.constraints)
            self.config.position.sample(self.config.number.value)
        if uniform is None:
            uniform = self.uniformity.value
        if uniform:
            if len(self.entities) > 0 and carryover:
                self.entities = [self.entities[0]]
            else:
                self.entities = [Entity(
                    name=str(0), bbox=self.config.position.value[0], constraints=self.constraints)]
            for i, bbox in self.config.position.value[1:]:
                entity = copy.deepcopy(self.entities[0])
                entity.name = str(i)
                entity.bbox = bbox
                self.entities.append(entity)
        else:
            self.entities = [
                Entity(name=str(i), bbox=bbox, constraints=self.constraints)
                for i, bbox in enumerate(self.config.position.value)]

    def sample_unique(self, attr, ground_truth):
        if attr is AttributeType.NUMBER:
            self.config.sample_unique(
                self.initial_constraints, ground_truth.history,
                record=True, overwrite=True)
            self.sample(sample_position=True, carryover=False)
        elif attr is AttributeType.POSITION:
            self.config.position.sample_unique(
                self.config.number.value, ground_truth.history,
                record=True, overwrite=True)
            self.set_position()
        elif attr is AttributeType.ANGLE or \
                attr is AttributeType.UNIFORMITY:
            raise ValueError(
                f"unsupported operation on attribute of type: {attr!s}"
            )
        elif attr in AttributeType:
            def sample_attr_unique(e, attr):
                attr = getattr(e, attr.name.lower())
                attr.sample_unique(
                    self.initial_constraints, ground_truth.history,
                    record=True, overwrite=True)
            if self.uniformity.value:
                sample_attr_unique(self.entities[0], attr)
                self.make_uniform(attr)
            else:
                for entity in self.entities:
                    sample_attr_unique(entity, attr)
        else:
            raise ValueError("unsupported operation")

    def reset_history(self):
        self.history = AttributeHistory(self.initial_constraints)


class StructureType(Enum):
    SINGLETON = auto()
    LEFT_RIGHT = auto()
    UP_DOWN = auto()
    OUT_IN = auto()


class ComponentType(Enum):
    GRID = auto()
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()
    OUT = auto()
    IN = auto()


class LayoutType(Enum):
    CENTER_SINGLE = auto()
    DISTRIBUTE_FOUR = auto()
    DISTRIBUTE_NINE = auto()
    LEFT_CENTER_SINGLE = auto()
    RIGHT_CENTER_SINGLE = auto()
    UP_CENTER_SINGLE = auto()
    DOWN_CENTER_SINGLE = auto()
    OUT_CENTER_SINGLE = auto()
    IN_CENTER_SINGLE = auto()
    IN_DISTRIBUTE_FOUR = auto()


class Panel:

    def __init__(self,
                 structure_type,
                 component_1_type,
                 layout_1_type,
                 position_type_1,
                 positions_1,
                 number_min_1=NUM_MIN,
                 number_max_1=NUM_MAX,
                 type_min_1=TYPE_MIN,
                 type_max_1=TYPE_MAX,
                 size_min_1=SIZE_MIN,
                 size_max_1=SIZE_MAX,
                 color_min_1=COLOR_MIN,
                 color_max_1=COLOR_MAX,
                 angle_min_1=ANGLE_MIN,
                 angle_max_1=ANGLE_MAX,
                 uniformity_min_1=UNI_MIN,
                 uniformity_max_1=UNI_MAX,
                 component_2_type=None,
                 layout_2_type=None,
                 position_type_2=None,
                 positions_2=None,
                 number_min_2=NUM_MIN,
                 number_max_2=NUM_MAX,
                 type_min_2=TYPE_MIN,
                 type_max_2=TYPE_MAX,
                 size_min_2=SIZE_MIN,
                 size_max_2=SIZE_MAX,
                 color_min_2=COLOR_MIN,
                 color_max_2=COLOR_MAX,
                 angle_min_2=ANGLE_MIN,
                 angle_max_2=ANGLE_MAX,
                 uniformity_min_2=UNI_MIN,
                 uniformity_max_2=UNI_MAX):
        self.structure_type = structure_type
        self.component_1_type = component_1_type
        self.component_1 = Component(
            component_type=component_1_type,
            layout_type=layout_1_type,
            constraints=Constraints(
                number=Bounds(min=number_min_1, max=number_max_1),
                type=Bounds(min=type_min_1, max=type_max_1),
                size=Bounds(min=size_min_1, max=size_max_1),
                color=Bounds(min=color_min_1, max=color_max_1),
                angle=Bounds(min=angle_min_1, max=angle_max_1),
                uniformity=Bounds(min=uniformity_min_1, max=uniformity_max_1),
                position_type=position_type_1,
                positions=positions_1))
        if component_2_type and layout_2_type:
            self.component_2 = Component(
                component_type=component_2_type,
                layout_type=layout_2_type,
                constraints=Constraints(
                    number=Bounds(min=number_min_2, max=number_max_2),
                    type=Bounds(min=type_min_2, max=type_max_2),
                    size=Bounds(min=size_min_2, max=size_max_2),
                    color=Bounds(min=color_min_2, max=color_max_2),
                    angle=Bounds(min=angle_min_2, max=angle_max_2),
                    uniformity=Bounds(min=uniformity_min_2,
                                      max=uniformity_max_2),
                    position_type=position_type_2,
                    positions=positions_2))
            self.components = (self.component_1, self.component_2)
        else:
            self.components = (self.component_1,)

    def prune(self, rules: Rules) -> Optional[Panel]:
        """
        Modify the bounds of attributes based on the rules that will be applied
        to them to ensure those rules can be properly expressed with the given
        range of values; if this cannot be done by tightening bounds, returns None.
        """
        pruned = copy.deepcopy(self)
        for component, component_rules in zip(pruned.components, rules):
            del component.entities[:]
            for rule in component_rules.all:
                if rule.attr in AttributeType and \
                        rule.attr is not AttributeType.ANGLE and \
                        rule.attr is not AttributeType.UNIFORMITY:
                    if rule.attr is AttributeType.NUMBER or rule.attr is AttributeType.POSITION:
                        bounds = getattr(component, rule.attr.name.lower())
                    else:
                        bounds = getattr(component.constraints,
                                         rule.attr.name.lower())
                    if rule.name is RuleType.PROGRESSION:
                        if rule.attr is AttributeType.POSITION:
                            # bounds.max is setting indicating number of positions
                            # in this layout;
                            #
                            #   bounds.max >= bounds.min + 2 * abs(rule.value)
                            #
                            # ensures that entities can be shifted `rule.value`
                            # positions over before landing in a position that
                            # was occupied within the row.
                            bounds.max = bounds.max - 2 * abs(rule.value)
                        elif bounds:
                            # bounds.max >= bounds.min + 2 * rule.value
                            if rule.value > 0:
                                bounds.max = bounds.max - 2 * rule.value
                            else:
                                bounds.min = bounds.min - 2 * rule.value
                    elif rule.name is RuleType.ARITHMETIC:
                        if rule.attr is AttributeType.POSITION:
                            # Forbid filling all positions so that set union
                            #  appears different from constancy; disallow the
                            #  sampling of disjoint subsets so that set difference
                            #  is visible.
                            if rule.value <= 0:
                                bounds.min = bounds.max // 2
                            bounds.max = bounds.max - 1
                        elif rule.attr is AttributeType.NUMBER or rule.attr is AttributeType.SIZE:
                            # bounds.max >= 2 * bounds.min + 1
                            if rule.value > 0:
                                bounds.max = bounds.max - bounds.min - 1
                            else:
                                bounds.min = 2 * bounds.min + 1
                        elif rule.attr is AttributeType.COLOR:
                            # bounds.max >= bounds.min + 1
                            #   && bounds.max >= 2 * bounds.min
                            if bounds.max - bounds.min < 1:
                                return None
                            elif rule.value > 0:
                                bounds.max = bounds.max - bounds.min
                            elif rule.value < 0:
                                bounds.min = 2 * bounds.min
                    elif rule.name is RuleType.DISTRIBUTE_THREE:
                        if rule.attr is AttributeType.POSITION:
                            # There are n choose k sets of positions in which to
                            # place k entities within a panel containing at most
                            # n entities (n is initially given by `bounds.max + 1`).
                            # Since the lowest setting of number corresponds to one
                            # entity, k >= 1.  If we restrict n >= 3 and k < n, then
                            # n choose k >= 3, as is required to apply the rule.
                            if bounds.max + 1 < 3:
                                return None
                            else:
                                bounds.max = bounds.max - 1
                        elif bounds:
                            # require three distinct settings
                            if bounds.max - bounds.min + 1 < 3:
                                return None
                    if bounds.max < bounds.min:
                        return None
                    component.config.reset()
                else:
                    return None
        return pruned

    @ classmethod
    def make_center_single(cls):
        return cls(
            structure_type=StructureType.SINGLETON,
            component_1_type=ComponentType.GRID,
            layout_1_type=LayoutType.CENTER_SINGLE,
            position_type_1=PositionType.PLANAR,
            positions_1=[PlanarPosition(x_c=0.5, y_c=0.5, max_w=1, max_h=1)],
            number_min_1=0,
            number_max_1=0,
            type_min_1=1)

    @ classmethod
    def make_distribute_four(cls):
        return cls(
            structure_type=StructureType.SINGLETON,
            component_1_type=ComponentType.GRID,
            layout_1_type=LayoutType.DISTRIBUTE_FOUR,
            position_type_1=PositionType.PLANAR,
            positions=[
                PlanarPosition(x_c=0.25, y_c=0.25, max_w=0.5, max_y=0.5),
                PlanarPosition(x_c=0.25, y_c=0.75, max_w=0.5, max_y=0.5),
                PlanarPosition(x_c=0.75, y_c=0.25, max_w=0.5, max_y=0.5),
                PlanarPosition(x_c=0.75, y_c=0.75, max_w=0.5, max_y=0.5)
            ],
            number_min_1=0,
            number_max_1=3,
            type_min_1=1)

    @ classmethod
    def make_distribute_nine(cls):
        return cls(
            structure_type=StructureType.SINGLETON,
            component_1_type=ComponentType.GRID,
            layout_1_type=LayoutType.DISTRIBUTE_NINE,
            position_type_1=PositionType.PLANAR,
            positions_1=[
                PlanarPosition(x_c=0.16, y_c=0.16, max_w=0.33, max_y=0.33),
                PlanarPosition(x_c=0.16, y_c=0.5, max_w=0.33, max_y=0.33),
                PlanarPosition(x_c=0.16, y_c=0.83, max_w=0.33, max_y=0.33),
                PlanarPosition(x_c=0.5, y_c=0.16, max_w=0.33, max_y=0.33),
                PlanarPosition(x_c=0.5, y_c=0.5, max_w=0.33, max_y=0.33),
                PlanarPosition(x_c=0.5, y_c=0.83, max_w=0.33, max_y=0.33),
                PlanarPosition(x_c=0.83, y_c=0.16, max_w=0.33, max_y=0.33),
                PlanarPosition(x_c=0.83, y_c=0.5, max_w=0.33, max_y=0.33),
                PlanarPosition(x_c=0.83, y_c=0.83, max_w=0.33, max_y=0.33)
            ],
            number_min_1=0,
            number_max_1=8,
            type_min_1=1)

    @ classmethod
    def make_left_center_single_right_center_single(cls):
        return cls(
            structure_type=StructureType.LEFT_RIGHT,
            component_1_type=ComponentType.LEFT,
            layout_1_type=LayoutType.LEFT_CENTER_SINGLE,
            position_type_1=PositionType.PLANAR,
            positions_1=[
                PlanarPosition(x_c=0.5, y_c=0.25, max_w=0.5, max_h=0.5)
            ],
            number_min_1=0,
            number_max_1=0,
            type_min_1=1,
            component_2_type=ComponentType.RIGHT,
            layout_2_type=LayoutType.RIGHT_CENTER_SINGLE,
            position_type_2=PositionType.PLANAR,
            positions_2=[
                PlanarPosition(x_c=0.5,
                               y_c=0.75,
                               max_w=0.5,
                               max_h=0.5)
            ],
            number_min_2=0,
            number_max_2=0,
            type_min_2=1)

    @ classmethod
    def make_up_center_single_down_center_single(cls):
        cls(
            structure_type=StructureType.UP_DOWN,
            component_1_type=ComponentType.UP,
            layout_1_type=LayoutType.UP_CENTER_SINGLE,
            position_type_1=PositionType.PLANAR,
            positions_1=[
                PlanarPosition(x_c=0.25, y_c=0.5, max_w=0.5, max_h=0.5)
            ],
            number_min_1=0,
            number_max_1=0,
            type_min_1=1,
            component_2_type=ComponentType.DOWN,
            layout_2_type=LayoutType.DOWN_CENTER_SINGLE,
            position_type_2=PositionType.PLANAR,
            positions=[
                PlanarPosition(x_c=0.75, y_c=0.5, max_w=0.5, max_h=0.5)
            ],
            number_min_2=0,
            number_min_2=0,
            type_min_2=1)

    @ classmethod
    def make_in_center_single_out_center_single(cls):
        cls(
            structure_type=StructureType.OUT_IN,
            component_1_type=ComponentType.OUT,
            layout_1_type=LayoutType.OUT_CENTER_SINGLE,
            position_type_1=PositionType.PLANAR,
            positions_1=[PlanarPosition(x_c=0.5, y_c=0.5, max_w=1, max_h=1)],
            number_min_1=0,
            number_max_1=0,
            type_min_1=1,
            size_min_1=3,
            color_max_1=0,
            component_2_type=ComponentType.IN,
            layout_2_type=LayoutType.IN_CENTER_SINGLE,
            position_type_2=PositionType.PLANAR,
            positions_2=[
                PlanarPosition(x_c=0.5, y_c=0.5, max_w=0.33, max_h=0.33)
            ],
            number_min_2=0,
            number_max_2=0,
            type_min_2=1
        )

    @ classmethod
    def make_in_distribute_four_out_center_single(cls):
        return cls(
            structure_type=StructureType.OUT_IN,
            component_1_type=ComponentType.OUT,
            layout_1_type=LayoutType.OUT_CENTER_SINGLE,
            position_type_1=PositionType.PLANAR,
            positions_1=[PlanarPosition(x_c=0.5, y_c=0.5, max_w=1, max_h=1)],
            number_min_1=0,
            number_max_1=0,
            type_min_1=1,
            size_min_1=3,
            color_max_1=0,
            component_2_type=ComponentType.IN,
            layout_2_type=LayoutType.IN_DISTRIBUTE_FOUR,
            position_type_2=PositionType.PLANAR,
            positions_2=[
                PlanarPosition(x_c=0.42, y_c=0.42, max_w=0.15, max_h=0.15),
                PlanarPosition(x_c=0.42, y_c=0.58, max_w=0.15, max_h=0.15),
                PlanarPosition(x_c=0.58, y_c=0.42, max_w=0.15, max_h=0.15),
                PlanarPosition(x_c=0.58, y_c=0.58, max_w=0.15, max_h=0.15)
            ],
            number_min_2=0,
            number_max_2=3,
            type_min_2=1,
            size_min_2=2)

    @ classmethod
    def make_all(cls):
        return {
            "center_single":
            Panel.make_center_single(),
            "distribute_four":
            Panel.make_distribute_four(),
            "distribute_nine":
            Panel.make_distribute_nine(),
            "left_center_single_right_center_single":
            Panel.make_left_center_single_right_center_single(),
            "up_center_single_down_center_single":
            Panel.make_up_center_single_down_center_single(),
            "in_center_single_out_center_single":
            Panel.make_in_center_single_out_center_single(),
            "in_distribute_four_out_center_single":
            Panel.make_in_distribute_four_out_center_single()
        }

    def render(self):
        canvas = np.ones((IMAGE_SIZE, IMAGE_SIZE), np.uint8) * 255
        entities = []
        for component in self.components:
            entities.extend(component.entities)
        background = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
        # note left components entities are in the lower layer
        for entity in entities:
            entity_img = entity.render()
            background[entity_img > 0] = 0
            background += entity_img
        structure_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
        if self.structure is StructureType.LEFT_RIGHT:
            structure_img[:, int(0.5 * IMAGE_SIZE)] = 255.0
        elif self.structure is StructureType.UP_DOWN:
            structure_img[int(0.5 * IMAGE_SIZE), :] = 255.0
        background[structure_img > 0] = 0
        background += structure_img
        return canvas - background

    def json(self):
        pass
