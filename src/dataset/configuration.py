# -*- coding: utf-8 -*-

from enum import Enum, auto
from dataclasses import dataclass
from typing import Union


class RuleType(Enum):
    CONSTANT = auto()
    PROGRESSION = auto()
    ARITHMETIC = auto()
    DISTRIBUTE_THREE = auto()


class LevelType(Enum):
    ATTRIBUTE = -1
    SCENE = 0
    STRUCTURE = 1
    COMPONENT = 2
    LAYOUT = 3
    ENTITY = 4

    def below(self):
        cls = self.__class__
        if self is cls.ATTRIBUTE:
            return self
        else:
            n_incrementable_tags = len(cls.__members__) - 1
            next_value = self.value + 1
            if next_value < n_incrementable_tags:
                return cls(next_value)
            else:
                raise Exception(
                    "Level.ENTITY: ENTITY is lowest level: cannot call `below` method"
                )


class NodeType(Enum):
    LEAF = auto()
    AND = auto()
    OR = auto()


class AttributeType(Enum):
    NUMBER = auto()
    TYPE = auto()
    SIZE = auto()
    COLOR = auto()
    ANGLE = auto()
    UNIFORMITY = auto()
    POSITION = auto()
    NUMBER_OR_POSITION = auto()


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
class Point:
    x: Union[int, float]
    y: Union[int, float]

    def __iter__(self):
        for attr in ["y", "x"]:
            yield getattr(self, attr)


class Shape(Enum):
    NONE = auto()
    TRIANGLE = auto()
    SQUARE = auto()
    PENTAGON = auto()
    HEXAGON = auto()
    CIRCLE = auto()


# Canvas parameters
IMAGE_SIZE = 160
CENTER = Point(x=IMAGE_SIZE // 2, y=IMAGE_SIZE // 2)
DEFAULT_RADIUS = IMAGE_SIZE // 4
DEFAULT_WIDTH = 2

# Attribute parameters
# Number
NUM_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
NUM_MIN = 0
NUM_MAX = len(NUM_VALUES) - 1

# Uniformity
UNI_VALUES = [False, False, False, True]
UNI_MIN = 0
UNI_MAX = len(UNI_VALUES) - 1

# Type
TYPE_VALUES = [t for t in Shape]
TYPE_MIN = 0
TYPE_MAX = len(TYPE_VALUES) - 1

# Size
SIZE_VALUES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SIZE_MIN = 0
SIZE_MAX = len(SIZE_VALUES) - 1

# Color
COLOR_VALUES = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]
COLOR_MIN = 0
COLOR_MAX = len(COLOR_VALUES) - 1

# Angle: self-rotation
ANGLE_VALUES = [-135, -90, -45, 0, 45, 90, 135, 180]
ANGLE_MIN = 0
ANGLE_MAX = len(ANGLE_VALUES) - 1


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


META_TARGET_FORMAT = [
    e.name.title()
    for e in (RuleType.CONSTANT, RuleType.PROGRESSION, RuleType.ARITHMETIC,
              RuleType.DISTRIBUTE_THREE, AttributeType.NUMBER,
              AttributeType.POSITION, AttributeType.TYPE, AttributeType.SIZE,
              AttributeType.COLOR)
]

META_STRUCTURE_FORMAT = [
    e.name.title()
    for e in (StructureType.SINGLETON, StructureType.LEFT_RIGHT,
              StructureType.UP_DOWN, StructureType.OUT_IN, ComponentType.LEFT,
              ComponentType.RIGHT, ComponentType.UP, ComponentType.DOWN,
              ComponentType.OUT, ComponentType.IN, ComponentType.GRID,
              LayoutType.CENTER_SINGLE, LayoutType.DISTRIBUTE_FOUR,
              LayoutType.DISTRIBUTE_NINE, LayoutType.LEFT_CENTER_SINGLE,
              LayoutType.RIGHT_CENTER_SINGLE, LayoutType.UP_CENTER_SINGLE,
              LayoutType.DOWN_CENTER_SINGLE, LayoutType.OUT_CENTER_SINGLE,
              LayoutType.IN_CENTER_SINGLE, LayoutType.IN_DISTRIBUTE_FOUR)
]