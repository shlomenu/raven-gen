from typing import Union
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

NUM_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
NUM_MIN = 0
NUM_MAX = len(NUM_VALUES) - 1

UNI_VALUES = [False, False, False, True]
UNI_MIN = 0
UNI_MAX = len(UNI_VALUES) - 1


class Shape(Enum):
    NONE = auto()
    TRIANGLE = auto()
    SQUARE = auto()
    PENTAGON = auto()
    HEXAGON = auto()
    CIRCLE = auto()


TYPE_VALUES = [t for t in Shape]
TYPE_MIN = 0
TYPE_MAX = len(TYPE_VALUES) - 1

SIZE_VALUES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SIZE_MIN = 0
SIZE_MAX = len(SIZE_VALUES) - 1

COLOR_VALUES = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]
COLOR_MIN = 0
COLOR_MAX = len(COLOR_VALUES) - 1

ANGLE_VALUES = [-135, -90, -45, 0, 45, 90, 135, 180]
ANGLE_MIN = 0
ANGLE_MAX = len(ANGLE_VALUES) - 1


class AttributeType(Enum):
    NUMBER = auto()
    TYPE = auto()
    SIZE = auto()
    COLOR = auto()
    ANGLE = auto()
    UNIFORMITY = auto()
    POSITION = auto()
    CONFIGURATION = auto()


class Attribute:

    def __init__(self, name: AttributeType, values):
        self.name = name
        self.values = np.array(values, dtype="object")

    @property
    def value(self):
        return self.values[self.setting]

    def value_of_setting(self, setting):
        return self.values[setting]

    def reset(self):
        self.setting = self.initial_setting

    def __repr__(self):
        return f"{self.name.name.title()}(setting={self.setting}, value={self.value}))"

    def __str__(self):
        return f"{self.name.name.title()}(setting={self.setting}, value={self.value}))"


class Sampleable(Attribute):

    def __init__(self, name: AttributeType, values, constraints):
        super(Sampleable, self).__init__(name, values)
        self.sample(constraints)
        self.initial_setting = self.setting

    def sample(self, constraints):
        constraint = getattr(constraints, self.name.name.lower())
        self.setting = np.random.choice(
            range(constraint.min, constraint.max + 1))


class UniqueSampleable(Sampleable):

    def sample_unique(self, constraints, history, record=True, inplace=False):
        constraint = getattr(constraints, self.name.name.lower())
        previous_settings = getattr(history, self.name.name.lower())
        all_settings = range(constraint.min, constraint.max + 1)
        new_setting = np.random.choice(
            list(
                set(all_settings) - set(previous_settings) -
                set((self.setting, ))))
        if record:
            if self.setting not in previous_settings:
                previous_settings.append(self.setting)
            previous_settings.append(new_setting)
        if inplace:
            self.setting = new_setting
        return new_setting


class Number(UniqueSampleable):

    def __init__(self, constraints):
        super(Number, self).__init__(name=AttributeType.NUMBER,
                                     values=NUM_VALUES,
                                     constraints=constraints)


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


class Position(Attribute):

    def __init__(self, constraints, n_entities: int):
        self.position_type = constraints.position_type
        super(Position, self).__init__(name=AttributeType.POSITION,
                                       values=constraints.positions)
        self.sample(n_entities)

    def sample(self, size):
        assert size <= self.values.shape[0]
        self.setting = np.random.choice(self.values.shape[0],
                                        size=size,
                                        replace=False)

    def record_setting(self, history, setting, positions_set=None):
        position_history = getattr(history, self.name.name.lower())
        position_history[len(setting)].available -= 1
        position_history[len(setting)].sampled.append(
            set(setting) if positions_set is None else positions_set)

    def sample_unique(self, size, history, record=True, inplace=False):
        position_history = getattr(history, self.name.name.lower())
        new_setting = np.random.choice(self.values.shape[0],
                                       size=size,
                                       replace=False)
        while True:
            unique, new_positions_set = True, set(new_setting)
            current_positions_set = set(self.setting)
            if new_positions_set in position_history[size].sampled or \
               new_positions_set == current_positions_set:
                unique = False
            if unique:
                if record:
                    if current_positions_set not in position_history[len(
                            self.setting)].sampled:
                        self.record_setting(
                            history,
                            self.setting,
                            positions_set=current_positions_set)
                    self.record_setting(history,
                                        new_setting,
                                        positions_set=new_positions_set)
                if inplace:
                    self.setting = new_setting
                return new_setting
            else:
                new_setting = np.random.choice(self.values.shape[0],
                                               size=size,
                                               replace=False)


class Configuration:

    def __init__(self, constraints):
        self.number = Number(constraints)
        self.position = Position(constraints, self.number.value)

    def __repr__(self):
        return f"Configuration(number={self.number!r}, position={self.position!r})"

    def __str__(self):
        return f"Configuration(number={self.number!r}, position={self.position!r})"

    def sample(self, constraints):
        self.number.sample(constraints)
        self.position.sample(self.number.value)

    def sample_unique(self, constraints, history, inplace=False):
        while True:
            number_setting = self.number.sample_unique(constraints,
                                                       history,
                                                       record=False)
            number_value = self.number.value_of_setting(number_setting)
            if history.position[number_value].available == 0:
                continue
            position_setting = self.position.sample_unique(number_value,
                                                           history,
                                                           record=False)
            position_setting_set = set(position_setting)
            if position_setting_set not in history.position[
                    number_value].sampled:
                self.position.record_setting(history, self.position.setting)
                self.position.record_setting(
                    history,
                    position_setting,
                    positions_set=position_setting_set)
                if self.number.setting not in history.number:
                    history.number.append(self.number.setting)
                history.number.append(number_setting)
                if inplace:
                    self.number.setting = number_setting
                    self.position.setting = position_setting
                return self.number.setting, self.position.setting

    def reset(self):
        self.number.reset()
        self.position.reset()


class Type(UniqueSampleable):

    def __init__(self, constraints):
        super(Type, self).__init__(name=AttributeType.TYPE,
                                   values=TYPE_VALUES,
                                   constraints=constraints)


class Size(UniqueSampleable):

    def __init__(self, constraints):
        super(Size, self).__init__(name=AttributeType.SIZE,
                                   values=SIZE_VALUES,
                                   constraints=constraints)


class Color(UniqueSampleable):

    def __init__(self, constraints):
        super(Color, self).__init__(name=AttributeType.COLOR,
                                    values=COLOR_VALUES,
                                    constraints=constraints)


class Angle(UniqueSampleable):

    def __init__(self, constraints):
        super(Angle, self).__init__(name=AttributeType.ANGLE,
                                    values=ANGLE_VALUES,
                                    constraints=constraints)


class Uniformity(Sampleable):

    def __init__(self, constraints):
        super(Uniformity, self).__init__(name=AttributeType.UNIFORMITY,
                                         values=UNI_VALUES,
                                         constraints=constraints)
