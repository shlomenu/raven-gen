# -*- coding: utf-8 -*-

import numpy as np
from typing import Union, List

from configuration import (ANGLE_MAX, ANGLE_MIN, ANGLE_VALUES, COLOR_MAX,
                           COLOR_MIN, COLOR_VALUES, NUM_MAX, NUM_MIN,
                           NUM_VALUES, SIZE_MAX, SIZE_MIN, SIZE_VALUES,
                           TYPE_MAX, TYPE_MIN, TYPE_VALUES, UNI_MAX, UNI_MIN,
                           UNI_VALUES, AttributeType, LevelType,
                           AngularPosition, PlanarPosition, PositionType)


class Attribute:
    """
    Provides associative-array functionality between a integer-valued 
    settings of an attribute and corresponding real values.  Settings 
    are called value levels, and their corresponding object is called 
    a value.  Value levels are constraint by boundaries; for most 
    subclasses, this is simply a minimum and maximum index into 
    the integer-array of value levels.  For others it may be more 
    complex.  These boundaries are used by `Rule`s to enforce patterns 
    by constraining the random variation in values of attributes.

    Subclasses are expected to provide the following methods:

        `sample`: set this instance's value level to a bounded 
            randomly-sampled value.

        `sample_new`: return a bounded randomly-sampled value level 
            that this attribute has not taken before according 
            to `self.previous_values` and the current value level.  
            This method does not update `self.previous_values` or 
            modify the current value level in place.  It only returns
            a unique value level.
    """

    def __init__(self, name: AttributeType, values, value_level):
        self.name = name
        self.level = LevelType.ATTRIBUTE
        self.previous_values = []
        self.value_level = value_level
        self.values = np.array(values, dtype="object")

    def value(self, value_level=None):
        return self.values[value_level if value_level is not None else self.
                           value_level].tolist()

    def __repr__(self):
        return self.level + "." + self.name

    def __str__(self):
        return self.level + "." + self.name


class BoundedLevel(Attribute):

    def __init__(self, name, values, value_level, min_level, max_level):
        super(BoundedLevel, self).__init__(name=name,
                                           values=values,
                                           value_level=value_level)
        self.min_level = min_level
        self.max_level = max_level


class NonResampleable(BoundedLevel):
    """
    A subclass of Attribute supporting bounded levels and `sample`,
    but not `sample_new`.
    """

    def sample(self, min_level=None, max_level=None):
        self.value_level = np.random.choice(
            range(self.min_level, self.max_level + 1))

    def sample_new(self, min_level=None, max_level=None, previous_values=None):
        pass


class Resampleable(BoundedLevel):
    """
    A subclass of Attribute supporting bounded levels, `sample`,
    and `sample_new`.
    """

    def sample(self, min_level=NUM_MIN, max_level=NUM_MAX):
        min_level = max(self.min_level, min_level)
        max_level = min(self.max_level, max_level)
        self.value_level = np.random.choice(range(min_level, max_level + 1))

    def sample_new(self, min_level=None, max_level=None, previous_values=None):
        if min_level is None or max_level is None:
            min_level, max_level = self.min_level, self.max_level
        values = range(min_level, max_level + 1)
        if not previous_values:
            previous_values = self.previous_values
        return np.random.choice(
            list(set(values) - set(previous_values) - set([self.value_level])))


class Number(Resampleable):

    def __init__(self, min_level=NUM_MIN, max_level=NUM_MAX):
        super(Number, self).__init__(name=AttributeType.NUMBER,
                                     values=NUM_VALUES,
                                     value_level=0,
                                     min_level=min_level,
                                     max_level=max_level)


class Type(Resampleable):

    def __init__(self, min_level=TYPE_MIN, max_level=TYPE_MAX):
        super(Type, self).__init__(name=AttributeType.TYPE,
                                   values=TYPE_VALUES,
                                   value_level=0,
                                   min_level=min_level,
                                   max_level=max_level)


class Size(Resampleable):

    def __init__(self, min_level=SIZE_MIN, max_level=SIZE_MAX):
        super(Type, self).__init__(name=AttributeType.SIZE,
                                   values=SIZE_VALUES,
                                   value_level=3,
                                   min_level=min_level,
                                   max_level=max_level)


class Color(Resampleable):

    def __init__(self, min_level=COLOR_MIN, max_level=COLOR_MAX):
        super(Color, self).__init__(name=AttributeType.COLOR,
                                    values=COLOR_VALUES,
                                    value_level=0,
                                    min_level=min_level,
                                    max_level=max_level)


class Angle(Resampleable):

    def __init__(self, min_level=ANGLE_MIN, max_level=ANGLE_MAX):
        super(Angle, self).__init__(name=AttributeType.ANGLE,
                                    values=ANGLE_VALUES,
                                    value_level=3,
                                    min_level=min_level,
                                    max_level=max_level)


class Uniformity(NonResampleable):

    def __init__(self, min_level=UNI_MIN, max_level=UNI_MAX):
        super(Uniformity, self).__init__(name=AttributeType.UNIFORMITY,
                                         values=UNI_VALUES,
                                         value_level=0,
                                         min_level=min_level,
                                         max_level=max_level)


class Position(Attribute):
    """
    Stores the (im/proper) subset of bounding boxes available in a layout
    that are currently occupied by this layout's entities.  `Position` instances
    are tightly-coupled with corresponding `Number` instances, which are used to 
    determine the size of the subset.  A `Position` instance may consist of 
    bounding boxes that encode rotation information (`AngularPosition`s) or 
    bounding boxes which do not (`PlanarPosition`s).  Handles boundaries in 
    a more complex way and implements `sample` and `sample_new`.
    """

    def __init__(self, position_type: PositionType,
                 positions: Union[List[PlanarPosition],
                                  List[AngularPosition]]):
        assert position_type in PositionType
        self.position_type = position_type
        super(Position, self).__init__(name=AttributeType.POSITION,
                                       values=positions,
                                       value_level=None)

    def sample(self, size):
        assert size <= self.values.shape[0]
        self.value_level = np.random.choice(self.values.shape[0],
                                            size=size,
                                            replace=False)

    def sample_new(self, size, previous_values=None):
        constraints = (previous_values if previous_values is not None else
                       self.previous_values) + [self.value_level]
        while True:
            new_value_level = np.random.choice(self.values.shape[0],
                                               size=size,
                                               replace=False)
            nobreak = True
            for current_or_previous_value in constraints:
                if set(new_value_level) == set(current_or_previous_value):
                    nobreak = False
                    break
            if nobreak:
                return new_value_level

    def sample_addable(self, size: int) -> List[int]:
        """
        Sample `size` additional positions that may be added to the layout.
        Add additional positions to current value level and return the
        positions added.
        """
        unused_positions = list(
            set(range(self.values.shape[0])) - set(self.value_level))
        new_positions = np.random.choice(unused_positions,
                                         size=size,
                                         replace=False)
        self.value_level = np.insert(self.value_level, 0, new_positions)
        return self.values[new_positions].tolist()

    def remove(self, bbox: Union[PlanarPosition, AngularPosition]):
        """
        Remove the specified bounding box from the possible values and 
        current value level of this attribute.
        """
        bbox_idx = np.min(np.nonzero(self.values == bbox)[0])
        bbox_idx_loc = np.nonzero(self.value_level == bbox_idx)[0]
        assert bbox_idx_loc.shape == (1, )
        self.value_level = np.delete(self.value_level, bbox_idx_loc[0])
