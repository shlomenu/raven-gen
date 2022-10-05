# -*- coding: utf-8 -*-

import copy
from dataclasses import dataclass
from typing import Union, List

from configuration import (ANGLE_MAX, ANGLE_MIN, COLOR_MAX, COLOR_MIN, NUM_MAX,
                           NUM_MIN, SIZE_MAX, SIZE_MIN, TYPE_MAX, TYPE_MIN,
                           UNI_MAX, UNI_MIN, PositionType, PlanarPosition,
                           AngularPosition, RuleType, AttributeType)


@dataclass
class Bounds:
    min: int
    max: int


class PositionConstraint:
    position_type: PositionType
    positions: Union[List[PlanarPosition], List[AngularPosition]]


@dataclass
class LayoutConstraints:
    number: Bounds
    position: PositionConstraint
    uniformity: Bounds


def layout_constraints(position_type: PositionType,
                       positions: Union[List[PlanarPosition],
                                        List[AngularPosition]],
                       num_min=NUM_MIN,
                       num_max=NUM_MAX,
                       uni_min=UNI_MIN,
                       uni_max=UNI_MAX):
    return LayoutConstraints(number=Bounds(min=num_min, max=num_max),
                             position=PositionConstraint(
                                 position_type=position_type,
                                 positions=positions[:]),
                             uniformity=Bounds(min=uni_min, max=uni_max))


@dataclass
class EntityConstraints:
    type: Bounds
    size: Bounds
    color: Bounds
    angle: Bounds


def entity_constraints(type_min=TYPE_MIN,
                       type_max=TYPE_MAX,
                       size_min=SIZE_MIN,
                       size_max=SIZE_MAX,
                       color_min=COLOR_MIN,
                       color_max=COLOR_MAX,
                       angle_min=ANGLE_MIN,
                       angle_max=ANGLE_MAX):
    return EntityConstraints(type=Bounds(min=type_min, max=type_max),
                             size=Bounds(min=size_min, max=size_max),
                             color=Bounds(min=color_min, max=color_max),
                             angle=Bounds(min=angle_min, max=angle_max))


def rule_constraint(component_rules, layout_constraints, entity_constraints):
    """
    Modify the bounds of attributes based on the rules that will be applied 
    to them to ensure those rules can be properly expressed with the given 
    range of values.

    When this method is called via `Panel.prune` in the manner done in 
    `main.py`, `layout_constraints` and `entity_constraints` are default 
    bounds based on the structure, components, and layout of the panel.
    Consequently, they may be too tightly or loosely restricted for a 
    sampled rule to be properly expressed with the available value levels.
    When bounds are too tight for a rule to be expressed this function 
    invalidates them by setting the maximum below the minimum; when 
    bounds are too loose, it attempts to tighten them by the minimal
    necessary amount.

    In several rules, you can see calculations that appear to be recapturing
    "original" bounds on attributes.  I believe that references the 
    (temporary) undoing or further manipulation of changes made by 
    this function   
    
    Note that each attribute has at most one rule applied to it and that

        layout_constraint.number.max + 1 == layout.position.values.shape[0].

    :returns: new layout and entity constraints
    """
    new_layout_constraints = copy.deepcopy(layout_constraints)
    new_entity_constraints = copy.deepcopy(entity_constraints)
    for rule in component_rules.all:
        bounds = None
        if rule.attr is AttributeType.NUMBER or rule.attr is AttributeType.POSITION:
            bounds = new_layout_constraints.number
        if rule.attr is AttributeType.TYPE:
            bounds = new_entity_constraints.type
        if rule.attr is AttributeType.SIZE:
            bounds = new_entity_constraints.size
        if rule.attr is AttributeType.COLOR:
            bounds = new_entity_constraints.color

        # rule.value: add/sub how many levels
        if rule.name is RuleType.PROGRESSION:
            if rule.attr is AttributeType.POSITION:
                # creates variable number of empty slots so that shuffling
                #  between positions is visible?
                bounds.max = bounds.max - 2 * abs(rule.value)
            elif bounds:
                # ensures there is room to progress `rule.value` twice
                if rule.value > 0:
                    bounds.max = bounds.max - 2 * rule.value
                else:
                    bounds.min = bounds.min - 2 * rule.value

        # rule.value > 0 if add col_0 + col_1
        # rule.value < 0 if sub col_0 - col_1
        if rule.name is RuleType.ARITHMETIC:
            if rule.attr is AttributeType.POSITION:
                # If the maximum number of objects is present, set union
                #  would be no different from constancy.  And for set
                #  difference to be detectable, you want to disallow
                #  the sampling of disjoint subsets.
                if rule.value <= 0:
                    bounds.min = bounds.max // 2
                bounds.max = bounds.max - 1
            elif rule.attr is AttributeType.NUMBER or rule.attr is AttributeType.SIZE:
                if rule.value > 0:
                    bounds.max = bounds.max - bounds.min - 1
                else:
                    # ensures that both terms of the subtraction are nonzero/non-null
                    bounds.min = 2 * bounds.min + 1
            elif rule.attr is AttributeType.COLOR:
                # at least two different colors
                if bounds.max - bounds.min < 1:
                    bounds.max = bounds.min - 1
                elif rule.value > 0:
                    bounds.max = bounds.max - bounds.min
                elif rule.value < 0:
                    bounds.min = 2 * bounds.min

        if rule.name is RuleType.DISTRIBUTE_THREE:
            # must be at least three configurations
            if rule.attr is AttributeType.POSITION:
                # bounds.max indicates the number of positions; if it
                #  is less than three than this layout doesn't permit
                #  at least three configurations of entities in different
                # positions
                if bounds.max + 1 < 3:
                    bounds.max = bounds.min - 1
                # num_max + 1 == len(layout.position.values)
                # C_{num_max + 1}^{num_value} >= 3
                # C_{num_max + 1} = num_max + 1 >= 3
                # hence only need to constrain num_max: num_max = num_max - 1
                # Check Yang Hui’s Triangle (Pascal's Triangle): https://www.varsitytutors.com/hotmath/hotmath_help/topics/yang-huis-triangle
                else:
                    bounds.max = bounds.max - 1
            elif bounds:
                if bounds.max - bounds.min + 1 < 3:
                    bounds.max = bounds.min - 1

    return new_layout_constraints, new_entity_constraints
