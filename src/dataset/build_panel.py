# -*- coding: utf-8 -*-

from panel import Component, Layout, Panel, Structure
from configuration import LevelType, StructureType, ComponentType, LayoutType, PositionType, PlanarPosition
from constraints import entity_constraints, layout_constraints


def center_single():
    panel = Panel(LevelType.SCENE)
    panel.structure = Structure(StructureType.SINGLETON)
    panel.structure.insert(Component(ComponentType.GRID))
    panel.structure.components[0].layout = Layout(
        LayoutType.CENTER_SINGLE,
        layout_constraints=layout_constraints(
            position_type=PositionType.PLANAR,
            positions=[PlanarPosition(x_c=0.5, y_c=0.5, max_w=1, max_h=1)],
            num_min=0,
            num_max=0),
        entity_constraints=entity_constraints(type_min=1))
    return panel


def distribute_four():
    panel = Panel(LevelType.SCENE)
    panel.structure = Structure(StructureType.SINGLETON)
    panel.structure.insert(Component(ComponentType.GRID))
    panel.structure.components[0].layout = Layout(
        LayoutType.DISTRIBUTE_FOUR,
        layout_constraints=layout_constraints(
            position_type=PositionType.PLANAR,
            positions=[
                PlanarPosition(x_c=0.25, y_c=0.25, max_w=0.5, max_y=0.5),
                PlanarPosition(x_c=0.25, y_c=0.75, max_w=0.5, max_y=0.5),
                PlanarPosition(x_c=0.75, y_c=0.25, max_w=0.5, max_y=0.5),
                PlanarPosition(x_c=0.75, y_c=0.75, max_w=0.5, max_y=0.5)
            ],
            num_min=0,
            num_max=3),
        entity_constraints=entity_constraints(type_min=1))
    return panel


def distribute_nine():
    panel = Panel(LevelType.SCENE)
    panel.structure = Structure(StructureType.SINGLETON)
    panel.structure.insert(Component(ComponentType.GRID))
    panel.structure.components[0].layout = Layout(
        LayoutType.DISTRIBUTE_NINE,
        layout_constraints=layout_constraints(
            position_type=PositionType.PLANAR,
            positions=[
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
            num_min=0,
            num_max=8),
        entity_constraints=entity_constraints(type_min=1))
    return panel


def left_center_single_right_center_single():
    panel = Panel(LevelType.SCENE)
    panel.structure = Structure(StructureType.LEFT_RIGHT)
    panel.structure.insert(Component(ComponentType.LEFT))
    panel.structure.components[0].layout = Layout(
        LayoutType.LEFT_CENTER_SINGLE,
        layout_constraints=layout_constraints(
            position_type=PositionType.PLANAR,
            positions=[
                PlanarPosition(x_c=0.5, y_c=0.25, max_w=0.5, max_h=0.5)
            ],
            num_min=0,
            num_max=0),
        entity_constraints=entity_constraints(type_min=1))
    panel.structure.insert(Component(ComponentType.RIGHT))
    panel.structure.components[1].layout = Layout(
        LayoutType.RIGHT_CENTER_SINGLE,
        layout_constraints=layout_constraints(PositionType.PLANAR,
                                              positions=[
                                                  PlanarPosition(x_c=0.5,
                                                                 y_c=0.75,
                                                                 max_w=0.5,
                                                                 max_h=0.5)
                                              ],
                                              num_min=0,
                                              num_max=0),
        entity_constraints=entity_constraints(type_min=1))
    return panel


def up_center_single_down_center_single():
    panel = Panel(LevelType.SCENE)
    panel.structure = Structure(StructureType.UP_DOWN)
    panel.structure.insert(Component(ComponentType.UP))
    panel.structure.components[0].layout = Layout(
        LayoutType.UP_CENTER_SINGLE,
        layout_constraints=layout_constraints(
            position_type=PositionType.PLANAR,
            positions=[
                PlanarPosition(x_c=0.25, y_c=0.5, max_w=0.5, max_h=0.5)
            ],
            num_min=0,
            num_max=0),
        entity_constraints=entity_constraints(type_min=1))
    panel.structure.insert(Component(ComponentType.DOWN))
    panel.structure.components[1].layout = Layout(
        LayoutType.DOWN_CENTER_SINGLE,
        layout_constraints=layout_constraints(
            position_type=PositionType.PLANAR,
            positions=[
                PlanarPosition(x_c=0.75, y_c=0.5, max_w=0.5, max_h=0.5)
            ],
            num_min=0,
            num_max=0),
        entity_constraints=entity_constraints(type_min=1))
    return panel


def in_center_single_out_center_single():
    panel = Panel(LevelType.SCENE)
    panel.structure = Structure(StructureType.OUT_IN)
    panel.structure.insert(Component(ComponentType.OUT))
    panel.structure.components[0].layout = Layout(
        LayoutType.OUT_CENTER_SINGLE,
        layout_constraints=layout_constraints(
            position_type=PositionType.PLANAR,
            positions=[PlanarPosition(x_c=0.5, y_c=0.5, max_w=1, max_h=1)],
            num_min=0,
            num_max=0),
        entity_constraints=entity_constraints(type_min=1,
                                              size_min=3,
                                              color_max=0))
    panel.structure.insert(Component(ComponentType.IN))
    panel.structure.components[1].layout = Layout(
        LayoutType.IN_CENTER_SINGLE,
        layout_constraints=layout_constraints(
            position_type=PositionType.PLANAR,
            positions=[
                PlanarPosition(x_c=0.5, y_c=0.5, max_w=0.33, max_h=0.33)
            ],
            num_min=0,
            num_max=0),
        entity_constraints=entity_constraints(type_min=1))
    return panel


def in_distribute_four_out_center_single():
    panel = Panel(LevelType.SCENE)
    panel.structure = Structure(StructureType.OUT_IN)
    panel.structure.insert(Component(ComponentType.OUT))
    panel.structure.components[0].layout = Layout(
        LayoutType.OUT_CENTER_SINGLE,
        layout_constraints=layout_constraints(
            position_type=PositionType.PLANAR,
            positions=[PlanarPosition(x_c=0.5, y_c=0.5, max_w=1, max_h=1)],
            num_min=0,
            num_max=0),
        entity_constraints=entity_constraints(type_min=1,
                                              size_min=3,
                                              color_max=0))
    panel.structure.insert(Component(ComponentType.IN))
    panel.structure.components[1].layout = Layout(
        LayoutType.IN_DISTRIBUTE_FOUR,
        layout_constraints=layout_constraints(
            position_type=PositionType.PLANAR,
            positions=[
                PlanarPosition(x_c=0.42, y_c=0.42, max_w=0.15, max_h=0.15),
                PlanarPosition(x_c=0.42, y_c=0.58, max_w=0.15, max_h=0.15),
                PlanarPosition(x_c=0.58, y_c=0.42, max_w=0.15, max_h=0.15),
                PlanarPosition(x_c=0.58, y_c=0.58, max_w=0.15, max_h=0.15)
            ],
            num_min=0,
            num_max=3),
        entity_constraints=entity_constraints(type_min=1, size_min=2))
    return panel
