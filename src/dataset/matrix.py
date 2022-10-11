from typing import List
from enum import Enum, auto
import copy
import random
import json

import numpy as np
from scipy.special import comb

from component import make_component, ComponentType, LayoutType
from panel import Panel, IMAGE_SIZE
from attribute import AttributeType, PositionType, PlanarPosition
from rule import Rules, RuleType, apply_rule


class StructureType(Enum):
    SINGLETON = auto()
    LEFT_RIGHT = auto()
    UP_DOWN = auto()
    OUT_IN = auto()


class MatrixType(Enum):
    CENTER_SINGLE = auto()
    DISTRIBUTE_FOUR = auto()
    DISTRIBUTE_NINE = auto()
    LEFT_CENTER_SINGLE_RIGHT_CENTER_SINGLE = auto()
    UP_CENTER_SINGLE_DOWN_CENTER_SINGLE = auto()
    IN_CENTER_SINGLE_OUT_CENTER_SINGLE = auto()
    IN_DISTRIBUTE_FOUR_OUT_CENTER_SINGLE = auto()


class Matrix:

    @classmethod
    def make(cls, matrix_type: MatrixType):
        getattr(cls, "make_" + matrix_type.name.lower())()

    @classmethod
    def make_center_single(cls):
        return cls(
            StructureType.SINGLETON,
            Panel(
                make_component(component_type=ComponentType.GRID,
                               layout_type=LayoutType.CENTER_SINGLE,
                               position_type=PositionType.PLANAR,
                               positions=[
                                   PlanarPosition(x_c=0.5,
                                                  y_c=0.5,
                                                  max_w=1,
                                                  max_h=1)
                               ],
                               number_min=0,
                               number_max=0,
                               type_min=1)))

    @classmethod
    def make_distribute_four(cls):
        return cls(
            StructureType.SINGLETON,
            Panel(
                make_component(component_type=ComponentType.GRID,
                               layout_type=LayoutType.DISTRIBUTE_FOUR,
                               position_type=PositionType.PLANAR,
                               positions=[
                                   PlanarPosition(x_c=0.25,
                                                  y_c=0.25,
                                                  max_w=0.5,
                                                  max_y=0.5),
                                   PlanarPosition(x_c=0.25,
                                                  y_c=0.75,
                                                  max_w=0.5,
                                                  max_y=0.5),
                                   PlanarPosition(x_c=0.75,
                                                  y_c=0.25,
                                                  max_w=0.5,
                                                  max_y=0.5),
                                   PlanarPosition(x_c=0.75,
                                                  y_c=0.75,
                                                  max_w=0.5,
                                                  max_y=0.5)
                               ],
                               number_min=0,
                               number_max=3,
                               type_min=1)))

    @classmethod
    def make_distribute_nine(cls):
        return cls(
            StructureType.SINGLETON,
            Panel(
                make_component(component_type=ComponentType.GRID,
                               layout_type=LayoutType.DISTRIBUTE_NINE,
                               position_type=PositionType.PLANAR,
                               positions=[
                                   PlanarPosition(x_c=0.16,
                                                  y_c=0.16,
                                                  max_w=0.33,
                                                  max_y=0.33),
                                   PlanarPosition(x_c=0.16,
                                                  y_c=0.5,
                                                  max_w=0.33,
                                                  max_y=0.33),
                                   PlanarPosition(x_c=0.16,
                                                  y_c=0.83,
                                                  max_w=0.33,
                                                  max_y=0.33),
                                   PlanarPosition(x_c=0.5,
                                                  y_c=0.16,
                                                  max_w=0.33,
                                                  max_y=0.33),
                                   PlanarPosition(x_c=0.5,
                                                  y_c=0.5,
                                                  max_w=0.33,
                                                  max_y=0.33),
                                   PlanarPosition(x_c=0.5,
                                                  y_c=0.83,
                                                  max_w=0.33,
                                                  max_y=0.33),
                                   PlanarPosition(x_c=0.83,
                                                  y_c=0.16,
                                                  max_w=0.33,
                                                  max_y=0.33),
                                   PlanarPosition(x_c=0.83,
                                                  y_c=0.5,
                                                  max_w=0.33,
                                                  max_y=0.33),
                                   PlanarPosition(x_c=0.83,
                                                  y_c=0.83,
                                                  max_w=0.33,
                                                  max_y=0.33)
                               ],
                               number_min=0,
                               number_max=8,
                               type_min=1)))

    @classmethod
    def make_left_center_single_right_center_single(cls):
        return (StructureType.LEFT_RIGHT,
                cls(
                    make_component(component_type=ComponentType.LEFT,
                                   layout_type=LayoutType.LEFT_CENTER_SINGLE,
                                   position_type=PositionType.PLANAR,
                                   positions=[
                                       PlanarPosition(x_c=0.5,
                                                      y_c=0.25,
                                                      max_w=0.5,
                                                      max_h=0.5)
                                   ],
                                   number_min=0,
                                   number_max=0,
                                   type_min=1),
                    make_component(component_type=ComponentType.RIGHT,
                                   layout_type=LayoutType.RIGHT_CENTER_SINGLE,
                                   position_type=PositionType.PLANAR,
                                   positions=[
                                       PlanarPosition(x_c=0.5,
                                                      y_c=0.75,
                                                      max_w=0.5,
                                                      max_h=0.5)
                                   ],
                                   number_min=0,
                                   number_max=0,
                                   type_min=1)))

    @classmethod
    def make_up_center_single_down_center_single(cls):
        (StructureType.UP_DOWN,
         cls(
             make_component(component_type=ComponentType.UP,
                            layout_type=LayoutType.UP_CENTER_SINGLE,
                            position_type=PositionType.PLANAR,
                            positions=[
                                PlanarPosition(x_c=0.25,
                                               y_c=0.5,
                                               max_w=0.5,
                                               max_h=0.5)
                            ],
                            number_min=0,
                            number_max=0,
                            type_min=1),
             make_component(component_type=ComponentType.DOWN,
                            layout_type=LayoutType.DOWN_CENTER_SINGLE,
                            position_type=PositionType.PLANAR,
                            positions=[
                                PlanarPosition(x_c=0.75,
                                               y_c=0.5,
                                               max_w=0.5,
                                               max_h=0.5)
                            ],
                            number_min=0,
                            number_max=0,
                            type_min=1)))

    @classmethod
    def make_in_center_single_out_center_single(cls):
        (StructureType.OUT_IN,
         cls(
             make_component(component_type=ComponentType.OUT,
                            layout_type=LayoutType.OUT_CENTER_SINGLE,
                            position_type=PositionType.PLANAR,
                            positions=[
                                PlanarPosition(x_c=0.5,
                                               y_c=0.5,
                                               max_w=1,
                                               max_h=1)
                            ],
                            number_min=0,
                            number_max=0,
                            type_min=1,
                            size_min=3,
                            color_max=0),
             make_component(component_type=ComponentType.IN,
                            layout_type=LayoutType.IN_CENTER_SINGLE,
                            position_type=PositionType.PLANAR,
                            positions=[
                                PlanarPosition(x_c=0.5,
                                               y_c=0.5,
                                               max_w=0.33,
                                               max_h=0.33)
                            ],
                            number_min=0,
                            number_max=0,
                            type_min=1)))

    @classmethod
    def make_in_distribute_four_out_center_single(cls):
        return (StructureType.OUT_IN,
                cls(
                    make_component(component_type=ComponentType.OUT,
                                   layout_type=LayoutType.OUT_CENTER_SINGLE,
                                   position_type=PositionType.PLANAR,
                                   positions=[
                                       PlanarPosition(x_c=0.5,
                                                      y_c=0.5,
                                                      max_w=1,
                                                      max_h=1)
                                   ],
                                   number_min=0,
                                   number_max=0,
                                   type_min=1,
                                   size_min=3,
                                   color_max=0),
                    make_component(component_type=ComponentType.IN,
                                   layout_type=LayoutType.IN_DISTRIBUTE_FOUR,
                                   position_type=PositionType.PLANAR,
                                   positions=[
                                       PlanarPosition(x_c=0.42,
                                                      y_c=0.42,
                                                      max_w=0.15,
                                                      max_h=0.15),
                                       PlanarPosition(x_c=0.42,
                                                      y_c=0.58,
                                                      max_w=0.15,
                                                      max_h=0.15),
                                       PlanarPosition(x_c=0.58,
                                                      y_c=0.42,
                                                      max_w=0.15,
                                                      max_h=0.15),
                                       PlanarPosition(x_c=0.58,
                                                      y_c=0.58,
                                                      max_w=0.15,
                                                      max_h=0.15)
                                   ],
                                   number_min=0,
                                   number_max=3,
                                   type_min=1,
                                   size_min=2)))

    def __init__(self, structure_type, base):
        self.structure_type = structure_type
        self.panels = []
        while True:
            self.rules = Rules.make_random(n_components=len(base.components))
            pruned = self.prune(base, self.rules)
            self.start = copy.deepcopy(
                pruned) if pruned is not None else pruned
            if self.start is not None:
                for component in base.components:
                    component.reset_constraints()
                break

        for _ in range(3):
            self.panels.extend(self.make_row())

        self.context = self.panels[:-1]
        self.panel_imgs = [panel.render() for panel in self.context] + \
            [np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)]
        self.answer = self.panels[-1]

        modifiable_attr = self.sample_modifiable()
        ground_truth = copy.deepcopy(self.answer)
        self.candidates = [ground_truth]
        for component in ground_truth.components:
            component.reset_history()
        for _ in range(7):
            c, attr = self.sample_attr(modifiable_attr)
            alternative = copy.deepcopy(ground_truth)
            alternative.components[c].sample_unique(attr,
                                                    ground_truth.components[c])
            alternative.sample_unique(c, attr, ground_truth)
            self.candidates.append(alternative)

        random.shuffle(self.candidates)
        self.candidate_imgs = [
            candidate.render() for candidate in self.candidates
        ]
        self.answer_location = self.candidates.index(ground_truth)

    def make_row(self):
        col_0 = copy.deepcopy(self.start)
        for comp in col_0.components:
            comp.sample(sample_position=True,
                        sample_number=True,
                        carryover=False)
        comp_rows = []
        for comp_rules, comp_0 in zip(self.rules, col_0.components):
            comp_1 = apply_rule(comp_rules.number_or_position, comp_0)
            comp_2 = apply_rule(comp_rules.number_or_position, comp_1)
            for rule in comp_rules:
                comp_1 = apply_rule(rule, comp_0, comp_1)
            for rule in comp_rules:
                comp_2 = apply_rule(rule, comp_1, comp_2)
            comp_rows.append((comp_0, comp_1, comp_2))
        if len(comp_rows) == 1:
            comp_rows.append((None, None, None))
        return [Panel(comp_1, comp_2) for (comp_1, comp_2) in zip(*comp_rows)]

    def sample_modifiable(self) -> List[list]:
        """
        Sample available attributes whose values could be modified.
        :returns: list of [component_idx, attribute, available_times, constraints]
        """
        samples = []
        for c, (component, component_rules) in enumerate(
                zip(self.answer.components, self.rules)):
            n_samples, max_entities = 0, component.config.position.values.shape[
                0]
            for n_entities in range(
                    component.initial_constraints.number.min + 1,
                    component.initial_constraints.number.max + 2):
                if n_entities != component.config.number.value:
                    n_samples += comb(max_entities, n_entities)
            if n_samples > 0:
                samples.append([c, AttributeType.NUMBER, n_samples])
            if component_rules.number_or_position.attr is not AttributeType.NUMBER:
                position_samples = comb(max_entities,
                                        component.config.number.value) - 1
                if position_samples > 0:
                    samples.append(
                        [c, AttributeType.POSITION, position_samples])
            for rule in component_rules:
                if rule.name is not RuleType.CONSTANT or \
                    component.uniformity.value or \
                    component_rules.number_or_position.name is RuleType.CONSTANT or \
                    (component_rules.number_or_position.attr is AttributeType.POSITION and
                     component_rules.number_or_position.name is not RuleType.ARITHMETIC):
                    bounds = getattr(component.initial_constraints,
                                     rule.attr.name.lower())
                    if bounds.max - bounds.min > 0:
                        samples.append([c, rule.attr, bounds.max - bounds.min])
        return samples

    def sample_attr(self, attributes):
        """
        Sample an attribute from a list of sampleable attributes and reduce this attributes count
        of available samples by one; if this reduces the count to zero, remove this entry from
        the list.
        """
        i = np.random.choice(len(attributes))
        component_idx, attr_name, available_samples = attributes[i]
        if available_samples - 1 == 0:
            del attributes[i]
        else:
            attributes[i][2] = available_samples - 1
        return component_idx, attr_name

    @staticmethod
    def render(self, panel):
        canvas = np.ones((IMAGE_SIZE, IMAGE_SIZE), np.uint8) * 255
        entities = []
        for component in panel.components:
            entities.extend(component.entities)
        background = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
        # left components entities are in the lower layer
        for entity in entities:
            entity_img = entity.render()
            background[entity_img > 0] = 0
            background += entity_img
        structure_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
        if self.structure_type is StructureType.LEFT_RIGHT:
            structure_img[:, int(0.5 * IMAGE_SIZE)] = 255.0
        elif self.structure_type is StructureType.UP_DOWN:
            structure_img[int(0.5 * IMAGE_SIZE), :] = 255.0
        background[structure_img > 0] = 0
        background += structure_img
        return canvas - background

    def generate_matrix(self, panels):
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

    def generate_answers(self, panels):
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

    def generate_matrix_answer(self, panels):
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

    def merge_matrix_answer(self, matrix_panels, answer_panels):
        """
        Merge question and answer panels into single 5x4 layout with the 3x3 
        question matrix centered above the 2x4 answer matrix.
        :param matrix_panels: list of nine ndarrays in left-to-right, top-to-bottom order
        :param answer_panels: list of eight ndarrays in left-to-right, top-to-bottom order
        :returns: merged ndarray
        """
        matrix_image = self.generate_matrix(matrix_panels)
        answer_image = self.generate_answers(answer_panels)
        img_grid = np.ones(
            (IMAGE_SIZE * 5 + 20, IMAGE_SIZE * 4), np.uint8) * 255
        img_grid[:IMAGE_SIZE * 3,
                 int(0.5 * IMAGE_SIZE):int(3.5 * IMAGE_SIZE)] = matrix_image
        img_grid[-(IMAGE_SIZE * 2):, :] = answer_image
        return img_grid

    def json(self):
        return json.dumps({
            "panels": [panel.json() for panel in self.panels],
            "candidates": [candidate.json() for candidate in self.candidates],
            "answer_index":
            self.answer_location
        })
