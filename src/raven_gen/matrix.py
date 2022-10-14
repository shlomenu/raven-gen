from typing import List
from enum import Enum, auto
import copy, os

import numpy as np
from scipy.special import comb
from PIL import Image

from .entity import IMAGE_SIZE
from .component import make_component, ComponentType, LayoutType, AttributeHistory
from .panel import Panel, prune
from .attribute import AttributeType, PositionType, PlanarPosition
from .rule import Rules, RuleType, apply_rule


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
    def make(cls, matrix_type: MatrixType, rulesets=None):
        return getattr(cls,
                       "make_" + matrix_type.name.lower())(rulesets=rulesets)

    @classmethod
    def make_center_single(cls, rulesets=None):
        return cls(StructureType.SINGLETON,
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
                                      type_min=1)),
                   rulesets=rulesets)

    @classmethod
    def make_distribute_four(cls, rulesets=None):
        return cls(StructureType.SINGLETON,
                   Panel(
                       make_component(component_type=ComponentType.GRID,
                                      layout_type=LayoutType.DISTRIBUTE_FOUR,
                                      position_type=PositionType.PLANAR,
                                      positions=[
                                          PlanarPosition(x_c=0.25,
                                                         y_c=0.25,
                                                         max_w=0.5,
                                                         max_h=0.5),
                                          PlanarPosition(x_c=0.25,
                                                         y_c=0.75,
                                                         max_w=0.5,
                                                         max_h=0.5),
                                          PlanarPosition(x_c=0.75,
                                                         y_c=0.25,
                                                         max_w=0.5,
                                                         max_h=0.5),
                                          PlanarPosition(x_c=0.75,
                                                         y_c=0.75,
                                                         max_w=0.5,
                                                         max_h=0.5)
                                      ],
                                      number_min=0,
                                      number_max=3,
                                      type_min=1)),
                   rulesets=rulesets)

    @classmethod
    def make_distribute_nine(cls, rulesets=None):
        return cls(StructureType.SINGLETON,
                   Panel(
                       make_component(component_type=ComponentType.GRID,
                                      layout_type=LayoutType.DISTRIBUTE_NINE,
                                      position_type=PositionType.PLANAR,
                                      positions=[
                                          PlanarPosition(x_c=0.16,
                                                         y_c=0.16,
                                                         max_w=0.33,
                                                         max_h=0.33),
                                          PlanarPosition(x_c=0.16,
                                                         y_c=0.5,
                                                         max_w=0.33,
                                                         max_h=0.33),
                                          PlanarPosition(x_c=0.16,
                                                         y_c=0.83,
                                                         max_w=0.33,
                                                         max_h=0.33),
                                          PlanarPosition(x_c=0.5,
                                                         y_c=0.16,
                                                         max_w=0.33,
                                                         max_h=0.33),
                                          PlanarPosition(x_c=0.5,
                                                         y_c=0.5,
                                                         max_w=0.33,
                                                         max_h=0.33),
                                          PlanarPosition(x_c=0.5,
                                                         y_c=0.83,
                                                         max_w=0.33,
                                                         max_h=0.33),
                                          PlanarPosition(x_c=0.83,
                                                         y_c=0.16,
                                                         max_w=0.33,
                                                         max_h=0.33),
                                          PlanarPosition(x_c=0.83,
                                                         y_c=0.5,
                                                         max_w=0.33,
                                                         max_h=0.33),
                                          PlanarPosition(x_c=0.83,
                                                         y_c=0.83,
                                                         max_w=0.33,
                                                         max_h=0.33)
                                      ],
                                      number_min=0,
                                      number_max=8,
                                      type_min=1)),
                   rulesets=rulesets)

    @classmethod
    def make_left_center_single_right_center_single(cls, rulesets=None):
        return cls(
            StructureType.LEFT_RIGHT,
            Panel(
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
                               type_min=1)),
            rulesets=rulesets)

    @classmethod
    def make_up_center_single_down_center_single(cls, rulesets=None):
        return cls(StructureType.UP_DOWN,
                   Panel(
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
                       make_component(
                           component_type=ComponentType.DOWN,
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
                           type_min=1)),
                   rulesets=rulesets)

    @classmethod
    def make_in_center_single_out_center_single(cls, rulesets=None):
        return cls(StructureType.OUT_IN,
                   Panel(
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
                                      type_min=1)),
                   rulesets=rulesets)

    @classmethod
    def make_in_distribute_four_out_center_single(cls, rulesets=None):
        return cls(StructureType.OUT_IN,
                   Panel(
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
                       make_component(
                           component_type=ComponentType.IN,
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
                           size_min=2)),
                   rulesets=rulesets)

    def __init__(self, structure_type, base, rulesets=None):
        self.structure_type = structure_type
        self.initial_constraints = tuple(
            copy.deepcopy(comp.constraints) for comp in base.components)
        self.make_ground_truth(base, rulesets=rulesets)

    def make_ground_truth(self, base, rulesets=None):
        while True:
            self.rules = Rules.make_random(n_components=len(base.components),
                                           rulesets=None)
            pruned = prune(base, self.rules)
            self.start = copy.deepcopy(
                pruned) if pruned is not None else pruned
            if self.start is not None:
                break

        panels = []
        for _ in range(3):
            panels.extend(self.make_row())

        self.context, self.answer = panels[:-1], panels[-1]
        self.context_imgs = [self.render(panel) for panel in self.context]
        self.answer_img = self.render(self.answer)

    def make_alternatives(self, n_alternatives):
        self.modifications = self.count_modifiable()
        self.uniques = [
            AttributeHistory(cst) for cst in self.initial_constraints
        ]
        self.alternatives = []
        for _ in range(n_alternatives):
            if len(self.modifications) == 0:
                break
            c, attr = self.sample_modification()
            alternative = copy.deepcopy(self.answer)
            alternative.components[c].sample_unique(
                attr, self.uniques[c], self.initial_constraints[c])
            self.alternatives.append(alternative)

        self.alternatives_imgs = [
            self.render(panel) for panel in self.alternatives
        ]

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

    def count_modifiable(self) -> List[list]:
        samples = []
        for c, (component, component_rules, initial_constraints) in enumerate(
                zip(self.answer.components, self.rules,
                    self.initial_constraints)):
            n_samples, max_entities = 0, component.config.position.values.shape[
                0]
            for n_entities in range(
                    component.config.number.value_of_setting(
                        initial_constraints.number.min),
                    component.config.number.value_of_setting(
                        initial_constraints.number.max) + 1):
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
                    bounds = getattr(initial_constraints,
                                     rule.attr.name.lower())
                    n_samples = bounds.max - bounds.min  # excludes one setting
                    if not component.uniformity.value:
                        n_samples //= len(component.entities)
                    if n_samples > 0:
                        samples.append([c, rule.attr, n_samples])
        return samples

    def sample_modification(self):
        i = np.random.choice(len(self.modifications))
        component_idx, attr_name, available_samples = self.modifications[i]
        if available_samples - 1 == 0:
            del self.modifications[i]
        else:
            self.modifications[i][2] = available_samples - 1
        return component_idx, attr_name

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

    def __str__(self):
        s = f"\nstructure: {self.structure_type.name}:\n"
        for i, panel in enumerate(self.context):
            if (i % 3) == 0:
                s += "\n\t+++++++++++ ROW +++++++++++\n\n"
            elif i != 0:
                s += "\n\t*********** PANEL ***********\n\n"
            s += str(panel)
        s += "\n\t*********** SOLUTION ***********\n\n"
        s += str(self.answer)
        for alternative in self.alternatives:
            s += "\n\t*********** ALTERNATIVE ***********\n\n"
            s += str(alternative)
        return s

    def generate_matrix(self, last_panel_img):
        img_grid = np.zeros((IMAGE_SIZE * 3, IMAGE_SIZE * 3), np.uint8)
        for pos, panel in enumerate(self.context_imgs + [last_panel_img]):
            i, j = divmod(pos, 3)
            img_grid[i * IMAGE_SIZE:(i + 1) * IMAGE_SIZE,
                     j * IMAGE_SIZE:(j + 1) * IMAGE_SIZE] = panel
        for x in [0.33, 0.67]:
            band_center = int(x * IMAGE_SIZE * 3)
            img_grid[band_center - 1:band_center + 1, :] = 0
        for y in [0.33, 0.67]:
            band_center = int(y * IMAGE_SIZE * 3)
            img_grid[:, band_center - 1:band_center + 1] = 0
        return Image.fromarray(img_grid)

    def save(self, path, puzzle_name):
        img = self.generate_matrix(self.answer_img)
        img.save(os.path.join(path, puzzle_name + "_answer.png"))
        for i, alternative_img in enumerate(self.alternatives_imgs):
            img = self.generate_matrix(alternative_img)
            img.save(os.path.join(path, puzzle_name + f"_alternative_{i}.png"))
