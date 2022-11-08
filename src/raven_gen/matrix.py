from typing import List
from enum import Enum, auto
import copy, os, math

import numpy as np
from scipy.special import comb
from PIL import Image

from .component import make_component, ComponentType, LayoutType, AttributeHistory
from .panel import Panel, prune
from . import attribute
from .attribute import AttributeType, PositionType, PlanarPosition
from .rule import Ruleset, RuleType, apply_rule


class StructureType(Enum):
    SINGLETON = auto()
    LEFT_RIGHT = auto()
    UP_DOWN = auto()
    OUT_IN = auto()


class MatrixType(Enum):
    ONE_SHAPE = auto()
    FOUR_SHAPE = auto()
    FIVE_SHAPE = auto()
    NINE_SHAPE = auto()
    TWO_SHAPE_VERTICAL_SEP = auto()
    TWO_SHAPE_HORIZONTAL_SEP = auto()
    SHAPE_IN_SHAPE = auto()
    FOUR_SHAPE_IN_SHAPE = auto()


class Matrix:

    attribute_bounds = {
        MatrixType.ONE_SHAPE: {
            (ComponentType.NONE, LayoutType.CENTER): {}
        },
        MatrixType.FOUR_SHAPE: {
            (ComponentType.NONE, LayoutType.GRID_FOUR): {}
        },
        MatrixType.FIVE_SHAPE: {
            (ComponentType.NONE, LayoutType.GRID_FIVE): {}
        },
        MatrixType.NINE_SHAPE: {
            (ComponentType.NONE, LayoutType.GRID_NINE): {}
        },
        MatrixType.TWO_SHAPE_VERTICAL_SEP: {
            (ComponentType.LEFT, LayoutType.CENTER): {},
            (ComponentType.RIGHT, LayoutType.CENTER): {}
        },
        MatrixType.TWO_SHAPE_HORIZONTAL_SEP: {
            (ComponentType.UP, LayoutType.CENTER): {},
            (ComponentType.DOWN, LayoutType.CENTER): {}
        },
        MatrixType.SHAPE_IN_SHAPE: {
            (ComponentType.OUT, LayoutType.CENTER): {},
            (ComponentType.IN, LayoutType.CENTER): {}
        },
        MatrixType.FOUR_SHAPE_IN_SHAPE: {
            (ComponentType.OUT, LayoutType.CENTER): {},
            (ComponentType.IN, LayoutType.GRID_FOUR): {}
        }
    }

    @staticmethod
    def oblique_angle_rotations(allowed: bool):
        if allowed:
            if attribute.ANGLE_VALUES != (-135, -90, -45, 0, 45, 90, 135, 180):
                attribute.ANGLE_VALUES = (-135, -90, -45, 0, 45, 90, 135, 180)
                attribute.ANGLE_MIN, attribute.ANGLE_MAX = 0, len(
                    attribute.ANGLE_VALUES) - 1
        else:
            if attribute.ANGLE_VALUES != (-90, 0, 90, 180):
                attribute.ANGLE_VALUES = (-90, 0, 90, 180)
                attribute.ANGLE_MIN, attribute.ANGLE_MAX = 0, len(
                    attribute.ANGLE_VALUES) - 1

    @classmethod
    def make(cls,
             matrix_type: MatrixType,
             ruleset: Ruleset = None,
             n_alternatives: int = 0):
        return getattr(cls, matrix_type.name.lower())(
            cls.get_attribute_bounds(matrix_type), ruleset, n_alternatives)

    @staticmethod
    def validate_bounds(src, dst, attribute: AttributeType, min_setting,
                        max_setting):
        if attribute is AttributeType.NUMBER:
            min_key, max_key = "number_min", "number_max"
        elif attribute is AttributeType.SHAPE:
            min_key, max_key = "shape_min", "shape_max"
        elif attribute is AttributeType.SIZE:
            min_key, max_key = "size_min", "size_max"
        elif attribute is AttributeType.COLOR:
            min_key, max_key = "color_min", "color_max"
        elif attribute is AttributeType.ANGLE:
            min_key, max_key = "angle_min", "angle_max"
        elif attribute is AttributeType.UNIFORMITY:
            min_key, max_key = "uniformity_min", "uniformity_max"
        if min_key not in src:
            dst[min_key] = min_setting
        else:
            dst[min_key] = max(min(src[min_key], max_setting), min_setting)
        if max_key not in src:
            dst[max_key] = max_setting
        else:
            dst[max_key] = max(min(src[max_key], max_setting), min_setting)
        if dst[max_key] < dst[min_key]:
            dst[max_key] = dst[min_key]

    @classmethod
    def get_attribute_bounds(cls, matrix_type: MatrixType):
        validated = {}
        for (component_type,
             layout_type), src in cls.attribute_bounds[matrix_type].items():
            assert (component_type in ComponentType
                    and layout_type in LayoutType)
            dst = {}
            if layout_type is LayoutType.CENTER:
                number_max = attribute.NUM_MIN
            elif layout_type is LayoutType.GRID_FOUR:
                number_max = attribute.NUM_MIN + 3
            elif layout_type is LayoutType.GRID_FIVE:
                number_max = attribute.NUM_MIN + 4
            else:  # layout_type is LayoutType.GRID_NINE
                number_max = attribute.NUM_MIN + 8
            cls.validate_bounds(src, dst, AttributeType.NUMBER,
                                attribute.NUM_MIN, number_max)
            cls.validate_bounds(src, dst, AttributeType.SHAPE,
                                attribute.SHAPE_MIN + 1, attribute.SHAPE_MAX)
            if component_type is ComponentType.OUT:
                size_min, color_max = attribute.SIZE_MIN + 3, attribute.COLOR_MIN
            elif component_type is ComponentType.IN and layout_type is LayoutType.GRID_FOUR:
                size_min = attribute.SIZE_MIN + 2
            else:
                size_min, color_max = attribute.SIZE_MIN, attribute.COLOR_MAX
            cls.validate_bounds(src, dst, AttributeType.SIZE, size_min,
                                attribute.SIZE_MAX)
            cls.validate_bounds(src, dst, AttributeType.COLOR,
                                attribute.COLOR_MIN, color_max)
            cls.validate_bounds(src, dst, AttributeType.ANGLE,
                                attribute.ANGLE_MIN, attribute.ANGLE_MAX)
            cls.validate_bounds(src, dst, AttributeType.UNIFORMITY,
                                attribute.UNI_MIN, attribute.UNI_MAX)
            validated[(component_type, layout_type)] = dst
        cls.attribute_bounds[matrix_type] = validated
        return validated

    @classmethod
    def one_shape(cls, bounds, ruleset, n_alternatives):
        return cls(
            StructureType.SINGLETON,
            Panel(
                make_component(
                    ComponentType.NONE, LayoutType.CENTER, PositionType.PLANAR,
                    [PlanarPosition(x_c=0.5, y_c=0.5, max_w=1, max_h=1)],
                    **(bounds[(ComponentType.NONE, LayoutType.CENTER)]))),
            ruleset, n_alternatives)

    @classmethod
    def four_shape(cls, bounds, ruleset, n_alternatives):
        return cls(
            StructureType.SINGLETON,
            Panel(
                make_component(
                    ComponentType.NONE, LayoutType.GRID_FOUR,
                    PositionType.PLANAR, [
                        PlanarPosition(
                            x_c=0.25, y_c=0.25, max_w=0.5, max_h=0.5),
                        PlanarPosition(
                            x_c=0.25, y_c=0.75, max_w=0.5, max_h=0.5),
                        PlanarPosition(
                            x_c=0.75, y_c=0.25, max_w=0.5, max_h=0.5),
                        PlanarPosition(
                            x_c=0.75, y_c=0.75, max_w=0.5, max_h=0.5)
                    ],
                    **(bounds[(ComponentType.NONE, LayoutType.GRID_FOUR)]))),
            ruleset, n_alternatives)

    @classmethod
    def five_shape(cls, bounds, ruleset, n_alternatives):
        return cls(
            StructureType.SINGLETON,
            Panel(
                make_component(
                    ComponentType.NONE, LayoutType.GRID_FIVE,
                    PositionType.PLANAR, [
                        PlanarPosition(
                            x_c=0.2, y_c=0.2, max_w=0.33, max_h=0.33),
                        PlanarPosition(
                            x_c=0.2, y_c=0.8, max_w=0.33, max_h=0.33),
                        PlanarPosition(
                            x_c=0.5, y_c=0.5, max_w=0.33, max_h=0.33),
                        PlanarPosition(
                            x_c=0.8, y_c=0.2, max_w=0.33, max_h=0.33),
                        PlanarPosition(
                            x_c=0.8, y_c=0.8, max_w=0.33, max_h=0.33)
                    ],
                    **(bounds[(ComponentType.NONE, LayoutType.GRID_FIVE)]))),
            ruleset, n_alternatives)

    @classmethod
    def nine_shape(cls, bounds, ruleset, n_alternatives):
        return cls(
            StructureType.SINGLETON,
            Panel(
                make_component(
                    ComponentType.NONE, LayoutType.GRID_NINE,
                    PositionType.PLANAR, [
                        PlanarPosition(
                            x_c=0.16, y_c=0.16, max_w=0.33, max_h=0.33),
                        PlanarPosition(
                            x_c=0.16, y_c=0.5, max_w=0.33, max_h=0.33),
                        PlanarPosition(
                            x_c=0.16, y_c=0.83, max_w=0.33, max_h=0.33),
                        PlanarPosition(
                            x_c=0.5, y_c=0.16, max_w=0.33, max_h=0.33),
                        PlanarPosition(
                            x_c=0.5, y_c=0.5, max_w=0.33, max_h=0.33),
                        PlanarPosition(
                            x_c=0.5, y_c=0.83, max_w=0.33, max_h=0.33),
                        PlanarPosition(
                            x_c=0.83, y_c=0.16, max_w=0.33, max_h=0.33),
                        PlanarPosition(
                            x_c=0.83, y_c=0.5, max_w=0.33, max_h=0.33),
                        PlanarPosition(
                            x_c=0.83, y_c=0.83, max_w=0.33, max_h=0.33)
                    ],
                    **(bounds[(ComponentType.NONE, LayoutType.GRID_NINE)]))),
            ruleset, n_alternatives)

    @classmethod
    def two_shape_vertical_sep(cls, bounds, ruleset, n_alternatives):
        return cls(
            StructureType.LEFT_RIGHT,
            Panel(
                make_component(
                    ComponentType.LEFT, LayoutType.CENTER, PositionType.PLANAR,
                    [PlanarPosition(x_c=0.5, y_c=0.25, max_w=0.5, max_h=0.5)],
                    **(bounds[(ComponentType.LEFT, LayoutType.CENTER)])),
                make_component(
                    ComponentType.RIGHT, LayoutType.CENTER,
                    PositionType.PLANAR,
                    [PlanarPosition(x_c=0.5, y_c=0.75, max_w=0.5, max_h=0.5)],
                    **(bounds[(ComponentType.RIGHT, LayoutType.CENTER)]))),
            ruleset, n_alternatives)

    @classmethod
    def two_shape_horizontal_sep(cls, bounds, ruleset, n_alternatives):
        return cls(
            StructureType.UP_DOWN,
            Panel(
                make_component(
                    ComponentType.UP, LayoutType.CENTER, PositionType.PLANAR,
                    [PlanarPosition(x_c=0.25, y_c=0.5, max_w=0.5, max_h=0.5)],
                    **(bounds[(ComponentType.UP, LayoutType.CENTER)])),
                make_component(
                    ComponentType.DOWN, LayoutType.CENTER, PositionType.PLANAR,
                    [PlanarPosition(x_c=0.75, y_c=0.5, max_w=0.5, max_h=0.5)],
                    **(bounds[(ComponentType.DOWN, LayoutType.CENTER)]))),
            ruleset, n_alternatives)

    @classmethod
    def shape_in_shape(cls, bounds, ruleset, n_alternatives):
        return cls(
            StructureType.OUT_IN,
            Panel(
                make_component(
                    ComponentType.OUT, LayoutType.CENTER, PositionType.PLANAR,
                    [PlanarPosition(x_c=0.5, y_c=0.5, max_w=1, max_h=1)],
                    **(bounds[(ComponentType.OUT, LayoutType.CENTER)])),
                make_component(
                    ComponentType.IN, LayoutType.CENTER, PositionType.PLANAR,
                    [PlanarPosition(x_c=0.5, y_c=0.5, max_w=0.33, max_h=0.33)],
                    **(bounds[(ComponentType.IN, LayoutType.CENTER)]))),
            ruleset, n_alternatives)

    @classmethod
    def four_shape_in_shape(cls, bounds, ruleset, n_alternatives):
        return cls(
            StructureType.OUT_IN,
            Panel(
                make_component(
                    ComponentType.OUT, LayoutType.CENTER, PositionType.PLANAR,
                    [PlanarPosition(x_c=0.5, y_c=0.5, max_w=1, max_h=1)],
                    **(bounds[(ComponentType.OUT, LayoutType.CENTER)])),
                make_component(
                    ComponentType.IN, LayoutType.GRID_FOUR,
                    PositionType.PLANAR, [
                        PlanarPosition(
                            x_c=0.42, y_c=0.42, max_w=0.15, max_h=0.15),
                        PlanarPosition(
                            x_c=0.42, y_c=0.58, max_w=0.15, max_h=0.15),
                        PlanarPosition(
                            x_c=0.58, y_c=0.42, max_w=0.15, max_h=0.15),
                        PlanarPosition(
                            x_c=0.58, y_c=0.58, max_w=0.15, max_h=0.15)
                    ], **(bounds[(ComponentType.IN, LayoutType.GRID_FOUR)]))),
            ruleset, n_alternatives)

    def __init__(self, structure_type, base, ruleset, n_alternatives):
        self.structure_type = structure_type
        self.initial_constraints = tuple(
            copy.deepcopy(comp.constraints) for comp in base.components)
        self.make_ground_truth(base, ruleset)
        self.make_alternatives(n_alternatives)

    def make_ground_truth(self, base, ruleset):
        if ruleset is None:
            ruleset = Ruleset()
        while True:
            self.rules = ruleset.sample(
                tuple((c.component_type, c.layout_type)
                      for c in base.components))
            pruned = prune(base, self.rules)
            self.start = copy.deepcopy(
                pruned) if pruned is not None else pruned
            if self.start is not None:
                break

        panels = []
        for _ in range(3):
            panels.extend(self.make_row())

        self.context, self.answer = panels[:-1], panels[-1]

    def make_row(self):
        col_0 = copy.deepcopy(self.start)
        for comp in col_0.components:
            comp.sample(sample_position=True,
                        sample_number=True,
                        carryover=False)
        comp_rows = []
        for comp_rules, comp_0 in zip(self.rules, col_0.components):
            comp_1 = apply_rule(comp_rules.configuration, comp_0)
            comp_2 = apply_rule(comp_rules.configuration, comp_1)
            for rule in comp_rules:
                comp_1 = apply_rule(rule, comp_0, comp_1)
            for rule in comp_rules:
                comp_2 = apply_rule(rule, comp_1, comp_2)
            comp_rows.append((comp_0, comp_1, comp_2))
        if len(comp_rows) == 1:
            comp_rows.append((None, None, None))
        return [Panel(comp_1, comp_2) for (comp_1, comp_2) in zip(*comp_rows)]

    def make_alternatives(self, n_alternatives):
        self.alternatives = []
        if n_alternatives > 0:
            self.modifications = self.count_modifiable()
            self.uniques = [
                AttributeHistory(cst) for cst in self.initial_constraints
            ]
            for _ in range(n_alternatives):
                if len(self.modifications) == 0:
                    break
                c, attr = self.sample_modification()
                alternative = copy.deepcopy(self.answer)
                alternative.components[c].sample_unique(
                    attr, self.uniques[c], self.initial_constraints[c])
                self.alternatives.append(alternative)

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
            if component_rules.configuration.attr is not AttributeType.NUMBER:
                position_samples = comb(max_entities,
                                        component.config.number.value) - 1
                if position_samples > 0:
                    samples.append(
                        [c, AttributeType.POSITION, position_samples])
            for rule in component_rules:
                if rule.name is not RuleType.CONSTANT or \
                    component.uniformity.value or \
                    component_rules.configuration.name is RuleType.CONSTANT or \
                    (component_rules.configuration.attr is AttributeType.POSITION and
                     component_rules.configuration.name is not RuleType.ARITHMETIC):
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

    def render(self, panel, background_color, panel_size, line_thickness,
               shape_border_thickness):
        entities = []
        for component in panel.components:
            entities.extend(component.entities)
        panel_img = np.ones(
            (panel_size, panel_size), np.uint8) * background_color
        for entity in entities:
            entity_img = entity.render(background_color, panel_size,
                                       shape_border_thickness)
            panel_img[entity_img != background_color] = 0
            entity_img[entity_img == background_color] = 0
            panel_img += entity_img
        return panel_img

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

    def generate_matrix(self, last_panel, background_color, image_size,
                        line_thickness, shape_border_thickness):
        panel_size = image_size // 3
        img_grid = np.ones(
            (image_size, image_size), np.uint8) * background_color
        for pos, panel in enumerate([
                self.render(panel, background_color, panel_size,
                            line_thickness, shape_border_thickness)
                for panel in self.context
        ] + [
                self.render(last_panel, background_color, panel_size,
                            line_thickness, shape_border_thickness)
        ]):
            i, j = divmod(pos, 3)
            img_grid[i * panel_size:(i + 1) * panel_size,
                     j * panel_size:(j + 1) * panel_size] = panel
        neg_offset, pos_offset = math.floor(line_thickness / 2), math.ceil(
            line_thickness / 2)
        for x in [0.33, 0.67]:
            band_center = int(x * panel_size * 3)
            img_grid[band_center - neg_offset:band_center + pos_offset, :] = 0
        for y in [0.33, 0.67]:
            band_center = int(y * panel_size * 3)
            img_grid[:, band_center - neg_offset:band_center + pos_offset] = 0
        if self.structure_type is StructureType.LEFT_RIGHT:
            for i in range(3):
                band_center = i * panel_size + int(0.5 * panel_size)
                img_grid[:, band_center - neg_offset:band_center +
                         pos_offset] = 0.
        elif self.structure_type is StructureType.UP_DOWN:
            for i in range(3):
                band_center = i * panel_size + int(0.5 * panel_size)
                img_grid[band_center - neg_offset:band_center +
                         pos_offset, :] = 0.
        return Image.fromarray(img_grid)

    def save(self,
             path,
             puzzle_name,
             background_color=255,
             image_size=480,
             line_thickness=3,
             shape_border_thickness=2):
        image_size, background_color, line_thickness, shape_border_thickness = \
            int(abs(image_size)), int(abs(background_color)), int(abs(line_thickness)), int(abs(shape_border_thickness))
        assert (image_size != 0 and background_color <= 255)
        img = self.generate_matrix(self.answer, background_color, image_size,
                                   line_thickness, shape_border_thickness)
        img.save(os.path.join(path, puzzle_name + "_answer.png"))
        for i, alternative in enumerate(self.alternatives):
            img = self.generate_matrix(alternative, background_color,
                                       image_size, line_thickness,
                                       shape_border_thickness)
            img.save(os.path.join(path, puzzle_name + f"_alternative_{i}.png"))
