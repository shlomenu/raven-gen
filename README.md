# raven-gen

This repo contains a rewrite of the data-generation code originating from the CVPR paper:

[RAVEN: A Dataset for <u>R</u>elational and <u>A</u>nalogical <u>V</u>isual r<u>E</u>aso<u>N</u>ing](http://wellyzhang.github.io/attach/cvpr19zhang.pdf)  
Chi Zhang*, Feng Gao*, Baoxiong Jia, Yixin Zhu, Song-Chun Zhu  
*Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019   
(* indicates equal contribution.)

Dramatic progress has been witnessed in basic vision tasks involving low-level perception, such as object recognition, detection, and tracking. Unfortunately, there is still an enormous performance gap between artificial vision systems and human intelligence in terms of higher-level vision problems, especially ones involving reasoning. Earlier attempts in equipping machines with high-level reasoning have hovered around Visual Question Answering (VQA), one typical task associating vision and language understanding. In this work, we propose a new dataset, built in the context of Raven's Progressive Matrices (RPM) and aimed at lifting machine intelligence by associating vision with structural, relational, and analogical reasoning in a hierarchical representation. Unlike previous works in measuring abstract reasoning using RPM, we establish a semantic link between vision and reasoning by providing structure representation. This addition enables a new type of abstract reasoning by jointly operating on the structure representation. Machine reasoning ability using modern computer vision is evaluated in this newly proposed dataset. Additionally, we also provide human performance as a reference. Finally, we show consistent improvement across all models by incorporating a simple neural module that combines visual understanding and structure reasoning.

```
@inproceedings{zhang2019raven, 
    title={RAVEN: A Dataset for Relational and Analogical Visual rEasoNing}, 
    author={Zhang, Chi and Gao, Feng and Jia, Baoxiong and Zhu, Yixin and Zhu, Song-Chun}, 
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
    year={2019}
}
```

#### DISCLAIMER:

I do not offer assurances that the code in this repository functions identically to the code in the original author's repository, or that the performance of models on data generated with this code may be directly and fairly compared to the performance of models on the published RAVEN dataset of Zhang et al. (2019).  Authors who utilize this code to generate data and wish to compare model performance against that of models trained with the original RAVEN dataset should offer support for the fairness of this comparison and/or replicate previous work as necessary.

## Sampling and Format

Due to deficiencies in the sampling strategy implemented by the original RAVEN authors, a number of researchers observed that puzzles could be solved from the answer grid alone, and implemented improved perturbation strategies to prevent this issue.  This package does not yet implement these strategies, so it does not offer convenience methods to synthesize incomplete matrices or answer grids.  Saving a generated matrix instead produces completed matrices, one named `NAME_answer.png` with a correct bottom right tile, and zero or more alternatives named `NAME_alternative_{i}.png` with incorrect bottom right tiles.  This makes `raven-gen` datasets suitable for binary classification rather than 8-way classification when training models to solve RPMs in the traditional sense.  If training models to answer questions about other patterns exhibited by (correct) RPMs, various kinds of annotations may be derived by inspecting the abstractions discussed below and contained in this package.  Convenience functions for retrieving and saving such alternative annotations may be added in future releases.

## Goals & Usage

When producing RPM-style puzzles for downstream machine learning tasks, it is important to be able to adjust both the logical and rendering-related parameters of the RPM generation process.  This allows researchers to tailor the patterns they expect their models to generalize, and to adjust puzzle resolution to fit their needs.  `raven-gen` aims to offer simple and flexible abstractions for programmatic RPM generation.  The following demonstrates basic usage of the `Matrix` class to generate a correct matrix of a random type (one of `MatrixType.{ONE_SHAPE,FOUR_SHAPE,FIVE_SHAPE,NINE_SHAPE,TWO_SHAPE_VERTICAL_SEP,TWO_SHAPE_HORIZONTAL_SEP,SHAPE_IN_SHAPE,FOUR_SHAPE_IN_SHAPE}`) with default parameters (480x480 resolution, white background, 3 pixel line width, 2 pixel shape border width) in the current directory:

```
>>> from raven_gen import Matrix, MatrixType
>>> import numpy as np
>>> rpm = Matrix.make(np.random.choice(list(MatrixType)))
>>> rpm.save(".", "test")
>>> print(rpm)
>>> print(rpm.rules)
```

To generate _incorrect_ puzzles as well, one may add an `rpm.make_alternatives(n_alternatives)` call before calling `rpm.save`; or one may provide the `n_alternatives` keyword argument to `Matrix.make`.  Note that calls to `rpm.make_alternatives` are destructive; if repeated, they will erase the previously-generated incorrect puzzle variants.  

### Custom Rulesets & Attribute Setting Ranges

Puzzles exhibit all kinds of row-wise variation (any of `RuleType.{CONSTANT,PROGRESSION,ARITHMETIC,DISTRIBUTE_THREE}`) across all applicable traits (`AttributeType.{POSITION,NUMBER,SIZE,SHAPE,COLOR}`) except shape, for which the arithmetic rule is disallowed by default due to being unintuitive.  To customize this behavior, provide a custom `Ruleset` to `Matrix.make`:

```
>>> from raven_gen import Ruleset, RuleType
>>> ruleset = Ruleset(size_rules=[RuleType.CONSTANT, RuleType.PROGRESSION], shape_rules=list(RuleType))
>>> rpm = Matrix.make(np.random.choice(list(MatrixType)), ruleset=ruleset)
```

The resulting RPM may utilize the constant or progression rules to determine the sizes of shapes across rows, and may utilize any rule, including the arithmetic rule, to determine the number of sides those shapes will have. 

To adjust the permissible ranges of attributes, one may modify the `Matrix.attribute_bounds` class variable.  The contents of this dictionary are validated each time a matrix is generated, so the insertion of invalid settings should not prevent puzzle generation.  

### Low Resolution Puzzle Generation

A common reason for changing the valid range of an attribute setting is enforcing larger minimum shape sizes.  This is necessary to produce
legible puzzles at smaller resolutions.  At lower resolutions, oblique angle rotations can also cause shapes to be hard to discern.  The following demonstrates the application of custom settings suitable for generating legible (if not very aesthetically-pleasing) 96x96 resolution RPMs:

```

>>> Matrix.oblique_angle_rotations(allowed=False)
>>> ruleset = Ruleset(size_rules=[RuleType.CONSTANT])
>>> matrix_types = [
...     MatrixType.ONE_SHAPE, MatrixType.FOUR_SHAPE, MatrixType.FIVE_SHAPE,
...     MatrixType.TWO_SHAPE_VERTICAL_SEP, MatrixType.TWO_SHAPE_HORIZONTAL_SEP,
...     MatrixType.SHAPE_IN_SHAPE,
... ]
>>> weights = [.15, .2, .2, .15, .15, .15]
>>> Matrix.attribute_bounds[MatrixType.FOUR_SHAPE][(
...     ComponentType.NONE, LayoutType.GRID_FOUR)]["size_min"] = 3
>>> Matrix.attribute_bounds[MatrixType.FIVE_SHAPE][(
...     ComponentType.NONE, LayoutType.GRID_FIVE)]["size_min"] = 5
>>> Matrix.attribute_bounds[MatrixType.TWO_SHAPE_VERTICAL_SEP][(
...     ComponentType.LEFT, LayoutType.CENTER)]["size_min"] = 3
>>> Matrix.attribute_bounds[MatrixType.TWO_SHAPE_VERTICAL_SEP][(
...     ComponentType.RIGHT, LayoutType.CENTER)]["size_min"] = 3
>>> Matrix.attribute_bounds[MatrixType.TWO_SHAPE_HORIZONTAL_SEP][(
...     ComponentType.UP, LayoutType.CENTER)]["size_min"] = 3
>>> Matrix.attribute_bounds[MatrixType.TWO_SHAPE_HORIZONTAL_SEP][(
...     ComponentType.DOWN, LayoutType.CENTER)]["size_min"] = 3
>>> Matrix.attribute_bounds[MatrixType.SHAPE_IN_SHAPE][(
...     ComponentType.OUT, LayoutType.CENTER)]["size_min"] = 5
>>> Matrix.attribute_bounds[MatrixType.SHAPE_IN_SHAPE][(
...     ComponentType.IN, LayoutType.CENTER)]["size_min"] = 5
>>> for i in trange(size):
...     rpm = Matrix.make(np.random.choice(matrix_types, p=weights),
...                       ruleset=ruleset,
...                       n_alternatives=1)
...     for background_color in range(28, 225, 28):
...         rpm.save(".",
...                  f"rpm_{i}_background_{background_color}",
...                  background_color,
...                  image_size=96,
...                  line_thickness=1,
...                  shape_border_thickness=1)
```

The above also exhibits the usage of the `line_thickness`, `shape_border_thickness`, and `background_color` parameters to `rpm.save`.  Generating data in all background colors (or in random background colors) may be helpful for balancing rewards in representation learning tasks.

### Custom Attribute Values

To change the values that underlie specific attribute settings, one may redefine the "constants" defined in `raven_gen.attribute`:

```
>>> import raven_gen
>>> raven_gen.attribute.SIZE_VALUES
(0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
>>> raven_gen.attribute.SIZE_VALUES = (0.3, 0.45, 0.6, 0.75, 0.9)
>>> raven_gen.attribute.SIZE_MIN
0
>>> raven_gen.attribute.SIZE_MAX = len(raven_gen.attribute.SIZE_VALUES) - 1
```

The bounds applied in `Matrix.attribute_bounds` will now refer (by index) to this underlying set of values when determining the sizes of shapes.  Please take caution when adjusting attribute values as the defaults have been tuned for appearance under the default rendering settings.  If one desires only to change the range of settings that is permitted, it is preferable to modify `Matrix.attribute_bounds`.  Specific assumptions governing these constants include:
 - `raven_gen.attribute.*_VALUES` constants are assumed to represent strictly
increasing or decreasing sequences.  If this assumption is violated rules may have an incoherent presentation.
 - `raven_gen.attribute.NUM_VALUES` must be of the form `tuple(range(1, x))` where `x >= 10`.
 - `raven_gen.attribute.SHAPE_VALUES` cannot be adjusted.
 - `raven_gen.attribute.SIZE_VALUES` must be numbers in (0, 1]; extreme values in this range will be indiscernible or too large for the panels they occupy.
 - `raven_gen.attribute.COLOR_VALUES` are integers in [0, 256).
 - `raven_gen.attribute.UNI_VALUES` are booleans; the relative occurrence of `True` and `False` in this sequence indicates the probability that angles are random or uniform across rows.

Upon altering `raven_gen.attribute.*_VALUES`, one must ensure that `raven_gen.attribute.*_{MIN, MAX}` refer to the correct boundary indices of the new sequence.

### Custom Components/Layouts

The only setting that cannot be customized by the above means (aside from the set of valid shapes) are the positions and sizes of the boxes that shapes occupy within a panel.  These settings are determined by the branch of `MatrixType` that is supplied to `Matrix.make`.  One may generate matrices that customize this behavior by inheriting from the `Matrix` class, imitating the structure of the methods called by `Matrix.make`, and adding top-level entries to `Matrix.attribute_bounds`.  However, the stdlib enums used to describe the types of matrices, components, and layouts cannot be flexibly extended, so the `str` representations of `rpm` and `rpm.rules` will not be available.  This may be ameliorated in later releases.  

### Serialization and Deserialization

The `Matrix` class does not currently offer deserialization from human readable formats.  Users interested in rerendering a matrix with different settings or otherwise interrogating the logical specification of a matrix outside of the originating program/interpreter lifetime must pickle Matrix objects.  `Matrix` objects do not store previous renderings, so this will not duplicate the memory of previously generated images.  A human readable serialization of `Matrix` objects and their rules may be accessed/persisted by calling `str` and saving this output.  This summary CANNOT be used to regain the original `Matrix` object and this package does not offer any programmatic tools for inspecting these summaries.