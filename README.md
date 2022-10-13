# raven-tool

This repo contains a rewrite of the data-generation code originating from the CVPR paper:

---

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

# Scientific Replication

I do not offer assurances that the code in this repository functions identically to the code in the original authors repository, or that the performance of models on data generated with this code may be directly and fairly compared to the performance of models on the published RAVEN dataset of Zhang et al. (2019).  Authors who utilize this code to generate data and wish to compare model performance against that of models trained with the original RAVEN dataset should offer support for the fairness of this comparison and/or replicate previous results as appropriate.

# Goals

THe original code used to generate the RAVEN dataset did not offer suitable abstractions for generating RPM problems in close-coordination with other programs.  Reasons for wanting more programmatically-friendly RPM generation code include: custom specifications for RPMs, on-demand generation for testing data efficiency-related metrics.  With the included modules, one can

```
>> from matrix import Matrix
>> rpm = Matrix.make(MatrixType.BRANCH, rulesets)
>> rpm.make_alternatives(N_ALTERNATIVES)
>> rpm.save("path/to/data/dir", "PUZZLENAME")
>> with open("path/to/meta/dir/PUZZLENAME_rpm.txt") as f:
...    f.write(str(rpm))
>> with open("path/to/meta/dir/PUZZLENAME_rules.txt") as f:
...    f.write(str(rpm.rules))
```

where the `MatrixType` enumeration includes the expected seven branches:
- `CENTER_SINGLE`
- `DISTRIBUTE_FOUR`
- `DISTRIBUTE_NINE`
- `LEFT_CENTER_SINGLE_RIGHT_CENTER_SINGLE`
- `UP_CENTER_SINGLE_DOWN_CENTER_SINGLE`
- `IN_CENTER_SINGLE_OUT_CENTER_SINGLE`
- `IN_DISTRIBUTE_FOUR_OUT_CENTER_SINGLE`

The `rulesets` parameter of `Matrix.make` may be `None` or may be omitted.  Custom rulesets of type `List[List[Tuple[RuleType, AttributeType]]]` are expected in the following format: 
```
[[..., (RuleType.*, AttributeType.{NUMBER,POSITION,CONFIGURATION}), ...],
 [..., (RuleType.{CONSTANT, PROGRESSION, DISTRIBUTE_THREE}, AttributeType.TYPE)],
 [..., (RuleType.*, AttributeType.SIZE)],
 [..., (RuleType.*, AttributeType.COLOR)]].
```

where no inner list is empty.

To generate at most `N_ALTERNATIVES` wrong answers (fewer if there are not more modifications possible), `Matrix.make_alternatives` must be called.  `Matrix.save` may be called with or without a prior call to `Matrix.make_alternatives`.  `Matrix.make_alternatives` may be called any number of times, but overwrites the results of previous calls.  

The call to `Matrix.save` results in `PUZZLENAME_answer.png` and `PUZZLENAME_alternative_{i}.png` files being created in the specified directory.  These images show a completed Raven's progressive matrix with either the correct answer or the `i`th alternative (incorrect) answer as the ninth (bottom-right) panel.  This data format is intended for a binary prediction task rather than the task of picking the right completion out of a lineup.  As such, we do not implement the alternative sampling mechanisms of the balanced RAVEN dataset or similar improvement efforts. 

# Serialization and Deserialization

The `Matrix` object not currently offer full serialization and deserialization methods to/from human readable formats.  However, if individuals want only to save a human readable report on a particular puzzle, they may call `str` on the RPM and its `rules` attribute.  These outputs provide a full view of the patterns expressed in a puzzle, though an understanding of the code implementing them may be needed to understand all of their governing conventions.  These outputs cannot be used to regain the originating `Matrix` object if it has not been otherwise saved; please pickle your `Matrix` objects if you anticipate inspecting them programatically after they and their human-readable metadata have been saved and the originating object lost.