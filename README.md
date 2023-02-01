# RobustKnowledgeDistillation

## Installation
### Install from Source:
```bash
$ git clone https://github.com/andyz245/RobustKD.git
$ cd easyrobust
$ pip install -e .
```

### Evaluation Data and Models
```bash
$ sh download_data.sh
```

Teacher can be found at:
https://drive.google.com/file/d/1I7h2oe-LP4djuMwFAwhD4kmvOrsgoDEN/view?usp=sharing

## Running

Scripts can be found in easyrobust/scripts

ARD: Baseline knowledge distillation with DAT

KDARD: Knowledge distillation with DAT, distillation loss used to generate perturbation

ARDWD: Knowledge distillation with invariance loss. Also used to generate perturbation

Final: Knowledge distillation with invariance loss. Perturbation generated with all three losses (CE, KD, Invar)

RSLAD: Knowledge distillation without hard labels. Perturbation generated with KD and Invar loss. CE loss removed from objective
