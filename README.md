# Distilling Out-of-Distribution Robustness from Vision-Language Foundation Models [NeurIPS 2023]

## Installation
### Install from Source:
```bash
$ git clone https://github.com/andyz245/RobustKD.git
$ cd easyrobust
$ pip install -e .
$ pip install PyWavelets
$ pip install matplotlib
$ pip install tensorboard 
```

### Evaluation Data and Models
```bash
$ sh download_data.sh
```

Teacher can be found at:
https://drive.google.com/file/d/1I7h2oe-LP4djuMwFAwhD4kmvOrsgoDEN/view?usp=sharing

Student models will be released soon.

## Running

Scripts can be found in easyrobust/scripts

Please update the paths with your actual system path.

### DAD Samples

We open source our set of DAD clip samples, from ImageNet. This will be released soon.

### Citation

If you found our paper or repo interesting, thanks! Please feel free to give us a star or cite our paper!

```bibtex
@inproceedings{
    zhou2023distilling,
    title={Distilling Out-of-Distribution Robustness from Vision-Language Foundation Models},
    author={Andy Zhou and Jindong Wang and Yu-Xiong Wang and Haohan Wang},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=iwp3H8uSeK}
}

```
