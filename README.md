# Distilling Out-of-Distribution Robustness from Vision-Language Foundation Models [NeurIPS 2023]

<img width="974" alt="image" src="https://github.com/andyz245/DiscreteAdversarialDistillation/assets/82240111/c8add5e7-4463-43ed-82fe-7125d700bd02">

https://arxiv.org/abs/2311.01441

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


## Running

Scripts can be found in easyrobust/scripts

Please update the paths with your actual system path.


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
