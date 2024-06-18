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

### Example Checkpoints

Base ResNet50
https://drive.google.com/file/d/1TGvOW6vit4wA1PlzOFomRnKUf1ZjxuWu/view?usp=sharing

Base ViT-B-16/224
https://drive.google.com/file/d/1ly0j_2nfphNgFd2NOJ0NhIea9o3JXqZ_/view?usp=sharing

ResNet50 + DAD
https://drive.google.com/file/d/1lbYzO7dp9xqwVMx2NOjPS_X9B3jh-LHv/view?usp=sharing

## Running

Scripts can be found in easyrobust/scripts

Please update the paths with your actual system path. To reproduce paper results, please initialize the model with the base pretrained ViT or ResNet50 models.


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
