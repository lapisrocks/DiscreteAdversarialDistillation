"""
@misc{robustness,
   title={Robustness (Python Library)},
   author={Logan Engstrom and Andrew Ilyas and Hadi Salman and Shibani Santurkar and Dimitris Tsipras},
   year={2019},
   url={https://github.com/MadryLab/robustness}
}
"""

import random
import torch
import torch.nn as nn
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from easyrobust.third_party.vqgan import VQModel, reconstruct_with_vqgan
from distance_attack import *

import torch.nn.functional as F

class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, orig_input, eps, step_size, use_grad=True):
        '''
        Initialize the attacker step with a given perturbation magnitude.
        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set
        Args:
            ch.tensor x : the input to project back into the feasible set.
        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).
        Parameters:
            g (ch.tensor): the raw gradient
        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

    def to_image(self, x):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''
        return x

# L-infinity threat model
class LinfStep(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = torch.clamp(diff, -self.eps, self.eps)
        return torch.clamp(diff + self.orig_input, 0, 1)

    def step(self, x, g):
        """
        """
        step = torch.sign(g) * self.step_size
        return x + step

    def random_perturb(self, x):
        """
        """
        new_x = x + 2 * (torch.rand_like(x) - 0.5) * self.eps
        return torch.clamp(new_x, 0, 1)

class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return torch.clamp(self.orig_input + diff, 0, 1)

    def step(self, x, g):
        """
        """
        l = len(x.shape) - 1
        g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def random_perturb(self, x):
        """
        """
        l = len(x.shape) - 1
        rp = torch.randn_like(x)
        rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))
        return torch.clamp(x + self.eps * rp / (rp_norm + 1e-10), 0, 1)

def replace_best(loss, bloss, x, bx, m):
    if bloss is None:
        bx = x.clone().detach()
        bloss = loss.clone().detach()
    else:
        replace = m * bloss < m * loss
        bx[replace] = x[replace].clone().detach()
        bloss[replace] = loss[replace]

    return bloss, bx

def robustkd_attack(y_pred_student, y_pred_teacher, y_pred_aug, y_true):
    loss_fn=nn.KLDivLoss(reduction="batchmean", log_target=True)
    ce_loss = SoftTargetCrossEntropy()
    temp = 4.0
        
    soft_teacher_out = F.log_softmax(y_pred_teacher / temp, dim=1)
    soft_student_aug_out = F.log_softmax(y_pred_aug / temp, dim=1)
    soft_student_out = F.log_softmax(y_pred_student / temp, dim=1)
    
    loss = temp * temp * loss_fn(soft_student_aug_out, soft_teacher_out)
    dl = temp * temp * loss_fn(soft_student_aug_out, soft_student_out)
    class_loss = ce_loss(y_pred_aug, y_true)

    return loss + dl + class_loss

def kd_attack(y_pred_student, y_pred_teacher, y_pred_aug, y_true):
    loss_fn=nn.KLDivLoss(reduction="batchmean", log_target=True)
    temp = 4.0
        
    soft_teacher_out = F.log_softmax(y_pred_teacher / temp, dim=1)
    soft_student_aug_out = F.log_softmax(y_pred_aug / temp, dim=1)
    
    loss = temp * temp * loss_fn(soft_student_aug_out, soft_teacher_out)

    return loss

def invar_kd_attack(y_pred_student, y_pred_teacher, y_pred_aug, y_true):
    loss_fn=nn.KLDivLoss(reduction="batchmean", log_target=True)
    temp = 4.0
        
    soft_teacher_out = F.log_softmax(y_pred_teacher / temp, dim=1)
    soft_student_aug_out = F.log_softmax(y_pred_aug / temp, dim=1)
    soft_student_out = F.log_softmax(y_pred_student / temp, dim=1)
    
    loss = temp * temp * loss_fn(soft_student_aug_out, soft_teacher_out)
    dl = temp * temp * loss_fn(soft_student_aug_out, soft_student_out)

    loss += dl

    return loss

def invar_attack(y_pred_student, y_pred_teacher, y_pred_aug, y_true):
    distance_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    tau=4.0
    ce_loss = SoftTargetCrossEntropy()

    y_s = (y_pred_student / tau).log_softmax(dim=1)
    y_al = (y_pred_aug / tau).log_softmax(dim=1)

    loss = ce_loss(y_pred_aug, y_true)
    dist_loss = tau**2 * distance_loss(y_al, y_s)
    
    loss += dist_loss

    return loss

def dis_attack(y_pred_student, y_pred_teacher, y_pred_aug, y_true):
    ce_loss = SoftTargetCrossEntropy()
    distance_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    beta=2.0
    gamma=2.0
    tau=4.0
    delta=1.0

    y_t = (y_pred_teacher / tau).softmax(dim=1)
    y_s = (y_pred_student / tau).log_softmax(dim=1)
    y_a = (y_pred_aug / tau).softmax(dim=1)
    y_al = (y_pred_aug / tau).log_softmax(dim=1)

    inter_loss = tau**2 * inter_class_relation(y_a, y_t)
    intra_loss = tau**2 * intra_class_relation(y_a, y_t)

    dist_loss = tau**2 * distance_loss(y_al, y_s)
    kd_loss = beta * inter_loss + gamma * intra_loss
    class_loss = ce_loss(y_pred_aug, y_true)

    return kd_loss + class_loss + dist_loss

def pgd_generator(images, ogimages, target, model, teacher, model_out, teacher_out, vqgan_aug, attack_type='Linf', eps=4/255, attack_steps=3, attack_lr=4/255*2/3, random_start_prob=0.0, targeted=False, attack_criterion='regular', use_best=True, eval_mode=True):
    # generate adversarial examples
    attack = None
    prev_training = bool(model.training)
    if eval_mode:
        model.eval()
    orig_input = images.detach()
    assert attack_type in ['Linf', 'L2'], '{} is not supported!'.format(attack_type)
    
    if attack_type == 'Linf':
        step = LinfStep(eps=eps, orig_input=orig_input, step_size=attack_lr)
    elif attack_type == 'L2':
        step = L2Step(eps=eps, orig_input=orig_input, step_size=attack_lr)

    if attack_criterion == 'regular':
        attack = torch.nn.CrossEntropyLoss(reduction='none')
    elif attack_criterion == 'smooth':
        attack = LabelSmoothingCrossEntropy()
    elif attack_criterion == 'mixup':
        attack = SoftTargetCrossEntropy()
    
    temp_criterion = SoftTargetCrossEntropy()

    m = -1 if targeted else 1
    best_loss = None
    best_x = None

    y_t = teacher_out.clone().detach()
    y_s = model_out.clone().detach()

    if random.random() < random_start_prob:
        images = step.random_perturb(images)

    prev_images = images.clone().detach()

    for attack_step in range(attack_steps):
        images = images.clone().detach().requires_grad_(True)
        
        if attack_criterion == 'robustkd':
            adv_losses = robustkd_attack(y_s, y_t, model(images), target)
        elif attack_criterion == 'kd':
            adv_losses = kd_attack(y_s, y_t, model(images), target)
        elif attack_criterion == 'invar':
            adv_losses = invar_attack(y_s, y_t, model(images), target)
        elif attack_criterion == 'invarkd':
            adv_losses = invar_kd_attack(y_s, y_t, model(images), target)
        elif attack_criterion == 'cos':
            adv_losses = dis_attack(y_s, y_t, model(images), target)
        else:
            adv_losses = temp_criterion(model(images), target)

        torch.mean(m * adv_losses).backward()
        grad = images.grad.detach()

        with torch.no_grad():
            images = step.step(images, grad)
            images = step.project(images)
            images = reconstruct_with_vqgan(images, vqgan_aug)

            pred = torch.argmax(teacher(images), dim=1)
            if pred != target:
                images = prev_images
                break
            else:
                prev_images = images.clone()

    if prev_training:
        model.train()
    
    return images