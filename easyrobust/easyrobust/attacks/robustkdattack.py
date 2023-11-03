import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from timm.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import math

class Attack(nn.Module):
    """
    Original implementation of Knowledge distillation from the paper "Distilling the
    Knowledge in a Neural Network" https://arxiv.org/pdf/1503.02531.pdf
    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module):  Calculates loss during distillation
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
        self,
        loss_fn=nn.KLDivLoss(reduction="batchmean", log_target=True),
        temp=20.0,
        distil_weight=0.5,
        device="cuda",
    ):
        super(Attack, self).__init__()
        
    def forward(self, y_pred_student, y_pred_aug, y_pred_teacher, labels):
        distance_loss = nn.CosineEmbeddingLoss()
        
        loss = self.calculate_kd_loss(y_pred_student, y_pred_teacher, y_pred_aug, labels)
        y = torch.ones(y_pred_teacher.shape[0]).cuda()
        dl = 10 * distance_loss(y_pred_aug, y_pred_student, y)
        loss += dl
        
        return loss

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_pred_aug, y_true):
        """
        Function used for calculating the KD loss during distillation
        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """

        soft_teacher_out = F.log_softmax(y_pred_teacher / self.temp, dim=1)
        soft_student_aug_out = F.log_softmax(y_pred_aug / self.temp, dim=1)
        soft_student_out = F.log_softmax(y_pred_student / self.temp, dim=1)
        ce_loss = SoftTargetCrossEntropy()
        #ce_loss = nn.CrossEntropyLoss()
        
        #kl_div = F.kl_div(F.log_softmax(y_pred_aug, dim=1),F.softmax(y_pred_teacher, dim=1),reduction='batchmean')
        #kl_div2 = F.kl_div(F.log_softmax(y_pred_aug, dim=1),F.softmax(y_pred_teacher, dim=1),reduction='batchmean')
        
        kl_div = 0.5 * self.temp * self.temp * self.loss_fn(soft_student_out, soft_teacher_out)
        kl_div2 = 0.5 * self.temp * self.temp * self.loss_fn(soft_student_aug_out, soft_teacher_out)
        loss = ce_loss(y_pred_student, y_true)
        loss += kl_div
        loss += kl_div2
        #loss += (0.25) * F.cross_entropy(y_pred_aug, y_true)
        #loss = (kl_div)
        #print(loss)
        #loss += (kl_div2)
        
        #loss = F.cross_entropy(y_pred_student, y_true)
            
        return loss