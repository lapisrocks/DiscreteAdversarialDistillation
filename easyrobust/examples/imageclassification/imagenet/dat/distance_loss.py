import torch.nn as nn
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, tau=1.0, alpha=0.5, delta=1.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.delta = delta

    def forward(self, z_s, z_t, z_ta, z_a, target):
        ce_loss = SoftTargetCrossEntropy()
        kl_div = nn.KLDivLoss(reduction="batchmean", log_target=False)

        y_s = (z_s / self.tau).softmax(dim=1)
        y_t = (z_t / self.tau).softmax(dim=1)
        y_a = (z_a / self.tau).softmax(dim=1)
        y_al = (z_a / self.tau).log_softmax(dim=1)
        y_ta = (z_ta / self.tau).softmax(dim=1)
        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
        inter_loss_a = self.tau**2 * inter_class_relation(y_a, y_ta)
        intra_loss_a = self.tau**2 * intra_class_relation(y_a, y_ta)

        distance_loss = self.tau**2 * kl_div(y_al, y_s)
        kd_loss = self.alpha * (self.beta * inter_loss + self.gamma * intra_loss)
        kd_loss += self.alpha * (self.beta * inter_loss_a + self.gamma * intra_loss_a)
        class_loss = ce_loss(z_s, target)
        return kd_loss + (self.delta * class_loss) + distance_loss