import torch.nn as nn
import torch
import torch.nn.functional as F


class customLoss(nn.Module):
    def __init__(self, N, lamd = 1e-1, mse_lamd=1, epoch=None):
        super(customLoss, self).__init__()
        self.N = N
        self.lamd = lamd
        self.mse_lamd = mse_lamd
        self.epoch = epoch
        return

    def forward(self, res, label):
        mse = F.mse_loss(res, label, reduction='sum')
        # mse = func.l1_loss(res, label, size_average=False)
        loss = mse / (self.N * 2)
        esp = 1e-12
        H = label.size()[2]
        W = label.size()[3]
        Itrue = label.clone()
        Ifake = res.clone()
        nom = torch.mul(Itrue, Ifake).sum(dim=1)
        denominator = Itrue.norm(p=2, dim=1, keepdim=True).clamp(min=esp) * \
                      Ifake.norm(p=2, dim=1, keepdim=True).clamp(min=esp)
        denominator = denominator.squeeze()
        # sam = -np.pi/2*torch.div(nom, denominator) + np.pi/2
        sam = torch.div(nom, denominator).acos()
        sam[sam!=sam] = 0
        sam_sum = torch.sum(sam) / (self.N * H * W)
        if self.epoch is None:
            total_loss = self.mse_lamd * loss + self.lamd * sam_sum
        else:
            norm = self.mse_lamd + self.lamd * 0.1 **(self.epoch//10)
            lamd_sam = self.lamd * 0.1 ** (self.epoch // 10)
            total_loss = self.mse_lamd/norm * loss + lamd_sam/norm * sam_sum
        return total_loss

class gradientLoss(nn.Module):
    def __init__(self, N, mse_lambda = 8e-1, gradient_lambda = 2e-1):
        super().__init__()
        self.mse_lambda = mse_lambda
        self.gradient_lambda = gradient_lambda
        self.N = N


    def forward(self, pred, gt, epoch):

        mse = F.mse_loss(pred, gt) / (self.N*2)

        pred_diff = torch.diff(pred, dim = 1)
        gt_diff = torch.diff(gt, dim = 1)

        pred_diff_flat = pred_diff.view(-1, pred_diff.shape[1])
        gt_diff_flat = gt_diff.view(-1, gt_diff.shape[1])

        cosine_sim = torch.cosine_similarity(pred_diff_flat, gt_diff_flat, dim = 1)
        slope_loss = 1-cosine_sim
        slope_loss = slope_loss.mean()

        if epoch == 0: 
            return self.mse_lambda*mse + self.gradient_lambda*slope_loss
        else:
            norm = self.mse_lambda + self.gradient_lambda * 0.1 **(epoch//10)
            lamd_slope = self.gradient_lambda * 0.1 ** (epoch // 10)
            total_loss = self.mse_lambda/norm * mse + lamd_slope/norm * slope_loss
            return total_loss
