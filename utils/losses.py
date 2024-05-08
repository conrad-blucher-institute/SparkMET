import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

device = torch.device("cuda:1")


class BCELossLogits(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logits, targets, reduction='mean'): 

        first = -targets.matmul(F.logsigmoid(logits))
        second = -(1 - targets).matmul(F.logsigmoid(logits) - logits)

        loss = first + second
        
        if reduction == 'mean':
            return loss / targets.shape[0]
        elif reduction == 'sum':
            return loss
        elif reduction == 'none':
            return first , second
        else:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

class MyBCELossLogits(nn.Module):
    def __init__(self, scale=8):
        super().__init__()
        self.scale = scale
 
    def forward(self, logits, targets, reduction='mean'):        
        
        hit, miss, fa, cr = 0, 0, 0, 0
        mask_ones = targets == 1
        mask_zeros = targets == 0

        fog_cases_targets = targets[mask_ones]
        fog_cases_logits  = logits[mask_ones]

        nofog_cases_targets = targets[mask_zeros]
        nofog_cases_logits  = logits[mask_zeros]

        for i in range(fog_cases_targets.shape[0]):  
            if torch.sigmoid(fog_cases_logits[i]) >= 0.5:
                hit += (- F.logsigmoid(fog_cases_logits[i])) 
            else:
                miss += (- F.logsigmoid(fog_cases_logits[i])) * self.scale

        for j in range(nofog_cases_targets.shape[0]):  
            if torch.sigmoid(nofog_cases_logits[j]) < 0.5:
                cr += - (F.logsigmoid(nofog_cases_logits[j]) - nofog_cases_logits[j])
            else:
                fa += (-(F.logsigmoid(nofog_cases_logits[j]) - nofog_cases_logits[j])) * self.scale
        
        loss = hit + miss + cr + fa
        
        if reduction == 'mean':
            return loss / targets.shape[0]
        elif reduction == 'sum':
            return loss
        elif reduction == 'none':
            return hit, miss, cr, fa
        else:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

class MyFocalLoss(nn.Module):
    def __init__(self, 
    alpha: float = -1,
    gamma: float = 2
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma 

    def forward(self, inputs: torch.Tensor,
                    targets: torch.Tensor,
                    reduction: str = "mean",
                                            ) -> torch.Tensor:
                                                
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        # loss = ce_loss * ((1 - p_t) ** self.gamma)
        
    
        alpha_miss   = torch.where(((1 - p_t) > 0.5) & (targets == 1), torch.tensor(self.gamma, device=device), torch.tensor(1, device=device))
        alpha_fa = torch.where(((1 - p_t) > 0.5) & (targets == 0), torch.tensor(self.gamma, device=device), torch.tensor(1, device=device))

        loss = alpha_fa * alpha_miss * ce_loss

        if reduction == "mean":
            loss_out = loss.mean()
        elif reduction == "sum":
            loss_out = loss.sum()

        return loss_out#, [inputs, targets, p, ce_loss, loss]
    
class SigmoidFocalLoss(nn.Module):
    def __init__(self, 
    alpha: float = -1,
    gamma: float = 2
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma 

    def forward(self, inputs: torch.Tensor,
                    targets: torch.Tensor,
                    reduction: str = "mean",
                                            ) -> torch.Tensor:
                                                
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if reduction == "mean":
            loss_out = loss.mean()
        elif reduction == "sum":
            loss_out = loss.sum()

        return loss_out#, [inputs, targets, p, ce_loss, loss]

def sigmoid(x):
  return (1/(1+np.exp(-x)))

class CDB_loss(nn.Module):
  
    def __init__(self, class_difficulty, tau='dynamic', reduction='none'):
        
        super(CDB_loss, self).__init__()
        self.class_difficulty = class_difficulty
        if tau == 'dynamic':
            bias = (1 - np.min(class_difficulty))/(1 - np.max(class_difficulty) + 0.01)
            tau = sigmoid(bias)
        else:
            tau = float(tau) 
        self.weights = self.class_difficulty ** tau
        self.weights = self.weights / self.weights.sum() * len(self.weights)
        self.reduction = reduction
        self.loss = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.weights), reduction=self.reduction).cuda()
        

    def forward(self, input, target):

        return self.loss(input, target)
    
class EQLloss(nn.Module):
    def __init__(self, freq_info):
        super(EQLloss, self).__init__()
        self.freq_info = freq_info
        # self.pred_class_logits = pred_class_logits
        # self.gt_classes = gt_classes
        self.lambda_ = 0.03
        self.gamma = 0.95
    def threshold_func(self):
        # class-level weight
        weight = self.pred_class_logits.new_zeros(self.n_c)
        weight[self.freq_info < self.lambda_] = 1
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
        return weight

    def forward(self, pred_class_logits, gt_classes,):
        self.pred_class_logits = pred_class_logits
        self.gt_classes = gt_classes
        self.n_i, self.n_c = self.pred_class_logits.size()

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c + 1)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target[:, :self.n_c]

        target = expand_label(self.pred_class_logits, self.gt_classes)
        if torch.rand(1).item() > self.gamma:
            coeff = torch.zeros(1)
        else:
            coeff = torch.ones(1)
        coeff = coeff.cuda()
        eql_w = 1 - (coeff * self.threshold_func() * (1 - target))

        cls_loss = F.binary_cross_entropy_with_logits(self.pred_class_logits, target,
                                                      reduction='none')

        return torch.sum(cls_loss * eql_w) / self.n_i

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction='none'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        else: return loss

