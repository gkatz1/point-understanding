import torch
import torch.nn as nn
import torch.nn.functional as F
from .transforms import convert_labels_to_one_hot_encoding
from torch.autograd import Variable

# TODO: version of pytorch for cuda 7.5 doesn't have the latest features like
# reduce=False argument -- update cuda on the machine and update the code

# TODO: update the class to inherit the nn.Weighted loss with all the additional
# arguments

class FocalLoss(nn.Module):
    """Focal loss puts more weight on more complicated examples."""
   
    def __init__(self, gamma=1):
        
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, flatten_logits, flatten_targets):
        
        flatten_targets = flatten_targets.data
        
        number_of_classes = flatten_logits.size(1)
        
        flatten_targets_one_hot = convert_labels_to_one_hot_encoding(flatten_targets, number_of_classes)

        all_class_probabilities = F.softmax(flatten_logits)

        probabilities_of_target_classes = all_class_probabilities[flatten_targets_one_hot]

        elementwise_loss =  - (1 - probabilities_of_target_classes).pow(self.gamma) * torch.log(probabilities_of_target_classes)
        
        return elementwise_loss.sum()


class CrossEntropyLossElementwise(nn.Module):
    """
        This criterion (`CrossEntropyLoss`) combines `LogSoftMax` and `NLLLoss` in one single class.
        
        NOTE: Computes per-element losses for a mini-batch (instead of the average loss over the entire mini-batch).
        """
    log_softmax = nn.LogSoftmax()
    
    def __init__(self): #, class_weights):
        super(CrossEntropyLossElementwise, self).__init__()
        # self.class_weights = Variable(torch.FloatTensor(class_weights).cuda())
    
    def forward(self, logits, target):
        log_probabilities = self.log_softmax(logits)  # .cpu()
        target = target.view(-1, 1)
        # target = target.cpu()
        # NLLLoss(x, class) = -weights[class] * x[class]
        # return -self.class_weights.index_select(0, target) * log_probabilities.index_select(-1, target).diag()
        # print("[CrossEntropyLossElementwise] log_prob.size = {}".format(log_probabilities.size()))
        # step = 5000
        # print("log_prob size = {}".format(log_probabilities.size()[0]))
        # for i in xrange(0, log_probabilities.size()[0], step):
        #     cur_prob = log_probabilities[i:i+step, :]
        #     cur_tar = target[i:i+step]
        #     print("cur_tar size = {}".format(cur_tar.size()))
        #     cur = -cur_prob.index_select(-1, cur_tar)
        #     if i == 0:   # other option - initialize outside
        #         res = cur
        #     else:
        #         res = torch.cat((res, cur), 0)
        # res = Variable(res.cuda())
        # return res
        print("[CrossEntropyLossElementwise] log_prob.size = {}, target.size = {}".format(
            log_probabilities.size()[0], target.size()[0]))
 
        return -torch.gather(log_probabilities, 1, target)
        # return -log_probabilities.index_select(-1, target)
