'''
authors: Gilad Katz & William Hinthorn
with many functions adapted from @warmspringwinds
'''
# import sys
import os
import shutil
import tqdm
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

# Must be after to use locally modified torchvision libs
import torchvision
import torchvision.transforms

# from .models import resnet_dilated
# from .models import partsnet
from .models import objpart_net
from .models import resnet_fcn
from .datasets.pascal_voc import PascalVOCSegmentation
from .transforms import (ComposeJoint,
                         RandomHorizontalFlipJoint,
                         RandomCropJoint)
from .losses import CrossEntropyLossElementwise
from .utils.pascal_part import get_valid_circle_indices


# pylint: disable=too-many-arguments, invalid-name, len-as-condition,
# pylint: disable=too-many-locals, too-many-branches,too-many-statements
# pylint: disable=no-member

# PATH = os.path.dirname(os.path.realpath(__file__))
# PATHARR = PATH.split(os.sep)
# home_dir = os.path.join(
#     '/', *PATHARR[:PATHARR.index('obj_part_segmentation') + 1])
# VISION_DIR = os.path.join(home_dir, 'vision')
# DATASET_DIR = os.path.join(home_dir, 'datasets')
# sys.path.insert(0, home_dir)
# sys.path.insert(0, VISION_DIR)


def poly_lr_scheduler(optimizer, init_lr, iteration, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iteration is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
        Credit @trypag
        https://discuss.pytorch.org/t/solved-learning-rate-decay/6825/5
    """
    if iteration % lr_decay_iter or iteration > max_iter:
        return optimizer

    lr = init_lr * (1 - iteration / max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def get_training_loaders(dataset_dir, network_dims, batch_size=8,
                         num_workers=4,
                         mask_type = {'train': "consensus", 'val': "consensus"},
                         which='binary',
                         validate_and_output_im=False):
    '''Returns loaders for the training set.
        args:
            :param ``network_dims``: ``dict`` which will
            store the output label splits
            :param ``dataset_dir``: str indicating the directory
            inwhich the Pascal VOC dataset is stored
            ... etc.
            :param ``mask_type``: dictionary (keys: train, val)
            :param ``which``: one of 'binary,', 'trinary', 'merged', or 'sparse'
            'binary': for each class: object or part
            'trinary': for each class: object, part or ambiguous
            'merged': for each class: object or one of k "super-parts"
            'sparse': for each calss: object or one of N parts
    '''

    print("[get_training_loaders], mask_type: {} => {}, {} => {}".format(
          'train', mask_type['train'], 'val', mask_type['val']))

    assert isinstance(network_dims, dict)
    insize = 512
    train_transform = ComposeJoint(
        [
            RandomHorizontalFlipJoint(),
            RandomCropJoint(crop_size=(insize, insize), pad_values=[
                0, 255, -2]),
            [torchvision.transforms.ToTensor(), None, None],
            [torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), None, None],
            # convert labels to tensors
            [None, torchvision.transforms.Lambda(
                lambda x: torch.from_numpy(np.asarray(x)).long()),
             # Point Labels
             torchvision.transforms.Lambda(
                 lambda x: torch.from_numpy(np.asarray(x)).long())]
        ])

    trainset = PascalVOCSegmentation(dataset_dir,
                                     network_dims=network_dims,
                                     download=False,
                                     joint_transform=train_transform,
                                     mask_type=mask_type['train'],
                                     which=which)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, drop_last=True)

    valid_transform = ComposeJoint(
        [
            [torchvision.transforms.ToTensor(), None, None],
            [torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), None, None],
            # convert labels to tensors
            [None, torchvision.transforms.Lambda(
                lambda x: torch.from_numpy(np.asarray(x)).long()),
             # Point Labels
             torchvision.transforms.Lambda(
                 lambda x: torch.from_numpy(np.asarray(x)).long())]
        ])

    valset = PascalVOCSegmentation(dataset_dir,
                                   network_dims={},
                                   train=False,
                                   download=False,
                                   joint_transform=valid_transform,
                                   mask_type=mask_type['val'],
                                   which=which)

    valset_loader = torch.utils.data.DataLoader(valset, batch_size=1,
                                                shuffle=False, num_workers=2)

    valset_for_output_im = None
    valset_for_output_im_loader = None
    if validate_and_output_im:
        valset_for_output_im = PascalVOCSegmentation(dataset_dir,
                                       network_dims={},
                                       train=False,
                                       download=False,
                                       joint_transform=valid_transform,
                                       mask_type=mask_type['val'],
                                       which=which,
                                       return_imid=True)

        valset_for_output_im_loader = torch.utils.data.DataLoader(valset_for_output_im, batch_size=1,
                                                    shuffle=False, num_workers=2)
   
        

    return (trainloader, trainset), (valset_loader, valset), (valset_for_output_im_loader, valset_for_output_im)


def flatten_logits(logits, number_of_classes):
    """Flattens the logits batch except for the logits dimension"""
    logits_permuted = logits.permute(0, 2, 3, 1)
    logits_permuted_cont = logits_permuted.contiguous()
    logits_flatten = logits_permuted_cont.view(-1, number_of_classes)
    return logits_flatten


def get_valid_logits(logits, index, number_of_classes):
    ''' processes predictions based on the valid indices (selected
    from annotations)
    '''
    if len(index) == 0:
        return torch.Tensor([])
    logits_flatten = flatten_logits(
        logits, number_of_classes=number_of_classes)
    return torch.index_select(logits_flatten, 0, index)


def flatten_annotations(annotations):
    '''Literally just remove dimensions of tensor.
    '''
    return annotations.view(-1)


def get_valid_annotations_index(flat_annos, mask_out_value=255):
    ''' Returns a tensor of indices of all nonzero values
    in a flat tensor.
    '''
    nonz = torch.nonzero((flat_annos != mask_out_value))
    if nonz.numel() == 0:
        return torch.LongTensor([])
    return torch.squeeze(nonz, 1)


def get_valid_annos(anno, mask_out_value):
    ''' selects labels not masked out
        returns a flattened tensor of annotations and the indices which are
        valid
    '''
    anno_flatten = flatten_annotations(anno)
    index = get_valid_annotations_index(
        anno_flatten, mask_out_value=mask_out_value)
    if index.numel() == 0:
        return index.clone(), index
    anno_flatten_valid = torch.index_select(anno_flatten, 0, index)
    return anno_flatten_valid, index


def numpyify_logits_and_annotations(logits, anno, flatten=True):
    ''' process logits and annotations for input into
    confusion matrix function.

    args::
        ``logits``: network predictoins
        ``anno``: ground truth annotations

    returns::
        flattened predictions, flattened annotations
    '''
    # First we do argmax on gpu and then transfer it to cpu
    _logits = logits.data
    _, prediction = _logits.max(1)
    prediction = prediction.squeeze(1)
    prediction_np = prediction.cpu().numpy()
    anno_np = anno.numpy()
    if flatten:
        return prediction_np.flatten(), anno_np.flatten()
    return prediction_np, anno_np


def outputs_tonp_gt(logits, anno, op_map, compression_method='max_of_all', semseg_logits=None, compress=False,
    flatten=True, force_binary=False, class_to_compress_by=None):
    ''' process logits and annotations for input into
    confusion matrix function.

    args::
        ``logits``: network predictoins
        ``anno``: ground truth annotations
        ``force_binary``: force 61-way to predict a label out of the 41-way objpart (unambiguous channels)
         ``compress``: boolean indicating whether to compress 41/61-way into 2/3-way
    returns::
        flattened predictions, flattened annotations
    '''
    # def to_pair(ind, num_to_aggregate):
    #     '''Get the indices for corresponding obj-part pairs.'''
    #     if ind > num_to_aggregate:
    #         return [ind - num_to_aggregate, ind]
    #     return [ind, ind + num_to_aggregate]
    def to_pair(label, op_map):
        '''Use gt labels to select the correct pair of
           indices'''
        other_label = op_map[label]
        # new_label, pair = to_pair(label, 20)
        if label < other_label:
            return [label, other_label]
        return [other_label, label]

    def to_triple(label, op_map):
        '''Use gt labels to select the correct pair of
           indices'''
        try:
            other_label1, other_label2 = op_map[label]
        except:
            print("[compress_objpart_logits] op_map = {}".format(op_map))
            raise

        # new_label, pair = to_pair(label, 20)
        if label < other_label1:   # label of object
            triple = [label, other_label1, other_label2]
        elif other_label1 < label and label < other_label2:  # label of part
            triple = [other_label1, label, other_label2]
        else:   # label of ambiguous
            triple = [other_label1, other_label2, label]

        return triple

    def get_channels_of_cls(cls):
        return [cls, cls + 20, cls + 40]

    # *** params ***
    compressed_object_val = 0
    compressed_part_val = 1
    compressed_ambiguous_val = 2
    object_max_ch, part_max_ch, ambiguous_max_ch = 20, 40, 60
    bg_pred_ctr = 0
    amb_pred_ctr = 0
    compress_annos = compress
    compress_predictions = compress

    _logits = logits.data.cpu()
    anno_np = anno.numpy()
    predictions = np.zeros_like(anno_np)
    if compression_method == 'use_semseg_prediction':
        _, semseg_predictions = torch.max(semseg_logits, dim=1)
        # semseg_predictions = semseg_predictions.cpu().data.numpy()
    num_classes = logits.size()[1]
    if num_classes == 41:
        force_binary = True

    print("[outputs_tonp_gt] num_classes = {}".format(num_classes))
    print("[outputs_tonp_gt] logits.size = {}".format(logits.size()))

    # for index, anno_ind in np.ndenumerate(anno_np):
    for index in zip(*np.where(np.logical_and(anno_np > 0, anno_np != 255))):
        anno_ind = anno_np[index]
        batch_ind = index[0]
        i = index[1]
        j = index[2]
        
        if compression_method == 'gt':
            if num_classes == 61:
                channel_indices = to_triple(anno_ind, op_map)  # num_to_aggregate)
            else:
                channel_indices = to_pair(anno_ind, op_map)  # num_to_aggregate)

            if force_binary:
                # print('*' * 50 + ' gt, force binary')
                # print(channel_indices, [ci for ci in channel_indices if ci <= part_max_ch])
                aided_prediction = channel_indices[np.argmax(
                    [_logits[batch_ind, ci, i, j] for ci in channel_indices if ci <= part_max_ch])]
            else:
                aided_prediction = channel_indices[np.argmax(
                    [_logits[batch_ind, ci, i, j] for ci in channel_indices])]

        elif compression_method == 'max_of_all':
            if force_binary:
                channel_indices = range(min(num_classes, part_max_ch + 1))
                print('*' * 50 + ' force_binary, channel_indices')
                print(channel_indices)
            else:
                channel_indices = range(num_classes)
            channel_indices.remove(0)

            # print("*" * 50 + "max of all")
            aided_prediction = channel_indices[np.argmax(
                [_logits[batch_ind, ci, i, j] for ci in channel_indices])]  # Not good, need to +1?
            # print(_logits[batch_ind, :, i, j], aided_prediction)
            # amax = np.argmax(_logits[batch_ind, :, i, j].numpy())
            # print(amax)
            # raise NotImplementedError()

        elif compression_method == 'use_semseg_prediction':
            # what if the prediction is background? use other method for this point
            semseg_ind = semseg_predictions[index].data[0]
            # print('&' * 30,'semseg_prediction', type(semseg_ind), semseg_ind)
            if semseg_ind == 0:
                with open("compression_bg_debug", 'w') as f:
                    f.write("{}".format(bg_pred_ctr))
                    bg_pred_ctr += 1
                    semseg_ind = anno_np[index]   # TMP

            if num_classes == 61:
                channel_indices = to_triple(semseg_ind, op_map)  # num_to_aggregate)
            else:
                channel_indices = to_pair(semseg_ind, op_map)  # num_to_aggregate)

            if force_binary:
                # print('*' * 50 + ' use_semseg_pred, force binary')
                # print(channel_indices, [ci for ci in channel_indices if ci <= part_max_ch])
                
                aided_prediction = channel_indices[np.argmax(
                    [_logits[batch_ind, ci, i, j] for ci in channel_indices if ci <= part_max_ch])]
            else:
                aided_prediction = channel_indices[np.argmax(
                   [_logits[batch_ind, ci, i, j] for ci in channel_indices])]

            
        elif compression_method == 'sum':
            compress = False
            compress_annos = True
            # sum all object channels, part channels, ambiguous channels (optional)
            ambiguous_score = -float('inf')

            channels = _logits[batch_ind, :, i, j].numpy()
            # print(channels.shape, channels)

            cum_sum = channels.cumsum()
            # print(cum_sum)

            object_score = cum_sum[object_max_ch]
            part_score = cum_sum[part_max_ch]
            if num_classes == 61 and not force_binary:
                ambiguous_score = cum_sum[ambiguous_max_ch]
                ambiguous_score = ambiguous_socre - part_score
            part_score = part_score - object_score
            object_score = object_score - channels[0]    # subtract bg

	    aided_prediction = np.argmax(np.array([object_score, part_score, ambiguous_score]))
            print("*" * 50 + " sum, aided pred = {}".format(aided_prediction))
            # print(object_score, part_score, aided_prediction)
            # raise NotImplementedError()

        elif compression_method == 'by_given_class':
            if not class_to_compress_by:
                raise TypeError("No class to compress by was specified")
            channel_indices = get_channels_of_cls(class_to_compress_by)
            print(class_to_compress_by, channel_indices)

            if force_binary:
                aided_prediction = channel_indices[np.argmax(
                    [_logits[batch_ind, ci, i, j] for ci in channel_indices if ci <= part_max_ch])]
            else:
                aided_prediction = channel_indices[np.argmax(
                    [_logits[batch_ind, ci, i, j] for ci in channel_indices])]
        else:
            raise NotImplementedError("{} compression method is not implemented".format(
                compression_method))

        if compress_predictions: 
            predictions[batch_ind, i, j] = compressed_object_val if aided_prediction <= object_max_ch \
                else compressed_part_val if aided_prediction <= part_max_ch \
                    else compressed_ambiguous_val
            # print("*" * 50 + " compress, pred = {}".format(predictions[batch_ind, i, j])) 
        else:
            predictions[batch_ind, i, j] = aided_prediction
            if aided_prediction > part_max_ch:
                with open('ambiguous_prediction_debug', 'w') as f:
                    f.write('{}'.format(amb_pred_ctr))
                    amb_pred_ctr += 1

    # outside of the loop - compress anno_np
    if compress_annos:
        anno_np[np.logical_and(anno_np > 0, anno_np <= object_max_ch)] = compressed_object_val
        anno_np[np.logical_and(anno_np > object_max_ch, anno_np <= part_max_ch)] = compressed_part_val
        anno_np[np.logical_and(anno_np > part_max_ch, anno_np <= ambiguous_max_ch)] = compressed_ambiguous_val
        print('*' * 50 + 'compress anno, truth = {}'.format(np.any(anno_np == 2)))
    if flatten:
        return predictions.flatten(), anno_np.flatten()
    return predictions, anno_np


def outputs_tonp_gt_61_way(logits, anno, op_map, flatten=True):
    ''' process logits and annotations for input into
    confusion matrix function.

    args::
        ``logits``: network predictoins
        ``anno``: ground truth annotations

    returns::
        flattened predictions, flattened annotations
    '''
    def to_triple(label, op_map):
        '''Use gt labels to select the correct pair of
           indices'''
        try:
            other_label1, other_label2 = op_map[label]
        except:
            print("[compress_objpart_logits] op_map = {}".format(op_map))
            raise

        # new_label, pair = to_pair(label, 20)
        if label < other_label1:   # label of object
            triple = [label, other_label1, other_label2]
        elif other_label1 < label and label < other_label2:  # label of part
            triple = [other_label1, label, other_label2]
        else:   # label of ambiguous
            triple = [other_label1, other_label2, label]

        return triple
    
    _logits = logits.data.cpu()
    anno_np = anno.numpy()
    predictions = np.zeros_like(anno_np)
    # for index, anno_ind in np.ndenumerate(anno_np):
    for index in zip(*np.where(np.logical_and(anno_np > 0, anno_np != 255))):
        anno_ind = anno_np[index]
        batch_ind = index[0]
        i = index[1]
        j = index[2]
        channel_indices = to_triple(anno_ind, op_map)  # num_to_aggregate)
        aided_prediction = channel_indices[np.argmax(
            [_logits[batch_ind, ci, i, j] for ci in channel_indices])]

        predictions[batch_ind, i, j] = aided_prediction
    if flatten:
        return predictions.flatten(), anno_np.flatten()
    return predictions, anno_np


# check
def compress_objpart_logits(logits, anno, op_map, method="gt", semseg_logits=None):
    ''' Reduce N x 41 tensor ``logits`` to an N x 2 tensor ``compressed``,
        where ``compressed``[0] => "object" and ``compressed``[1] => part
        (generic).
    args::
        ``logits``: network predictions => 2D tensor of shape (N, 41)
        ``anno``: ground truth annotations => 1D tensor of length N

    returns::
        compressed tensor of shape (N, 2)
    '''
    if method == 'none' or method == 'max_of_all':
        return logits, anno

    anno = anno.data.cpu()

    def to_pair(label, op_map):
        '''Use gt labels to isloate op loss
        '''
        try:
            other_label = op_map[label]
        except:
            print("[compress_objpart_logits] op_map = {}".format(op_map))
            raise

        # new_label, pair = to_pair(label, 20)
        if label < other_label:
            pair = [label, other_label]
            new_label = 0
        else:
            pair = [other_label, label]
            new_label = 1
        return new_label, pair

    def to_triple(label, op_map):
        '''Use gt labels to isloate op loss
        '''
        try:
            other_label1, other_label2 = op_map[label]
        except:
            print("[compress_objpart_logits] op_map = {}".format(op_map))
            raise

        # new_label, pair = to_pair(label, 20)
        if label < other_label1:   # label of object
            triple = [label, other_label1, other_label2]
            new_label = 0
        elif other_label1 < label and label < other_label2:  # label of part
            triple = [other_label1, label, other_label2]
            new_label = 1
        else:   # label of ambiguous
            triple = [other_label1, other_label2, label]
            new_label = 2 

        return new_label, triple

    print("^" * 50 + "compress_logits, method = {}".format(method))

    num_classes = logits.size()[-1]
    indices = []
    new_anno = []
    for _, label in enumerate(anno):
        # new_label, pair = to_pair(label, 20)
        if num_classes == 61:
            new_label, triple = to_triple(label, op_map)
            indices.append(triple)
        else:
            new_label, pair = to_pair(label, op_map)
            indices.append(pair)
        new_anno.append(new_label)
    len_ = len(indices)
    new_anno = Variable(torch.LongTensor(new_anno).cuda())
    indices = Variable(torch.LongTensor(indices).cuda())
    if len_ == 0:
        compressed_logits = Variable(torch.Tensor([]).cuda())
    else:
        compressed_logits = torch.gather(logits, 1, indices)
    
    if method == "sum":
        indices = None
        compressed_logits = None
        if num_classes == 41:
            object_max_label = 20   # inclusive
            part_max_label = 40
            indices = Variable(torch.LongTensor([object_max_label, part_max_label]).expand(
                logits.size()[0], 2).cuda())
            csum = torch.cumsum(logits, dim=1)
            compressed_logits = torch.gather(csum, 1, indices)
            # tor_ar = torch.arange(0,compressed_logits.size()[0]).long().cuda()
            # comp_cloned = compressed_logits[:, 0].clone()
            compressed_logits[:, 1] = compressed_logits[:, 1] - compressed_logits[:, 0] #comp_cloned #compressed_logits[:, 0].clone() #.contiguous()
            compressed_logits[:, 0] = compressed_logits[:, 0] - logits[:, 0]#.contiguous()   # subtract bg

        elif num_classes == 61:
            object_max_label = 20   # inclusive
            part_max_label = 40
            ambiguous_max_label = 60
            indices = Variable(torch.LongTensor([object_max_label, part_max_label, 
                ambiguous_max_label]).expand(logits.size()[0], 3).cuda())
            csum = torch.cumsum(logits, dim=1)
            compressed_logits = torch.gather(csum, 1, indices)
            compressed_logits[:, 2] = compressed_logits[:, 2] - compressed_logits[:, 1]
            compressed_logits[:, 1] = compressed_logits[:, 1] - compressed_logits[:, 0]
            compressed_logits[:, 0] = compressed_logits[:, 0] - logits[:, 0]   # subtract bg

    elif method == "use_semseg_prediction":
        if logits.size()[0] != semseg_logits.size()[0]:
            with open('compress_logits_sizes_debug', 'a+') as f:
                f.write("logits size = {}, semseg_pred size = {}\n".format(logits.size()[0],
                    semseg_logits.size()[0]))
            return compressed_logits, new_anno   # for now - use gt instead of prediction
 
        indices = []
        compressed_logits = None
        # use semsg_logits
        # extract the prediction
        # use the prediction as the label
        _, semseg_predictions = torch.max(semseg_logits, dim=1)    # what about invalid logits? take the gt?
        ctr = 0
        for idx, label in enumerate(semseg_predictions):
            if label == 0:
                ctr += 1
                with open('bg_label_debug', 'wb') as f:
                    f.write("{}".format(ctr))
                label = anno[idx]
            # new_label, pair = to_pair(label, 20)
            if num_classes == 61:
                _, triple = to_triple(label, op_map)
                indices.append(triple)
            else:
                _, pair = to_pair(label, op_map)
                indices.append(pair)
        len_ = len(indices)
        indices = Variable(torch.LongTensor(indices).cuda())
        if len_ == 0:
            compressed_logits = Variable(torch.Tensor([]).cuda())
        else:
            print("indices_size = {}, logits_size = {}".format(logits.size(), indices.size()))
            compressed_logits = torch.gather(logits, 1, indices)

    elif method == "gt":
        pass

    return compressed_logits, new_anno


def compress_objpart_logits_61_way(logits, anno, op_map):
    """ Reduce N x 61 tensor ``logits`` to an N x 3 tensor ``compressed``,
        where ``compressed``[0] => "object" and ``compressed``[1] => part
        ``compressed``[2] => ambiguous (generic).
    args::
        ``logits``: network predictions => 2D tensor of shape (N, 61)
        ``anno``: ground truth annotations => 1D tensor of length N

    returns::
        compressed tensor of shape (N, 2)
    """
    print("[compress_objpart_logits_61_way] op_map = {}".format(op_map))

    anno = anno.data.cpu()

    def to_triple(label, op_map):
        '''Use gt labels to isloate op loss
        '''
        try:
            other_label1, other_label2 = op_map[label]
        except:
            print("[compress_objpart_logits] op_map = {}".format(op_map))
            raise

        # new_label, pair = to_pair(label, 20)
        if label < other_label1:   # label of object
            triple = [label, other_label1, other_label2]
            new_label = 0
        elif other_label1 < label and label < other_label2:  # label of part
            triple = [other_label1, label, other_label2]
            new_label = 1
        else:   # label of ambiguous
            triple = [other_label1, other_label2, label]
            new_label = 2 

        return new_label, triple
    
    indices = []
    new_anno = []
    for _, label in enumerate(anno):
        # new_label, pair = to_pair(label, 20)
        new_label, triple = to_triple(label, op_map)
        indices.append(triple)
        new_anno.append(new_label)
    len_ = len(indices)
    new_anno = Variable(torch.LongTensor(new_anno).cuda())
    indices = Variable(torch.LongTensor(indices).cuda())
    if len_ == 0:
        compressed_logits = Variable(torch.Tensor([]).cuda())
    else:
        compressed_logits = torch.gather(logits, 1, indices)
    return compressed_logits, new_anno


def get_iou(conf_mat):
    '''

    Used for computing the intersection over union metric
    using a confusion matrix. Pads unseen labels (union)
    with epsilon to avoid nan.
    Returns a vector of length |labels| with
    the IoU for each class in its appropriate
    place.

    '''
    intersection = np.diag(conf_mat)
    gt_set = conf_mat.sum(axis=1)
    predicted_set = conf_mat.sum(axis=0)
    union = gt_set + predicted_set - intersection
    # Ensure no divide by 1 errors
    eps = 1  # 1e-5
    union[union == 0] = eps
    iou = intersection / union.astype(np.float32)
    return iou


def get_precision_recall(conf_mat):
    ''' Returns the class-wise precision and recall given a
        confusion matrix.
        Note that this defaults to 0 to avoids divide by zero errors.
    '''
    intersection = np.diag(conf_mat)
    gt_set = conf_mat.sum(axis=1)
    predicted_set = conf_mat.sum(axis=0)
    precision = intersection / \
        np.array([np.max([pred, 1.0]) for pred in predicted_set]).astype(
            np.float32)
    recall = intersection / \
        np.array([np.max([gt, 1.0]) for gt in gt_set]).astype(np.float32)
    return precision, recall


def validate_batch(
        objpart_dat,
        semantic_dat,
        overall_part_confusion_matrix,
        overall_semantic_confusion_matrix,
        labels,
        merge_level,
        op_map,
        writer=None,
        index=0):
    ''' Computes the running IoU for the semantic and object-part tasks.
        args::
            :param (objpart_logits, objpart_anno): prediction, ground_truth
                    tensors for the object-part inference task
            :param (semantic_logits, semantic_anno): ditto for the semantic
            segmentation task
            :param overal_semantic_confusion_matrix: None or tensor of length
                                    |segmentation classes|. Total confusion
                                    matrix for semantic segmentation task
                                    for this epoch.
            :param overal_part_confusion_matrix: None or tensor of length
                                    |segmentation classes|. Total confusion
                                    matrix for object-part inference task
                                    for this epoch.

    '''
    (objpart_logits, objpart_anno) = objpart_dat
    (semantic_logits, semantic_anno) = semantic_dat
    objpart_labels, semantic_labels = labels
    semantic_prediction_np, semantic_anno_np = numpyify_logits_and_annotations(
        semantic_logits, semantic_anno)
    # objpart_prediction_np, objpart_anno_np = numpyify_logits_and_annotations(
    #     objpart_logits, objpart_anno)
        
    if merge_level == 'binary' or merge_level == 'trinary':     # GILAD
        objpart_prediction_np, objpart_anno_np = numpyify_logits_and_annotations(
                objpart_logits, objpart_anno)
        no_parts = []
    elif merge_level == '61-way':
        objpart_prediction_np, objpart_anno_np = outputs_tonp_gt_61_way(
                objpart_logits, objpart_anno, op_map)
        no_parts = [0, 4, 9, 11, 18, 20, 24, 29, 31, 38, 40]
        # Make sure to ignore all background class values
        objpart_anno_np[objpart_anno_np == 0] = -2
        # objpart_prediction_np[objpart_anno_np == 0] = -1
    else:
        objpart_prediction_np, objpart_anno_np = outputs_tonp_gt(
                objpart_logits, objpart_anno, op_map)
        no_parts = [0, 4, 9, 11, 18, 20, 24, 29, 31, 38, 40]
        # Make sure to ignore all background class values
        objpart_anno_np[objpart_anno_np == 0] = -2
        # objpart_prediction_np[objpart_anno_np == 0] = -1

    # Mask-out value is ignored by default in the sklearn
    # read sources to see how that was handled
    # import pdb
    # pdb.set_trace()
    current_semantic_confusion_matrix = confusion_matrix(
        y_true=semantic_anno_np,
        y_pred=semantic_prediction_np,
        labels=semantic_labels)

    if overall_semantic_confusion_matrix is None:
        overall_semantic_confusion_matrix = current_semantic_confusion_matrix
    else:
        overall_semantic_confusion_matrix += current_semantic_confusion_matrix
    try:
        current_objpart_confusion_matrix = confusion_matrix(
            y_true=objpart_anno_np, y_pred=objpart_prediction_np,
            labels=objpart_labels)

        if overall_part_confusion_matrix is None:
            overall_part_confusion_matrix = current_objpart_confusion_matrix
        else:
            overall_part_confusion_matrix += current_objpart_confusion_matrix

        objpart_prec, objpart_rec = get_precision_recall(
            current_objpart_confusion_matrix)
        objpart_mPrec = np.mean(
            [prec for i, prec in enumerate(objpart_prec) if i not in no_parts])
        objpart_mRec = np.mean(
            [rec for i, rec in enumerate(objpart_rec) if i not in no_parts])

    except ValueError:
        current_objpart_confusion_matrix = None
        objpart_prec, objpart_rec = None, None
        objpart_mPrec, objpart_mRec = None, None

    semantic_IoU = get_iou(
        current_semantic_confusion_matrix)
    semantic_mIoU = np.mean(semantic_IoU)

    if writer is not None:
        # writer.add_scalar('data/objpart_mIoU', objpart_mIoU, index)
        writer.add_scalar('data/semantic_mIoU', semantic_mIoU, index)
        writer.add_scalars('data/semantic_IoUs',
                           {'cls ' + str(i): v for i,
                            v in enumerate(semantic_IoU)},
                           index)
        if objpart_mPrec is not None:
            writer.add_scalar('data/objpart_mPrec', objpart_mPrec, index)
        if objpart_mRec is not None:
            writer.add_scalar('data/objpart_mRec', objpart_mRec, index)
        if objpart_prec is not None:
            writer.add_scalars('data/part_prec',
                               {'cls ' + str(i): v for i,
                                v in enumerate(objpart_prec)},
                               index)
        if objpart_rec is not None:
            writer.add_scalars('data/part_rec',
                               {'cls ' + str(i): v for i,
                                v in enumerate(objpart_rec)},
                               index)

    return ((objpart_mPrec, objpart_mRec),
            semantic_mIoU, overall_part_confusion_matrix,
            overall_semantic_confusion_matrix)
    # return objpart_mIoU, semantic_mIoU, overall_part_confusion_matrix,
    # overall_semantic_confusion_matrix


def save_checkpoint(state, is_best, folder='models',
                    filename='checkpoint.pth.tar'):
    ''' Saves a model
        args::
            :param ``staet``: dictionary containing training data.
            :param ``is_best``: boolean determining if this represents
                            the best-trained model of this session
            :param ``folder``: relative path to folder in which to save
            checkpoint
            :param ``filename``: name of the checkpoint file

        additionally copies to "[architecture]" + "_model_best.pth.tar"
        if is_best.
    '''
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(
                folder, filename), os.path.join(
                    folder, state['arch'] + '_model_best.pth.tar'))


def load_checkpoint(load_path, fcn, optimizer):
    ''' Loads network parameters (and optimizer params) from a checkpoint file.
        args::
            :param ``load_path``: string path to checkpoint file.
            :param ``fcn``: torch.nn network
            :param ``optimizer``: duh
        returns the starting epoch and best scores
    '''
    if os.path.isfile(load_path):
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)

        try:
            start_epoch = checkpoint['epoch']
        except KeyError:
            start_epoch = 0
        try:
            best_semantic_val_score = checkpoint['best_semantic_mIoU']
            best_objpart_val_score = checkpoint['best_objpart_mIoU']
        except KeyError:
            best_semantic_val_score = 0.0
            best_objpart_val_score = 0.0
        try:
            best_objpart_accuracy = checkpoint['best_objpart_accuracy']
        except KeyError:
            best_objpart_accuracy = 0.0

        state_dict = {}
        model_sd = fcn.state_dict()

        if 'state_dict' in checkpoint:
            it = checkpoint['state_dict'].items()
        else:
            it = checkpoint.items()

        # import pdb; pdb.set_trace()
        # for k, v in checkpoint['state_dict'].items():
        for i, (k, v) in enumerate(it):
            k_ = k.split(".")
            if k_[0] == 'resnet34_8s':
                k_[0] = 'net'
            elif 'layer' in k_[0]:
                k_.insert(0, 'net')
            k_ = ".".join(k_)
            if k_ not in model_sd:
                print("Layer {} from checkpoint not found in model".format(k_))
                continue
            elif model_sd[k_].size() != v.size():
                print(
                    "{}: {} not equal to {}".format(
                        i,
                        model_sd[k_].size(),
                        v.size()))
                continue
            else:
                state_dict[k_] = v
        # optim_state_dict[k_] = checkpoint['optimizer'][k]
        # fcn.load_state_dict(checkpoint['state_dict'])

        fcn.load_state_dict(state_dict) # , strict=False)
        if optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except ValueError as e:
                print("{}".format(e))
            except KeyError:
                print("optimizer not found in checkpoint")
        _epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(load_path, _epoch))
    else:
        raise RuntimeError("{} does not exist.".format(load_path))

    return (start_epoch, best_semantic_val_score,
            best_objpart_val_score, best_objpart_accuracy)


def get_network_and_optimizer(
        arch,
        network_dims,
        num_classes_objpart, 
        loss_type,
        load_from_local=False,
        model_path=None,
        train_params=None,
        init_lr=0.0001):
    ''' Gets the network and corresponding optimizer.
        args::
            # of semantic segmentaton classes (final output)
            :param ``number_of_classes``:
            :param ``to_aggregate``: # of classes which have corresponding
            object and part channels.
            :param ``load_from_local``: boolean variable to determine whether
            to load parameters from a local checkpoint file
            :param ``model_path``: required if ``load_from_local`` is ``True``,
            String path to checkpoint file

        returns a net, optimizer, (criteria), and best scores

        TODO: make this flexible. This is really sloppy.

    '''
    print("[#] [evaluation.py] num_classes_objpart = {}".format(num_classes_objpart))
    # if arch == 'resnet34_8s':
    #     fcn = resnet_test.Resnet34_8s(objpart_num_classes=num_classes_objpart) 
    fcn = objpart_net.OPSegNet(
        arch=arch,
        output_dims=network_dims,
        num_classes_objpart=num_classes_objpart,
        pretrained=True)
    # fcn = resnet_dilated.Resnet34_8s(
    #     output_dims=network_dims,
    #     part_task=True)

    # if arch == 'resnet34_8s_2b':
    #     fcn = resnet_fcn.Resnet34_8s_2b(objpart_num_classes=num_classes_objpart)
    #     output_dims = {}
    #     print("Taking resnet34_8s_2b debug lane")
 
    optimizer = optim.Adam(fcn.parameters(), lr=init_lr, weight_decay=0.0001)
    if load_from_local:
        (start_epoch, best_semantic_val_score,
         best_objpart_val_score, best_objpart_accuracy) = load_checkpoint(
             model_path, fcn, optimizer)
    else:
        start_epoch = 0
        best_semantic_val_score, best_objpart_val_score = 0.0, 0.0
        best_objpart_accuracy = float("inf")	# GILAD
    fcn.cuda()
    fcn.train()
    
    if loss_type == '1_loss':
        semantic_criterion = CrossEntropyLossElementwise().cuda()   # CHECK
        objpart_criterion = CrossEntropyLossElementwise().cuda()   # CHECK
    else:
        semantic_criterion = nn.CrossEntropyLoss(size_average=False).cuda()
        objpart_criterion = nn.CrossEntropyLoss(size_average=False).cuda()

    _to_update = {
        'semantic_criterion': semantic_criterion,
        'objpart_criterion': objpart_criterion,
        'best_objpart_val_score': best_objpart_val_score,
        'best_semantic_val_score': best_semantic_val_score,
        'best_objpart_accuracy': best_objpart_accuracy,
        'start_epoch': start_epoch
    }
    train_params.update(_to_update)

    # optimizer = optim.Adam(fcn.parameters(), lr=0.0001, weight_decay=0.0001)
    return fcn, optimizer


def get_cmap():
    ''' Return a colormap stored on disk
    '''
    fname = os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)),
        'colortable.json')
    with open(fname, 'r') as f:
        cmap = json.load(f)
    return cmap


def validate_and_output_images(net, loader, op_map,
                               which='semantic', alpha=0.6, writer=None,
                               step_num=0, base_path='prediction/', save_name='output_images'):
    ''' Computes mIoU for``net`` over the a set.
        args:: :param ``net``: network (in this case resnet34_8s_dilated
            :param ``loader``: dataloader (in this case, validation set loader)

        returns the mIoU (ignoring classes where no parts were annotated) for
        the semantic segmentation task and object-part inference task.
      0: "background",
      1: "aeroplane",
      2: "bicycle",
      3: "bird",
      4: "boat",
      5: "bottle",
      6: "bus",
      7: "car",
      8: "cat",
      9: "chair",
      10: "cow",
      11: "diningtable",
      12: "dog",
      13: "horse",
      14: "motorbike",
      15: "person",
      16: "pottedplant",
      17: "sheep",
      18: "sofa",
      19: "train",
      20: "tvmonitor"
      21: "aeroplane_part",
      22: "bicycle_part",
      23: "bird_part",
      #24: "boat_part",
      24 - 25: "bottle_part",
      25 - 26: "bus_part",
      26 - 27: "car_part",
      27 - 28: "cat_part",
      29: "chair_part",
      28 - 30: "cow_part",
      31: "diningtable_part",
      29 - 32: "dog_part",
      30 - 33: "horse_part",
      31 - 34: "motorbike_part",
      32 - 35: "person_part",
      33 - 36: "pottedplant_part",
      34 - 37: "sheep_part",
      38: "sofa_part",
      35 - 39: "train_part",
      40: "tvmonitor_part"

    '''
    _MASK_OUT_VAL = -2
    point_small_region_val = 99
    point_small_region_color = [0, 0, 0]
    def save_visualization(image, prediction, im_idx, imid):
        i = im_idx  
        image_copy = image.numpy().squeeze(0).transpose(1, 2, 0)
        image_copy = image_copy.astype(np.float32)
        image_copy -= image_copy.min()
        image_copy /= image_copy.max()
        # image_copy*=255
        print("prediction in save_vis: ")
        print( prediction)
        prediction = prediction.squeeze()
        # prediction = prediction.reshape(prediction.size[1:])
        cmask = np.zeros_like(image_copy, dtype=np.float32)
        classes = np.unique(prediction)
        # sz = prediction.size
        for cls in classes:
            if cls <= 0:
                continue
            ind = prediction == cls
            cmask[ind, :] = cmap[cls]
            if cls == point_small_region_val:
                cmask[ind, :] = cmask[ind, :] * (float(alpha)) + \
                    cmask[ind, :] * point_small_region_color * (1 - alpha)
            print("cls = {}, cmap[cls] = {}".format(cls, cmap[cls]))
         
        cmask = cmask.astype(np.float32) / cmask.max()
        ind = prediction > 0
        image_copy[ind] = image_copy[ind] * \
            (1.0 - alpha) + cmask[ind] * (float(alpha))  
        print("shapes") 
        print(image_copy.shape, cmask.shape, image_copy[ind].shape, cmask[ind].shape, ind.shape)
#         ind_larger_region = prediction == point_small_region_val
#         image_copy[ind_larger_region] = image_copy[ind] * \
#             (1.0 - alpha) + np * (float(alpha))    # multiply by larger_region_color
        image_copy = image_copy - image_copy.min()
        image_copy = image_copy / np.max(image_copy)
        image_copy = image_copy * 255
        image_copy = image_copy.astype(np.uint8)
        print("image_copy.shape = {}".format(image_copy.shape))
        image_copy_torch_tensor = image_copy.transpose(2, 0, 1)
        image_copy_torch_tensor = torch.from_numpy(image_copy)
       
        if writer is not None:
            writer.add_image('images/image_{}_{}'.format(i, imid), image_copy, step_num)
        
        image_copy = Image.fromarray(image_copy)
        if not os.path.isdir("{}{}/".format(base_path, save_name)):
            os.makedirs("{}{}/".format(base_path, save_name))

        image_copy.save("{}{}/validation_{}_{}.png".format(base_path, save_name, which, i))
        image_copy.close()
        # hxwx(rgb)

    if which is None or which == 'None':
        which = 'semantic'

    from PIL import Image
    net.eval()
    # hardcoded in for the object-part infernce
    # no_parts = [0, 4, 9, 11, 18, 20, 24, 29, 31, 38, 40]
    # objpart_labels, semantic_labels = labels
    cmap = get_cmap()
    prediction_dict = dict()
    save_images_with_pt_anno_only = True
    paint_points_only = True
    paint_points_and_region = True
    annotated_pts_indices = []
    dump_semantic_pred = True

    i = 0
    for image, semantic_anno, objpart_anno, imid in tqdm.tqdm(loader):
        imid = imid[0]
        print("imid = {}".format(imid))
        img = Variable(image.cuda())
        objpart_logits, semantic_logits = net(img)
        num_classes = objpart_logits.size()[1]

        # First we do argmax on gpu and then transfer it to cpu
        if which == 'semantic':
            prediction, _ = numpyify_logits_and_annotations(
                semantic_logits, semantic_anno, flatten=False)
        elif which == 'separated':
            prediction, _ = numpyify_logits_and_annotations(
                objpart_logits, objpart_anno, flatten=False)
        elif which == 'separated_use_gt':
            prediction, _ = outputs_tonp_gt(
                objpart_logits, semantic_anno, op_map, flatten=False)
        elif which == 'objpart':
            # prediction, anno = outputs_tonp_gt(
            #     objpart_logits, objpart_anno,semantic_anno, flatten=False)
            prediction, _ = outputs_tonp_gt(
                objpart_logits, semantic_anno, op_map, flatten=False)
            prediction[np.logical_and(
                prediction > 0, prediction < 21)] = 1  # object
            prediction[np.logical_and(
                prediction > 20, prediction < 41)] = 2  # part
            if num_classes == 61:
                prediction[np.logical_and(
                    prediction > 40, prediction < 61)] = 3  # ambiguous  

        else:
            raise ValueError(
                '"which" value of {} not valid. Must be one of "semantic",'
                '"separated", or'
                '"objpart"'.format(which))

        if dump_semantic_pred:
            semantic_pred, _ = numpyify_logits_and_annotations(
                semantic_logits, semantic_anno, flatten=False)

        print("[evaluation.py, 704] squeezed image shape = {}".format(
                image.numpy().squeeze(0).shape))

        print("objpart_logits.size = {}".format(objpart_logits.size()))
        print("objpart_anno.size = {}".format(objpart_anno.size()))
        print("objpart_anno not -2 = {}".format(torch.nonzero(objpart_anno != -2)))
        print("prediction.size = {}".format(prediction.shape))
        # raise NotImplementedError()
        
        # if points_only:
        if paint_points_only or paint_points_and_region:
            r_small = 2
            r_large = 7
            # mask out prediction to contain only points with annotation - mask out the rest
            # also paint a little circle around the pixel - otherwise can't see it
            objpart_anno = objpart_anno.squeeze()
            prediction = prediction.squeeze()
            annotated_pts = torch.nonzero(objpart_anno != _MASK_OUT_VAL)
            annotated_pts_indices = annotated_pts.numpy()
            not_annotated = torch.nonzero(objpart_anno == _MASK_OUT_VAL)
            not_annotated_np = not_annotated.numpy()
            rows, cols = zip(*not_annotated_np)
            if paint_points_and_region and annotated_pts_indices.size != 0:
                # rows, col <- not_annotated & not_in_region
                region_rows = []
                region_cols = []
                for pt in annotated_pts:
                    cur_region_rows, cur_region_cols = get_valid_circle_indices(
                        prediction, (pt[0], pt[1]), r_large) # circle with larger raduis
                    region_rows += cur_region_rows
                    region_cols += cur_region_cols
                    print("region_rows, region_cols")
                    print(len(cur_region_rows), len(cur_region_cols))
                print(len(rows), len(cols))
                a1 = np.array(zip(rows, cols))
                a2 = np.array(zip(region_rows, region_cols))
                print(region_rows, region_cols)
                print(annotated_pts_indices.size == 0)
                print("shapes ", a1.shape, a2.shape)
                a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
                a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
                rows, cols = zip(*np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1]))
                # rows, cols = np.setdiff1d(zip(rows,cols), zip(region_rows,region_cols)).squeeze()
                # rows = np.setdiff1d(rows, region_rows).squeeze()
                # cols = np.setdiff1d(cols, region_cols).squeeze()
            prediction[rows, cols] = _MASK_OUT_VAL  # visualize only annotated points
            # ch, rows, cols = zip(*np.where(prediction != _MASK_OUT_VAL))
            print(zip(*np.where(prediction != _MASK_OUT_VAL)))
            print("annotated_pts_indices")
            # print(annotated_pts_indices)
            print(type(annotated_pts_indices))

            pt_dict = dict()
            for pt in annotated_pts_indices:
                # "x_y": [gt_value, predicted_value]
                if dump_semantic_pred:
                    semantic_anno, semantic_pred = semantic_anno.squeeze(), semantic_pred.squeeze()
                    print("debug, dump_semantic_pred, shapes:")
                    print(objpart_anno.size(), prediction.shape, semantic_anno.size(), semantic_pred.shape)
                    pt_dict["{}_{}".format(pt[1], pt[0])] = [objpart_anno[pt[0], pt[1]],
                        prediction[pt[0], pt[1]], semantic_anno[pt[0], pt[1]], semantic_pred[pt[0], pt[1]]]  # x,y
                else:
                    pt_dict["{}_{}".format(pt[1], pt[0])] = [objpart_anno[pt[0], pt[1]],
                        prediction[pt[0], pt[1]]]  # x,y

                # paint circle with same value, for a clear view of the point in visualizationi
                indices = get_valid_circle_indices(prediction, (pt[0], pt[1]), r_small)
                print("=== indices ===")
                print(len(indices[0]))
                print(type(indices), type(indices[0]))
                # print(indices)
                if not paint_points_and_region:
                    prediction[indices[0], indices[1]] = prediction[pt[0], pt[1]]

                    print(prediction[indices[0], indices[1]])

            if annotated_pts_indices != []:
                prediction_dict[imid] = pt_dict.copy() 

 
            # now for each annotated point - paint a circle around it
        if i == 0:
            # output legend - 1 image for each class color
            for cls in range(num_classes):
                pred_copy = np.zeros_like(prediction, dtype=np.uint8)
                pred_copy.fill(cls)
                max_val = torch.max(image)
                try:
                    max_val = max_val[0]
                except:
                    pass
                # white_image = torch.Tensor([max_val]).expand(image.size())
                # white_image[0,0,0] = 0
                save_visualization(image, pred_copy, 'class_{}'.format(cls), '')
        
        if save_images_with_pt_anno_only:   # only images with point annotation
            if annotated_pts_indices != []:
                save_visualization(image, prediction, i, imid)
        else:
            save_visualization(image, prediction, i, imid)

            
        i += 1

    print(prediction_dict)
    json_path = "{}{}/predictions.json".format(base_path, save_name)
    with open(json_path, 'w') as f:
        json.dump(prediction_dict, f)

