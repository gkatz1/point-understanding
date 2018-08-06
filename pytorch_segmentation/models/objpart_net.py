'''
authors: Gilad Katz & William Hinthorn
'''
import torch
import torch.nn as nn
from .resnet import resnet34
# from .upsampling import Upsampling
# from .upsampling_simplified import Upsampling
from .upsampling_simplified import UpsamplingBilinearlySpoof
from .drn import drn_d_107, drn_d_54
from .vgg import VGGNet
from .layers import size_splits

# pylint: disable=invalid-name, arguments-differ,too-many-locals
# pylint: disable=too-many-instance-attributes,too-many-statements


def _bake_function(f, **kwargs):
    import functools
    return functools.partial(f, **kwargs)


def _normal_initialization(layer):
    layer.bias.data.zero_()


class OPSegNet(nn.Module):
    ''' Unified network that allows for various
    feature extraction modules to be swapped in and out.
    '''
    def __init__(self,
                 arch,
                 output_dims,
                 num_classes_objpart=2,
                 upsample='bilinear',
                 pretrained=True):
        
        print("[#] [objpart_net.py] num_classes_objpart = {}".format(num_classes_objpart))
        print(arch)
        assert arch in ['drn', 'drn_54', 'resnet', 'vgg', 'hourglass']
        super(OPSegNet, self).__init__()
 
        # one dimension for each object and part (plus background)
        # to do: in order to train on other datasets, will need to
        # push the bg class to the point file
        mult = 1  # 5 if bbox
        self.mult = mult
        
        # TEMP
        # background + 20 classes + object or part
        num_classes_semseg = 21
        # num_classes_objpart = 2
        num_classes = num_classes_semseg + num_classes_objpart        # TBC - to be flexible for both 1-head, 2-head net
        
        
        if arch == 'resnet':
            step_size = 8
            outplanes = 512
            # This comment block refers to the parameters for the
            # depracated upsampling unit.
            # Net will output y -> list of outputs from important blocks
            # feature_ind denotes which output to concatenate with upsampled
            # 0: conv1
            # 1: layer1
            # Index the feature vector from the base network
            # feature_ind = [0, 1]
            # Give the widths of each feature tensor (i.e dim 2 length)
            # Order is from smallest spatial resolution to largest
            # feature_widths = [64, 64]
            # add skip connections AFTER which upsamples
            # 0 here means BEFORE the first
            # merge_which_skips = set([1, 2])
            # mergedat = {0: (64, 2), 1: (64, 4)}

            # Number of channels at each stage of the decoder
            # upsampling_channels = [512, 128, 64, 32]
            net = resnet34(fully_conv=True,
                           pretrained=pretrained,
                           output_stride=step_size,
                           out_middle=True,
                           remove_avg_pool_layer=True)

        elif arch == 'vgg':
            step_size = 16
            outplanes = 1024
            # feature_ind = [3, 8, 15, 22]
            # feature_widths = [512, 256, 128, 64]
            # merge_which_skips = set([1, 2, 3, 4])
            # upsampling_channels = [1024, 256, 128, 64, 32]
            # mergedat = {15: (512, 8), 8: (256, 4), 1: (128, 2)}
            net = VGGNet()
            raise NotImplementedError(
                "VGGNet architecture not yet debugged")

        elif arch == 'hourglass':
            # step_size = ???
            raise NotImplementedError(
                "Hourglass network architecture not yet implemented")

        elif arch == 'drn':
            step_size = 8
            outplanes = 512
            # feature_ind = [0, 1, 2]
            # feature_widths = [256, 32, 16]
            # merge_which_skips = set([1, 2, 3])
            # feature_ind = [1, 2]
            # feature_widths = [256, 32]
            # merge_which_skips = set([1, 2])
            # upsampling_channels = [512, 128, 64, 32]
            # mergedat = {2: (256, 4), 1: (32, 2)}
            net = drn_d_107(pretrained=pretrained, out_middle=True)

        elif arch == 'drn_54':
            step_size = 8
            outplanes = 512
            net = drn_d_54(pretrained=pretrained, out_middle=True)

        # self.inplanes = # num_classes # upsampling_channels[-1]
        # head_outchannels = outplanes // step_size
        # skip_outchannels = [mergedat[k][0] // mergedat[k][1]
        # for k in mergedat]
        # merge_inchannels = head_outchannels + sum(skip_outchannels)
        # self.inplanes = merge_inchannels
        self.inplanes = outplanes
        self.net = net
        
        # branch for sem.set & brangh for obj.part
        # for now it's only 1D convolution
        # TODO: expand each branch, intuition: each branch will get experienced in it's own task
        # test different architectures for each branch & their combinations (what's needed to be
        # learned by each branch have different properties
        # Can possibly add more inputs to each branch with respect to the corresponding task
        # For example - for the objpart branch we can add some ambiguity value / confidence
        self.semseg_branch_conv1 = nn.Conv2d(self.inplanes, num_classes_semseg, 1)
        self.objpart_branch_conv1 = nn.Conv2d(self.inplanes, num_classes_objpart, 1)

        # Randomly initialize the 1x1 Conv scoring layer
        _normal_initialization(self.semseg_branch_conv1)
        _normal_initialization(self.objpart_branch_conv1)

        self.num_classes = num_classes
    
        upsample = UpsamplingBilinearlySpoof(step_size)
        self.decode = upsample
 
        self.flat_map = None     # GILAD - used in order to keep code structre as it is

    def forward(self, x):
        '''
        returns predictions for the object-part segmentation task and
        the semantic segmentation task
            Format:
            x = [background, obj_1, obj_2, ..., parts_1, parts_2, ...]
            out = [background, obj_or_part_1, obj_or_part_2, , ...]
        '''
        
        insize = x.size()[-2:]
        x, _ = self.net(x)    # extract features
        
        # semseg branch
        semseg_x = self.semseg_branch_conv1(x)
        semseg_logits = self.decode([semseg_x], insize)
        
        # objpart branch
        objpart_x = self.objpart_branch_conv1(x)
        objpart_logits = self.decode([objpart_x], insize)

        return objpart_logits, semseg_logits
