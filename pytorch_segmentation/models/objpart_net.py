'''
authors: Gilad Katz & William Hinthorn
'''
import torch
import torch.nn as nn
# from .resnet import resnet34
from .resnet_fcn import resnet34
from .resnet_fcn import Resnet34_8s, Resnet34_8s_2b
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
        assert arch in ['drn', 'drn_54', 'resnet', 'resnet34_8s', 'resnet34_8s_2b','vgg', 'hourglass']
        super(OPSegNet, self).__init__()
 
        # one dimension for each object and part (plus background)
        # to do: in order to train on other datasets, will need to
        # push the bg class to the point file
        
        # GILAD
        self.object_max_val = 20  # in the op_map, needed for the 61-way case
        mult = 1  # 5 if bbox
        self.mult = mult
        # num_classes = 1 + sum(
        #     [1 * mult + output_dims[k] * mult for k in output_dims])
        # print("$" * 100)
        # print("[objpart.py] number of classes = {}".format(num_classes))
        # print("$" * 100)
        # TEMP
        # background + 20 classes + object or part
        num_classes_semseg = 21
        # num_classes_objpart = 2
        # num_classes = num_classes_semseg + num_classes_objpart   # TBC - be flexible for both 1-head, 2-head net
        
        self.two_branched = False
        self.already_upsampled = False

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
                           num_classes=num_classes_objpart,
                           out_middle=True,
                           remove_avg_pool_layer=True)

        elif arch == 'resnet34_8s':
            step_size = 32
            outplanes = None 	# output from the network already after 1d conv
                                # so outplanes already == num_classes
            self.already_upsampled = True
            net = Resnet34_8s(fully_conv=True,
                              pretrained=pretrained,
                              step_size=step_size,
                              num_classes=num_classes_objpart,
                              out_middle=True)

        elif arch == 'resnet34_8s_2b':
            step_size = 32
            outplanes = None
            self.already_upsampled = True
            # totally sperate branch for semseg (& not creating it from the objpart logits)
            self.two_branched = True
            net = Resnet34_8s_2b(fully_conv=True,
                                pretrained=pretrained,
                                step_size=step_size,
                                objpart_num_classes=num_classes_objpart)
            # net = Resnet34_8s_2b(objpart_num_classes=num_classes_objpart)
            print("[objpart_net.py] choosing resnet34_8s_2b, pretrained = {}".format(pretrained))

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
         
        self.fc = None        
        if self.inplanes:
            self.fc = nn.Conv2d(self.inplanes, num_classes_objpart, 1)
        
        if self.fc:
            _normal_initialization(self.fc)

        # branch for sem.set & brangh for obj.part
        # for now it's only 1D convolution
        # TODO: expand each branch, intuition: each branch will get experienced in it's own task
        # test different architectures for each branch & their combinations (what's needed to be
        # learned by each branch have different properties)
        # Can possibly add more inputs to each branch with respect to the corresponding task
        # For example - for the objpart branch we can add some ambiguity value / confidence
        # self.semseg_branch_conv1 = nn.Conv2d(self.inplanes, num_classes_semseg, 1)
        # self.objpart_branch_conv1 = nn.Conv2d(self.inplanes, num_classes_objpart, 1)

        # Randomly initialize the 1x1 Conv scoring layer
        # _normal_initialization(self.semseg_branch_conv1)
        # _normal_initialization(self.objpart_branch_conv1)

        self.num_classes = num_classes_objpart

        output_dims = {int(k): v for k, v in output_dims.items()}
        self.output_dims = output_dims
        # self.split_tensor = [1, len(self.output_dims)]
        split_tensor = [1]
        # maps each object prediction to its part(s) channels
        op_map = {}
        # Plus the number of part classes present for each cat
        i = 1
        for k in sorted(self.output_dims):
            if k > self.object_max_val:
                break   # for the 61-way
            split_tensor.append(mult)
            i += 1
        j = 1
        for k in sorted(self.output_dims):
            v = self.output_dims[k]
            if v > 0:
                split_tensor.append(v*mult)
                op_map[j] = i
                i += 1
            else:
                op_map[j] = -1
            j += 1

        self.op_map = op_map     # 61-way debugging needed. Make sure results are correct
        self.flat_map = {}
        if self.num_classes == 61:
            print("----- creating 61-way map")
            print("sorted(output_dims) = {}".format(sorted(self.output_dims)))
            for o in sorted(self.output_dims):
                if int(o) > self.object_max_val:
                    break
                p = op_map[o]
                amb = op_map[p]  
                self.flat_map[o] = (p, amb)   # sorted
                self.flat_map[p] = (o, amb)   # sorted
                self.flat_map[amb] = (o, p)   # sorted
        else:
            for k, v in op_map.items():
                if v > 0:
                    self.flat_map[k] = v
                    self.flat_map[v] = k
                    if self.num_classes == 61:
                        part_val = max(k, v)
                        self.flat_map[k] = (v,)

        self.split_tensor = split_tensor

        print("######## [objpart_net.py] flat_map = {}".format(self.flat_map))  
        
        if not self.already_upsampled:
            upsample = UpsamplingBilinearlySpoof(step_size)
            self.decode = upsample
        else:
            print("[objpart_net.py] already_upsampled = True")
            self.decode = None


    def forward(self, x):
        '''
        returns predictions for the object-part segmentation task and
        the semantic segmentation task
            Format:
            x = [background, obj_1, obj_2, ..., parts_1, parts_2, ...]
            out = [background, obj_or_part_1, obj_or_part_2, , ...]
        '''
        insize = x.size()[-2:]
        if self.two_branched:
            x, semseg_logits = self.net(x)
        else:
            x, _ = self.net(x)    # extract features
        
        # semseg branch
        # semseg_x = self.semseg_branch_conv1(x)
        # semseg_logits = self.decode([semseg_x], insize)
        
        # objpart branch
        # objpart_x = self.objpart_branch_conv1(x)
        # objpart_logits = self.decode([objpart_x], insize)

        if self.fc:
            x = self.fc(x)
        
        if self.decode:
            objpart_logits = self.decode([x], insize)
        else:
            objpart_logits = x
         
        if not self.two_branched:
            # Add object and part channels to predict a semantic segmentation
            splits = size_splits(objpart_logits, self.split_tensor, 1)
            # bg, objects, parts = splits[0], splits[1], splits[2:]

            bg, other_data = splits[0], splits[1:]
            op_data = [torch.sum(part, dim=1, keepdim=True)
                       for part in other_data]
            # the (-1) is since we separate out bg above
            out = []
            for o in sorted(self.op_map):
                # for the 61-way case, we also add the ambiguous val
                # and the op_map is for the range [1,40]
                # so we break after last value which is still object 
                if o > self.object_max_val:
                    break 
                to_add1 = op_data[o-1]
                p = self.op_map[o]
                if p > 0:
                    to_add2 = op_data[p-1]
                    # add here to_add3
                    if self.num_classes == 61:
                        print("##### [objpart_net.py] in forward(), inside num_classes == 61")
                        amb = self.op_map[p] 
                        to_add3 = op_data[amb-1]   # CHECK
                        out.append(to_add1 + to_add2 + to_add3)
                    else:
                        out.append(to_add1 + to_add2)
                else:
                    out.append(to_add1)

            # out = [op_data[o-1] + op_data[p-1] if p > 0 else
            #        op_data[o-1] for o, p in self.op_map.items()]
            out = torch.cat(out, dim=1)
            # parts = [torch.sum(part, dim=1, keepdim=True) for part in parts]
            # parts = torch.cat(parts, dim=1)
            # out = objects + parts
            semseg_logits = torch.cat([bg, out], dim=1)

        else:
            print("()()() returning semseg_logits taken directly from base net output")

	return objpart_logits, semseg_logits
