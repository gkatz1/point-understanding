'''Training various models on on Pascal Object-Part task
    authors: Gilad Katz & William Hinthorn
    Loosely based on Daniil Pakhomov's (warmspringwinds)
    training code for semantic segmentation.

'''
# flake8: noqa = E402
# pylint: disable = fixme,wrong-import-position,unused-import,import-error,too-many-statements,too-many-locals,
# pylint: disable = invalid-name, len-as-condition, too-many-branches
import sys
import os
import argparse
import datetime
import tqdm
import csv

import tensorboardX
import torch
import torch.autograd
from torch.autograd import Variable

# import tensorboardX

# from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix

# Allow for access to the modified torchvision library and
# various datasets
PATH = os.path.dirname(os.path.realpath(__file__))
PATHARR = PATH.split(os.sep)
print(PATHARR)
HOME_DIR = os.path.join(
    '/', *PATHARR[:PATHARR.index('point-understanding') + 1])
DATASET_DIR = os.path.join(HOME_DIR, 'datasets')
sys.path.insert(0, HOME_DIR)

import pytorch_segmentation
from pytorch_segmentation.evaluation import (
    poly_lr_scheduler,
    get_training_loaders,
    get_valid_logits,
    get_valid_annos,
    numpyify_logits_and_annotations,
    outputs_tonp_gt,
    compress_objpart_logits,
    get_iou,
    validate_batch,
    get_precision_recall,
    save_checkpoint,
    validate_and_output_images,
    get_network_and_optimizer)


def str2bool(v):
    ''' Helper for command line args.
    '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_dump(dump_dir, dump_filename, var_dict):
    if not os.path.isdir(dump_dir):
        os.makedirs(dump_dir)
    full_path = os.path.join(dump_dir, dump_filename)
    cur_path = full_path
    i = 1
    while True:
        if os.path.isfile(cur_path + '.csv'):
            cur_path = full_path + "_v{}".format(i)
            i += 1
            continue
        break
    file_path = cur_path + '.csv'
    with open(file_path, 'wb') as f:
        writer = csv.writer(f, delimiter=':')
        for k, v in var_dict.items():
            writer.writerow([k, v])


def load_dump_file_params(dump_file_path):
    delim = ':'
    saved_params = dict()

    if not os.path.isfile(dump_file_path):
        return None
    with open(dump_file_path, 'rb') as f:
        reader = csv.reader(f, delimiter=delim)
        for row in reader:
            key, val = row
            saved_params[key] = val

    return saved_params


def main(args):
    '''
    main(): Primary function to manage the training.
    '''
    print("[#](1) In main")
    print(args)
    # *************************************
    architecture = args.arch
    experiment_name = args.experiment_name
    batch_size = args.batch_size
    validate_first = args.validate_first
    validate_only = args.validate_only
    save_model_name = args.save_model_name
    if save_model_name == 'dbg':
        pass   # Make it dbg_x (where x is the minumin unsed number for dbg_x)
    load_model_name = args.load_model_name
    dump_file_name = args.dump_file_name
    num_workers = args.num_workers
    which_vis_type = args.paint_images
    print(which_vis_type)
    assert which_vis_type in [None, 'None', 'objpart', 'semantic', 'separated']
    merge_level = args.part_grouping
    assert merge_level in ['binary', 'trinary','sparse', 'merged', '61-way']
    mask_type = args.mask_type
    # 2 options: 1 value for both train & validation,
    # 1 value for train, 1 value for validation
    # 2 values will be separated by ','
    try:
        mask_type_train, mask_type_val = mask_type.split(',')
    except:
        mask_type_train, mask_type_val = mask_type, mask_type
    assert mask_type_train in ['mode', 'consensus', 'consensus_or_ambiguous', '61-way', 'consensus_or_mask_out']
    assert mask_type_val in ['mode', 'consensus', 'consensus_or_ambiguous','61-way', 'consensus_or_mask_out']
    mask_type = dict()
    mask_type['train'] = mask_type_train
    mask_type['val'] = mask_type_val
    print(mask_type)
 
    device = args.device
    epochs_to_train = args.epochs_to_train
    validate_batch_frequency = args.validate_batch_frequency
    _start_epoch = args.origin_epoch
    _load_folder = args.load_folder
    # model_type = args.model_type
    num_branches = args.num_branches
    # number_of_objpart_classes = args.num_objpart_classes
    dump_dir = args.dump_dir
    objpart_weight = args.objpart_weight
    assert objpart_weight >= 0 and objpart_weight <=1
    loss_type = args.loss_type
    assert loss_type in ['1_loss', '2_losses_combined', 'semseg_only']
    compression_method = args.compression_method
    assert compression_method in ['gt', 'sum', 'predicted_label', 'none']
    dump_output_images = args.dump_output_images

    # **************************************
    time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    # Define training parameters
    # edit below
    # *******************************************************
    experiment_name = experiment_name + '_' + time_now  # "drn_objpart_" + time_now
    # architecture = ['resnet', 'drn', 'vgg', 'hourglass'][0]
    # init_lr = 0.0016  # 2.5e-4
    init_lr = 0.0001
    # batch_size = 12  # 8
    # num_workers = 4
    # number_of_classes = 41
    number_of_semantic_classes = 21  # Pascal VOC
    # number_of_objpart_classes = 2   # object or part        # GILAD
    semantic_labels = range(number_of_semantic_classes)
   
    max_object_label = 20
 
    #  validate_first = False
    output_predicted_images = True    # GILAD
    # which_vis_type = ['None' ,'objpart', 'semantic', 'separated'][0]
    if which_vis_type is None or which_vis_type == 'None':
        output_predicted_images = False


    # merge_level = 'binary'  # 'merged', 'sparse'
    # save_model_name = "drn"  # 'resnet_34_8s'
    save_model_folder = os.path.join(
        HOME_DIR,
        'pytorch_segmentation',
        'training',
        'experiments', save_model_name)
    if not os.path.isdir(save_model_folder):
        os.makedirs(save_model_folder)

    load_model_path = os.path.join(
        HOME_DIR, 'pytorch_segmentation', 'training', _load_folder)
    if load_model_name is not None:
        # load_model_name = 'resnet_34_8s_model_best.pth.tar'
        # load_model_path = os.path.join(
        #     HOME_DIR, 'pytorch_segmentation', 'training', 'models', load_model_name)
        # load_model_name = 'drn_model_best.pth.tar'
        load_model_path = os.path.join(load_model_path, load_model_name)

    # iter_size = 20
    # epochs_to_train = 40
    # mask_type = "mode"  # "consensus"
    # device = '2'  # could be '0', '1', '2', or '3' on visualai#
    # validate_batch_frequency = 20  # compute mIoU every k batches
    # End define training parameters
    # **********************************************************
    
    if load_model_name and dump_file_name:
        # load params
        dump_file_path = os.path.join(dump_dir, dump_file_name)
        saved_params = load_dump_file_params(dump_file_path)
        if saved_params:
            print("==> loading params from dump file")
            architecture = type(architecture)(saved_params['architecture'])
            batch_size = type(batch_size)(saved_params['batch_size'])
            compression_method = type(compression_method)(saved_params['compression_method'])
            merge_level = type(merge_level)(saved_params['merge_level'])
            loss_type = type(loss_type)(saved_params['loss_type'])
            num_branches = type(num_branches)(saved_params['num_branches'])
            objpart_weight = type(objpart_weight)(saved_params['objpart_weight'])
            mask_type['train'] = mask_type_train = type(mask_type_train)(saved_params['mask_type_train'])
            mask_type['val'] = mask_type_val = type(mask_type_val)(saved_params['mask_type_val'])

    print("Setting visible GPUS to machine {}".format(device))

    # Use second GPU -pytorch-segmentation-detection- change if you want to
    # use a first one
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    print("=> Getting training and validation loaders")
    print(DATASET_DIR)
    network_dims = {}
    (trainloader, trainset), (valset_loader, _), (valset_for_output_im_loader, _) = get_training_loaders(
        DATASET_DIR, network_dims, batch_size,
        num_workers, mask_type, merge_level, validate_and_output_im=output_predicted_images)

    if merge_level == 'binary':
        number_of_objpart_classes = 2
        compression_method = 'none'
        if output_predicted_images:
            if which_vis_type == 'objpart':
                which_vis_type = 'separated'
    elif merge_level == 'trinary':
        number_of_objpart_classes = 3
        compression_method = 'none'
        if output_predicted_images:
            if which_vis_type == 'objpart':
                which_vis_type = 'separated'
    elif merge_level == '61-way':
        number_of_objpart_classes = 1
        print("network_dims: {}".format(network_dims))
        for k in sorted(network_dims):
            number_of_objpart_classes += network_dims[k]
            if int(k) <= max_object_label:
                number_of_objpart_classes += 1
    else:
        number_of_objpart_classes = 1  # 1 for background...
        print("network_dims: {}".format(network_dims))
        for k in network_dims:
            number_of_objpart_classes += 1 + network_dims[k]

    objpart_labels = range(number_of_objpart_classes)
    print("Training for {} objpart categories".format(
        number_of_objpart_classes))

    make_dump(dump_dir, save_model_name, locals())
    # GILAD
    # number_of_classes = 21
    
    print("=> Creating network and optimizer")
    train_params = {}
    net, optimizer = get_network_and_optimizer(
        architecture,
        network_dims,
        number_of_objpart_classes,
        loss_type,
        load_model_name is not None, load_model_path, train_params, init_lr)
    try:
        train_params['best_op_val_score'] = train_params['best_objpart_val_score']
        train_params['best_sem_val_score'] = train_params['best_semantic_val_score']
        # train_params['best_objpart_accuracy'] = 0.5
    except KeyError:
        print(list(train_params))
    writer = tensorboardX.SummaryWriter(os.path.join('runs', experiment_name))
    if _start_epoch is not None:
        train_params['start_epoch'] = _start_epoch
    train_params.update({
        'net': net,
        'optimizer': optimizer,
        'epochs_to_train': epochs_to_train,
        'trainloader': trainloader,
        'trainset': trainset,
        'valset_loader': valset_loader,
        'init_lr': init_lr,
        'writer': writer,
        'validate_batch_frequency': validate_batch_frequency,
        # 'number_of_classes': number_of_classes,
        'number_of_semantic_classes': number_of_semantic_classes,
        'number_of_objpart_classes' : number_of_objpart_classes,    # GILAD
        'save_model_name': save_model_name,
        'save_model_folder': save_model_folder,
        'num_branches' : num_branches,
        'which_vis_type' : which_vis_type,
        'merge_level' : merge_level,
        'objpart_weight' : objpart_weight,
        'loss_type' : loss_type,
        'compression_method' : compression_method,
        'dump_output_images' : dump_output_images
    })

    op_map = net.flat_map     # GILAD

    if output_predicted_images:
        print("=> Outputting predicted images to folder 'predictions'")
        validate_and_output_images(  
            net, valset_for_output_im_loader, op_map, which=which_vis_type, alpha=0.7, writer=writer,
             step_num=0, save_name=save_model_name)
        while True:
            resp = raw_input("=> Done. Do you wish to continue training? (y/n):\t")
            if resp[0] == 'n':
                return
            elif resp[0] == 'y':
                break
            else:
                print("{} not understood".format(resp))
     
    # if visulize_first:	# GILAD
    #     validate_and_output_images(net, valset_loader, None)

    if validate_first or validate_only:
        print("=> Validating network")
        sc1, sc2, sc3 = validate(
            net, valset_loader, (objpart_labels, semantic_labels), compression_method)
        print("results:\nobjpart mIoU = {}\nsemantic mIoU = {}\n".format(sc1, sc2) + 
              "overall objpart accuracy = {}".format(sc3))

    if not validate_only:
        print("=> Entering training function")
        train(train_params)
        print('=> Finished Training')
    writer.close()


def validate(net, loader, labels, compression_method):
    ''' Computes mIoU for``net`` over the a set.
        args::
            :param ``net``: network (in this case resnet34_8s_dilated
            :param ``loader``: dataloader (in this case, validation set loader)

        returns the mIoU (ignoring classes where no parts were annotated) for
        the semantic segmentation task and object-part inference task.
    '''

    net.eval()
    # hardcoded in for the object-part infernce
    # TODO: change to be flexible/architecture-dependent

    objpart_labels, semantic_labels = labels

    overall_sem_conf_mat = None
    overall_part_conf_mat = None
    gt_total = 0.0
    gt_right = 0.0
    showevery = 50
    # valset_loader
    for i, (image, semantic_anno, objpart_anno) in enumerate(
            tqdm.tqdm(loader)):
        image = Variable(image.cuda())
        objpart_logits, semantic_logits = net(image)
        print("objpart.size = {}, sem.size = {}".format(objpart_logits.size(), semantic_logits.size()))
        if i == 0:
            num_objpart_classes = objpart_logits.size()[-1]
        
        print("[#] DEBUG: train.py, validate()")

        # First we do argmax on gpu and then transfer it to cpu
        sem_pred_np, sem_anno_np = numpyify_logits_and_annotations(
            semantic_logits, semantic_anno)
        # objpart_pred_np, objpart_anno_np = numpyify_logits_and_annotations(
        #     objpart_logits, objpart_anno)
        if compression_method == 'none':     # GILAD

            objpart_pred_np, objpart_anno_np = numpyify_logits_and_annotations(
                    objpart_logits, objpart_anno)
        # elif merge_level == '61-way':
        #     objpart_pred_np, objpart_anno_np = outputs_tonp_gt(
        #         objpart_logits, objpart_anno, net.flat_map)
        else:
            objpart_pred_np, objpart_anno_np = outputs_tonp_gt(
                objpart_logits, objpart_anno, net.flat_map)

        print("objpart_pred_np = {}, objpart_anno_np = {}".format(objpart_pred_np,
                objpart_anno_np))
        print("objpart_pred_np: #1s = {}, #0s = {}".format(np.sum(objpart_pred_np == 1),
                np.sum(objpart_pred_np == 0)))
        print("objpart_anno_np: #1s = {}, #0s = {}".format(np.sum(objpart_anno_np == 1),
                np.sum(objpart_anno_np == 0)))
        # opprednonz = np.array([el for el in objpart_pred_np if el > -2])
        # opannononz = np.array([el for el in objpart_anno_np if el > -2])
        # print("opprednonz = {}, opannononz = {}".format(opprednonz, opannononz))
        # gt_right += np.sum((opprednonz == opannononz))
        gt_right += np.sum(objpart_pred_np == objpart_anno_np)
        gt_total += np.sum(objpart_anno_np != -2)	# TBC (-2 --> mask_out value)
        print("gt_right = {}, get_total = {}".format(gt_right, gt_total))
        correct_idxs = np.nonzero(objpart_pred_np == objpart_anno_np)
	print("correct idxs = {}".format(correct_idxs))
        print("pred[correct_idxs] = {}, annot[correct_idxs] = {}".format(
                objpart_pred_np[correct_idxs], objpart_anno_np[correct_idxs]))
        current_semantic_confusion_matrix = confusion_matrix(
            y_true=sem_anno_np, y_pred=sem_pred_np, labels=semantic_labels)

        if overall_sem_conf_mat is None:
            overall_sem_conf_mat = current_semantic_confusion_matrix
        else:
            overall_sem_conf_mat += current_semantic_confusion_matrix

        if (objpart_anno_np > 0).sum() == 0:
            continue
        current_objpart_conf_mat = confusion_matrix(
            y_true=objpart_anno_np, y_pred=objpart_pred_np,
            labels=objpart_labels)
        if overall_part_conf_mat is None:
            overall_part_conf_mat = current_objpart_conf_mat
        else:
            overall_part_conf_mat += current_objpart_conf_mat
        if i % showevery == 1:
            tqdm.tqdm.write(
                "Object-part accuracy ({}):\t{:%}".format(i, gt_right / gt_total))

    if num_objpart_classes == 2 or num_objpart_classes == 3:     # GILAD
        no_parts = []
    else:
        no_parts = [0, 4, 9, 11, 18, 20, 24, 29, 31, 38, 40]

    # Semantic segmentation task
    semantic_IoU = get_iou(
        overall_sem_conf_mat)

    semantic_mIoU = np.mean(
        semantic_IoU)

    objpart_prec, objpart_rec = get_precision_recall(
        overall_part_conf_mat)
    tqdm.tqdm.write(
        "precision/recall:\t\n{}\n\n{}\n".format(
            [
                objpart_prec[i] for i, _ in enumerate(objpart_prec) if i not in no_parts], [
                    objpart_rec[i] for i, _ in enumerate(objpart_rec) if i not in no_parts]))

    # Part segmentation task
    objpart_IoU = get_iou(overall_part_conf_mat)
    objpart_mIoU = np.mean([objpart_IoU[i] for i,
                            _ in enumerate(objpart_IoU) if i not in no_parts])
    overall_objpart_accuracy = gt_right / gt_total
    net.train()
    return objpart_mIoU, semantic_mIoU, overall_objpart_accuracy


def train(train_params):
    ''' Main function for training the net.
    '''
    net = train_params['net']
    optimizer = train_params['optimizer']
    start_epoch = train_params['start_epoch']
    epochs_to_train = train_params['epochs_to_train']
    trainloader = train_params['trainloader']
    # trainset = train_params['trainset']
    # batch_size = train_params['batch_size']
    valset_loader = train_params['valset_loader']
    init_lr = train_params['init_lr']
    writer = train_params['writer']
    validate_batch_frequency = train_params['validate_batch_frequency']
    best_op_val_score = train_params['best_op_val_score']
    best_op_gt_valscore = train_params['best_objpart_accuracy']
    best_op_gt_valscore = 0.0
    best_sem_val_score = train_params['best_sem_val_score']
    semantic_criterion = train_params['semantic_criterion']
    objpart_criterion = train_params['objpart_criterion']
    # number_of_classes = train_params['number_of_classes']
    number_of_semantic_classes = train_params['number_of_semantic_classes']
    number_of_objpart_classes = train_params['number_of_objpart_classes']
    save_model_folder = train_params['save_model_folder']
    save_model_name = train_params['save_model_name']
    num_branches = train_params['num_branches']
    which_vis_type = train_params['which_vis_type']
    merge_level = train_params['merge_level']
    objpart_weight = train_params['objpart_weight']
    semantic_weight = 1 - objpart_weight
    loss_type = train_params['loss_type']
    compression_method = train_params['compression_method']
    dump_output_images = train_params['dump_output_images']
    print("dump_output_images = {}".format(dump_output_images))

    # could try to learn these as parameters...
    # currently not implemented
    objpart_weight = Variable(torch.Tensor([objpart_weight])).cuda()
    semantic_weight = Variable(torch.Tensor([semantic_weight])).cuda()
    _one_weight = Variable(torch.Tensor([1])).cuda()

    objpart_labels = range(number_of_objpart_classes)   # GILAD
    semantic_labels = range(number_of_semantic_classes)

    spatial_average = False  # True
    batch_average = True

    # loop over the dataset multiple times
    print(
        "Training from epoch {} for {} epochs".format(
            start_epoch,
            epochs_to_train))
    sz = None
    number_training_batches = len(trainloader)
    print("#training batches = %d" % number_training_batches)
    for epoch in tqdm.tqdm(range(start_epoch, start_epoch + epochs_to_train)):
        with open("compress_logits_sizes_debug", 'a+') as f:
            f.write("epoch {}\n".format(epoch))
        # semantic_running_loss = 0.0
        # objpart_running_loss = 0.0
        poly_lr_scheduler(optimizer, init_lr, epoch,
                          lr_decay_iter=1, max_iter=100, power=0.9)
        overall_sem_conf_mat = None
        overall_part_conf_mat = None
        correct_points = 0.0
        num_points = 0.0
        tqdm.tqdm.write("=> Starting epoch {}".format(epoch))
        tqdm.tqdm.write("=> Current time:\t{}".format(datetime.datetime.now()))
        for i, data in tqdm.tqdm(
                enumerate(
                    trainloader, 0), total=number_training_batches):
            print("[+] step %d" % i)
            # if i == 5:   # for TESTING validate() func
            #     break
            # img, semantic_anno, objpart_anno, objpart_weights=data
            img, semantic_anno, objpart_anno = data
            # print("~~~~~ objpart_anno ~~~~~")
            # print(objpart_anno)    # DEBUG
            # DEBUG
            objpart_anno_0s_idxs = np.nonzero(objpart_anno == 0)
            objpart_anno_1s_idxs = np.nonzero(objpart_anno == 1)
            # print("[#](1.7) DEBUG: 1s idxs in objpart_anno: {} 0s idxs: {}".format(
            #         objpart_anno_1s_idxs, objpart_anno_0s_idxs))
            batch_size = img.size(0)

            if sz is None:
                sz = np.prod(img.size())

            # We need to flatten annotations and logits to apply index of valid
            # annotations. All of this is because pytorch doesn't have
            # tf.gather_nd()
            semantic_anno_flatten_valid, semantic_index = get_valid_annos(
                semantic_anno, 255)
            op_anno_flt_vld, objpart_index = get_valid_annos(
                objpart_anno, -2)

            # debug
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # print("[#] [train.py] (train) #(-2)'s = {}, #(-1)'s = {} in objpart annotation".format(
            #     torch.nonzero(objpart_anno == -2).size(), torch.nonzero(objpart_anno == -1).size()))

            # check if index arrays are sorted
            # sem_is_sorted = np.all(np.diff(semantic_index.numpy()) >= 0)
            # objpart_is_sorted = np.all(np.diff(objpart_index.numpy()) >= 0)
            # print("^" * 100)
            # print("sorted: sem? {}, objpart? {}".format(sem_is_sorted, objpart_is_sorted))
            # print(semantic_index.numpy())
            # print(objpart_index.numpy())
            # return
              
            # check here
            if loss_type == '1_loss' or (merge_level != 'binary' and merge_level != 'trinary'):
                sem_idx_np = semantic_index.numpy()
                objpart_idx_np = objpart_index.numpy()
                sorted_idxs = np.searchsorted(sem_idx_np, objpart_idx_np) 
                valid_objpart_idxs_mask = sem_idx_np[sorted_idxs] == objpart_idx_np
                objpart_mask_idxs = sorted_idxs[valid_objpart_idxs_mask]
                valid_objpart_idxs_to_loss = np.array(range(len(objpart_idx_np)))
                valid_objpart_idxs_to_loss = valid_objpart_idxs_to_loss[valid_objpart_idxs_mask]
                objpart_mask_for_loss = objpart_mask_idxs
                # objpart_mask_for_loss = np.full_like(sem_idx_np, False)
                # objpart_mask_for_loss[objpart_mask_idxs] = True

                # debug
                # print("^" * 80)
                # print("objpart_mask_for_loss = {}".format(objpart_mask_for_loss))
                # print("\n\n")
                # print(objpart_idx_np)
                # print("\n\n")
                # print(sem_idx_np[objpart_mask_for_loss])
                # print(objpart_idx_np == sem_idx_np[objpart_mask_for_loss])
                # print("^" * 80)
                with open('objpart_mask_for_loss_debug', 'ab') as f:
                    f.write("{}\n".format(np.all(objpart_idx_np == sem_idx_np[objpart_mask_for_loss])))
                # objpart_mask_for_loss = np.uinti8(objpart_mask_for_loss) # should it be long?
                objpart_mask_for_loss = torch.from_numpy(objpart_mask_for_loss).cuda()
                valid_objpart_idxs_to_loss = torch.from_numpy(valid_objpart_idxs_to_loss).long().cuda()
                # print("** type = {} **".format(type(objpart_mask_for_loss)))

            print("objpart_anno.shape = {}".format(objpart_anno.shape))
            existence_ctr = 0
            for objpart_idx in objpart_index:
                try:
                    exists = torch.nonzero(objpart_idx == semantic_index)[0]
                    existence_ctr += 1
                except:
                    pass

            with open("point_annotation_debug1", 'ab') as f:
                if not list(objpart_index.size())[0] == existence_ctr:
                    f.write("objpart_idx.size() = {}\n".format(list(objpart_index.size())[0]))
                    f.write("existence_ctr = {}".format(existence_ctr))
                    f.write("\n\n\n")

            # wrap them in Variable
            # the index can be acquired on the gpu
            img, semantic_anno_flatten_valid, semantic_index = Variable(
                img.cuda()), Variable(
                    semantic_anno_flatten_valid.cuda()), Variable(
                        semantic_index.cuda())
            op_anno_flt_vld, objpart_index = Variable(
                op_anno_flt_vld.cuda()), Variable(
                    objpart_index.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()
            # adjust_learning_rate(optimizer, loss_current_iteration)

            # forward + backward + optimize
            objpart_logits, semantic_logits = net(img)
            print("*" * 100)
            print("[train.py] objpart_logits.size = {}, semantic_logits.size = {}".format(
                  objpart_logits.size(), semantic_logits.size()))
            # print("[#](1.8) DEBUG: objpart_logits, semantic_logits")
            # print(objpart_logits.size())
            # print(semantic_logits.size())
            # import pdb; pdb.set_trace()
            print("[#] [train.py] (train) number_of_objpart_classes = {}".format(number_of_objpart_classes))
            op_log_flt_vld = get_valid_logits(
                objpart_logits, objpart_index, number_of_objpart_classes)
            semantic_logits_flatten_valid = get_valid_logits(
                semantic_logits, semantic_index, number_of_semantic_classes)
            # Compress objpart logits reduces the problem to a binary inference
            #
            # op_log_flt_vld, op_anno_flt_vld = compress_objpart_logits(
            #     op_log_flt_vld, op_anno_flt_vld)

            # Score the overall accuracy
            # prepend _ to avoid interference with the training
    
            if merge_level == 'binary' or merge_level == 'trinary':
                _op_log_flt_vld = op_log_flt_vld
                _op_anno_flt_vld = op_anno_flt_vld
            # elif merge_level == '61-way':
            #     _op_log_flt_vld, _op_anno_flt_vld = compress_objpart_logits_61_way(
            #         op_log_flt_vld, op_anno_flt_vld, net.flat_map)
            else:
                semantic_logits_to_compression = None
                if compression_method == 'predicted_label':
                    semantic_logits_to_compression = semantic_logits_flatten_valid.data.index_select(0,
                        objpart_mask_for_loss)   # is it right to use .data ?
                _op_log_flt_vld, _op_anno_flt_vld = compress_objpart_logits(
                    op_log_flt_vld, op_anno_flt_vld, net.flat_map,
                    method=compression_method,
                    semseg_logits=semantic_logits_to_compression)

            if len(_op_log_flt_vld) > 0:
                # if merge_level == 'binary' or merge_level == 'trinary' or True:  # exp_33, exp_34
                if compression_method != 'none':
                    _, logits_toscore = torch.max(_op_log_flt_vld, dim=1)
                    correct_idxs = np.nonzero(logits_toscore.data == _op_anno_flt_vld.data)
                    this_correct_points = torch.sum(logits_toscore == _op_anno_flt_vld).float().data[0]
                    # print("[train.py] correct idxs = {}".format(correct_idxs))   # DEBUG 
                    correct_points += this_correct_points
                    num_points += len(_op_anno_flt_vld)

                    # Balance the weights
                    this_num_points = len(_op_anno_flt_vld) # float(len(_ind_))   # GILAD
                    print("""this_num_coorect_points = {}, this_num_points = {}, total_number_of_correct = {},  
                          total_num_of_points = {}""".format(this_correct_points,
                          this_num_points, correct_points, num_points))
                    op_scale = Variable(torch.Tensor([this_num_points])).cuda() # _one_weight
                    # if num_points != 0:
                    #     multiplier = Variable(torch.Tensor(
                    #         [sz / this_num_points])).cuda()
               
                else:
                    _, logits_toscore = torch.max(_op_log_flt_vld, dim=1)
                    _ind_ = torch.gt(_op_anno_flt_vld, 0)   # why is this? if we don't compress, 0 is the bg, don't use it
                    op_pred_gz = logits_toscore[_ind_]
                    _op_anno_gz = _op_anno_flt_vld[_ind_]
                    print("!! (op_pred_gz == _op_anno_gz).sum().data[0] = {}".format(torch.sum(op_pred_gz == _op_anno_gz).data[0]))
                    correct_points += float(torch.sum(op_pred_gz == _op_anno_gz).data[0])
                    num_points += len(_op_anno_gz)
                    # Balance the weights
                    this_num_points = float(len(_ind_))
                    op_scale = Variable(torch.Tensor([this_num_points])).cuda() # _one_weight

                # Compute cross-entropy loss for the object-part inference task
                objpart_loss = objpart_criterion(
                    op_log_flt_vld, op_anno_flt_vld)

                # debug
                # _MASK_OUT_VAL = -2
                # non_mask_out = torch.nonzero(op_anno_flt_vld.data != _MASK_OUT_VAL)
                # with open("point_annotation_debug", 'ab') as f:
                #     f.write("\n\n")
                #     for el in enumerate(non_mask_out):
                #         f.write("{}, ".format(el[0]))

            else:
                objpart_loss = Variable(torch.Tensor([0])).cuda()
                op_scale = objpart_weight


            sem_scale = Variable(torch.FloatTensor(
                [(semantic_anno_flatten_valid.data > 0).sum()])).cuda()
            semantic_loss = semantic_criterion(
                semantic_logits_flatten_valid, semantic_anno_flatten_valid)

            # import pdb; pdb.set_trace()
            if spatial_average:
                semantic_loss /= sem_scale
                objpart_loss /= op_scale

            # tqdm.tqdm.write("oploss, semloss:\t({}, {}) - {} - {:%}".format(
            # objpart_loss.data[0], semantic_loss.data[0], len(_ind_),
            # correct_points/num_points))

            # TODO: Consider clipping??
            # Consider modulating the weighting of the losses?
            semantic_batch_weight = semantic_weight
            objpart_batch_weight = objpart_weight
            if loss_type == 'semseg_only':
                print("segmentation loss only!")
                loss = semantic_loss
            else:
                if loss_type == '1_loss':    # check here
                    print("=" * 80)
                    print("{}, {}".format(objpart_mask_for_loss.size(), objpart_loss.size()))
                    semantic_loss[objpart_mask_for_loss] = objpart_loss[valid_objpart_idxs_to_loss]  # both are 1D vectors
                    loss = semantic_loss.mean()
                    with open('loss_type_debug', 'ab') as f:
                        f.write("{}\n".format(loss_type))
                elif loss_type == '2_losses_combined':
                    print("[train.py] sem_weight = {}, objpart_weight= {}".format(
                        semantic_batch_weight, objpart_batch_weight))
                    loss = semantic_loss * semantic_batch_weight + \
                    objpart_loss * objpart_batch_weight
                else:
                    raise NotImplementedError("loss_type {} not implements".format(
                                              loss_type))

            # compute loss mean for the 1_loss method

            if batch_average:
                loss = loss / batch_size

            # summary of per pixel loss into one number
            # if loss_type == '1_loss':
            #    sem_loss_val = # mean of sem_los *before* assignment of objpart_loss into it
            if loss_type == '1_loss':
                writer.add_scalar(
                    'losses/loss',
                    loss,
                number_training_batches * epoch + i)
            else:
                sem_loss_val = semantic_loss.data[0] / semantic_logits_flatten_valid.size(0)
                writer.add_scalar(
                    'losses/semantic_loss',
                    sem_loss_val,
                    number_training_batches * epoch + i)

                if len(op_log_flt_vld) > 0:
                    objpart_loss_val = objpart_loss.data[0] / op_log_flt_vld.size(0)
                    writer.add_scalar(
                        'losses/objpart_loss',
                        objpart_loss_val,
                        number_training_batches * epoch + i)

                    combined_loss_val = (semantic_batch_weight * sem_loss_val +
                                        objpart_batch_weight * objpart_loss_val)
                    writer.add_scalar(
                        'losses/combined_loss',
                        combined_loss_val,
                        number_training_batches * epoch + i)

            loss.backward()
            optimizer.step()
            if i % validate_batch_frequency == 1:
                valout = validate_batch(
                    (objpart_logits, objpart_anno),
                    (semantic_logits, semantic_anno),
                    overall_part_conf_mat, overall_sem_conf_mat,
                    (objpart_labels, semantic_labels), merge_level, net.flat_map,
                    writer,
                    number_training_batches * epoch + i)
                ((objpart_mPrec, objpart_mRec),
                 semantic_mIoU,
                 overall_part_conf_mat,
                 overall_sem_conf_mat) = valout
                # tqdm.tqdm.write("OP Acc ({}):\t{:%}".format(i, correct_points/num_points))
                if num_points > 0:
                    writer.add_scalar('data/obj_part_accuracy',
                                      correct_points / num_points,
                                      number_training_batches * epoch + i)

                writer.add_scalar('data/semantic_mIoU',
                                  semantic_mIoU,
                                  number_training_batches * epoch + i)

                # if objpart_mPrec is not None:

                  #   writer.add_scalar('data/obj_part_mPrec',
                  #                     objpart_mPrec,
                  #                     number_training_batches * epoch + i)

                  #   writer.add_scalar('data/obj_part_mRec',
                  #                     objpart_mRec,
                  #                     number_training_batches * epoch + i)

                correct_points = 0.0
                num_points = 0.0

        # Validate and save if best model
        (curr_op_valscore,
         curr_sem_valscore,
         curr_op_gt_valscore) = validate(
             net, valset_loader, (objpart_labels, semantic_labels), compression_method)
        writer.add_scalar('validation/semantic_validation_mIoU',
                          curr_sem_valscore, epoch)
        writer.add_scalar('validation/objpart_validation_mIoU',
                          curr_op_valscore, epoch)
        writer.add_scalar('validation/overall_objpart_validation_accuracy',
                          curr_op_gt_valscore, epoch)
        
        # visualize
        # TODO: move this call to the validate() function
        if dump_output_images:
            validate_and_output_images(net, valset_loader, net.flat_map,
                    which=which_vis_type, alpha=0.7, writer=writer, step_num=epoch + 1,
                    save_name=save_model_name)

        is_best = False
        if curr_op_gt_valscore > best_op_gt_valscore:
            best_op_gt_valscore = curr_op_gt_valscore
            is_best = True
        if curr_op_valscore > best_op_val_score:
            best_op_val_score = curr_op_valscore
            # is_best = True
        if curr_sem_valscore > best_sem_val_score:
            best_sem_val_score = curr_sem_valscore
            # is_best = True

        # label as best IFF beats best obj-part inference score
        # Allow for equality (TODO check)
        tqdm.tqdm.write(
            "This epochs scores:\n\tSemantic:\t{}\n\tmOP:\t{}\n\tOP:\t{}".format(
                curr_sem_valscore, curr_op_valscore, curr_op_gt_valscore))
        tqdm.tqdm.write("Current Best Semantic Validation Score:\t{}".format(
            best_sem_val_score))
        tqdm.tqdm.write("Current Best objpart Validation Score:\t{}".format(
            best_op_val_score))
        tqdm.tqdm.write("Best  Overall objpart Validation Score:\t{}".format(
            best_op_gt_valscore))
        writer.export_scalars_to_json("./all_scalars.json")

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': save_model_name,
            'state_dict': net.state_dict(),
            'best_semantic_mIoU': best_sem_val_score,
            'best_objpart_mIoU': best_op_val_score,
            'best_objpart_accuracy': best_op_gt_valscore,
            'optimizer': optimizer.state_dict(),
        }, is_best, folder=save_model_folder,
                        filename='{}_epoch_{}.pth.tar'.format(save_model_name, epoch))


if __name__ == "__main__":
    # TODO: move parameters from hard_coded in fn to command_line variables
    print("[#](-1) DEBUG")

    parser = argparse.ArgumentParser(
        description='Hyperparameters for training.')
    parser.add_argument('-f', '--validate-batch-frequency',
                        metavar='F', type=int,
                        default=50,
                        help='Check training outputs every F batches.')
    parser.add_argument('-s', '--save-model-name', required=True,
                        type=str, help='prefix for model checkpoint')
    parser.add_argument('-l', '--load-model-name', default=None,
                        type=str, help="prefix for model checkpoint"
                        "to load")
    parser.add_argument('-df', '--dump-file-name', default=None,
                        type=str, help="name of dump file to load parameters from")
    parser.add_argument('-p', '--paint-images', type=str, default=None,
                        help="Type of masked images to output before"
                        "training (for debugging). One of:"
                        "['objpart', 'semantic', 'separated', 'None']")
    parser.add_argument('-v', '--validate-first', type=str2bool, default=False,
                        help="Whether or not to validate the loaded model"
                        "before training")
    parser.add_argument('-vo', '--validate-only', action='store_true',
                        help="*validate only* vs *validate + train* the loaded model")
    parser.add_argument('-d', '--device', type=str, default='0',
                        help="Which CUDA capable device on which to train")
    parser.add_argument('-w', '--num-workers', type=int, default=4,
                        help="Number of workers for the dataloader.")
    parser.add_argument('-e', '--epochs-to-train', type=int, default=10,
                        help="Number of epochs to train this time around.")
    parser.add_argument('-o', '--origin-epoch', type=int, required=False,
                        help="Epoch from which to originate. (default is "
                        "from checkpoint)")
    parser.add_argument('-m', '--mask-type', type=str, default="mode",
                        help="How to select the valid points for supervision."
                        " One of 'mode' or 'consensus'. ")
    parser.add_argument('-b', '--batch-size', type=int, default=12,
                        help="Number of inputs to process in each batch")
    parser.add_argument('-a', '--arch', '--architecture', type=str,
                        default='resnet', help='Which model to use')
    parser.add_argument('-n', '--experiment-name', type=str,
                        default='op', help='Experiment name for checkpointing')
    parser.add_argument(
        '-g',
        '--part-grouping',
        type=str,
        default='merged',
        help="Whether to predict for all parts, in part groups,"
        "or just for a binary obj/part task (for each class)."
        "Legal values are 'binary', 'merged', and 'sparse'")
    parser.add_argument('--load-folder', type=str, default='experiments')
    parser.add_argument('-nb', '--num-branches', type=int, 
        default=2, help="number of branches (output heads) for the network" \
        "[1-way or 2-way]")
    # parser.add_argument('-nop', '--num-objpart-classes', type=int, 
    #     default=2, help="# object/part's head outputs")
    parser.add_argument('-dd', '--dump-dir', type=str, default='dumps/',
        help="dump's directory path")
    parser.add_argument('-opw', '--objpart-weight', type=float, default=0.5,
        help="weight ([0, 1]) corresponding to objpart_loss part in the combined loss")
    parser.add_argument('-lt', '--loss-type', default='2_losses_combined',
        help="which type of loss function to use")
    parser.add_argument('-cm', '--compression-method', default="gt",
        help="compression method for compressing 41/61-way objpart logits into 2/3-way," \
             "for the purpose of evaluation")
    parser.add_argument('-di', '--dump-output-images', action='store_true',
        help="dump output images")
    
    argvals = parser.parse_args()
    main(argvals)
