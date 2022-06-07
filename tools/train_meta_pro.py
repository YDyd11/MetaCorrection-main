import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
from tensorboardX import SummaryWriter
import _init_paths
from evaluate_cityscapes import evaluate
from nets.deeplab_multi import DeeplabMulti
from nets.meta_deeplab_multi import Res_Deeplab
from nets.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from datasets.gta5_dataset import GTA5DataSet
from datasets.cityscapes_dataset import cityscapesPseudo
import datetime
import time
from datasets.dataset_prostate import *
from datasets.nci_isbi13_dataset import *
from torch import autograd

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'train_adv'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '/data12T/ydaugust/data/MRI_Prostate/Source'
DATA_LIST_PATH = '/data12T/ydaugust/code/domain_adaptation/MetaCorrection-main/datasets/decathlon_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '256,256'
DATA_DIRECTORY_TARGET = '/data12T/ydaugust/data/MRI_Prostate/Target'
DATA_LIST_PATH_TARGET = '/data12T/ydaugust/code/domain_adaptation/MetaCorrection-main/datasets/nci_list/pseudo.lst'  # redo √
INPUT_SIZE_TARGET = '256,256'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 3
NUM_STEPS = 250000
NUM_STEPS_STOP = 100000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = '/data12T/ydaugust/code/domain_adaptation/MetaCorrection-main/snapshots_medical/GTA5_best.pth'
SAVE_PRED_EVERY = 1000
WEIGHT_DECAY = 0.0005
LOG_DIR = '/data12T/ydaugust/code/domain_adaptation/MetaCorrection-main/log/medical_meta_debug'

LAMBDA_SEG = 0.1
GPU = '2'
TARGET = 'cityscapes'
SET = 'train'
T_WEIGHT = 0.11
IS_META = True
UPDATA_F = 1

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--is-meta", type=bool, default=IS_META,
                        help="Whether to update T")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--t-weight", type=float, default=T_WEIGHT,
                        help="grad weight to correct T.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--tensorboard", action='store_true', default=True, help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--gpu", type=str, default=GPU,
                        help="gpu id to run.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--update-f", type=int, default=UPDATA_F,
                        help="update frequency for T.")
    parser.add_argument("--uncertainty", type=bool, default=True,
                        help="choose adaptation set.")
    return parser.parse_args()


args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def build_model(args):

    net = Res_Deeplab(num_classes=args.num_classes)
    #print(net)

    if torch.cuda.is_available():
        net.cuda()
        torch.backends.cudnn.benchmark=True

    return net

def to_var(x, requires_grad=True):
    # 反向传播必要
    x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d(is_softmax=True).cuda()

    return criterion(pred, label)

def obtain_meta(source_img): # 找出source domain与target domain中相像的pixel
    #seg_model = DeeplabMulti(num_classes=19).cuda()
    seg_model = DeeplabMulti(num_classes=3).cuda()
    seg_model.load_state_dict(torch.load('/data12T/ydaugust/code/domain_adaptation/MetaCorrection-main/snapshots_medical/GTA5_best.pth'))
    #dis_model = FCDiscriminator(num_classes=19).cuda()
    dis_model = FCDiscriminator(num_classes=3).cuda()
    dis_model.load_state_dict(torch.load('/data12T/ydaugust/code/domain_adaptation/MetaCorrection-main/snapshots_medical/GTA5_best_D2.pth'))
    seg_model.eval()
    dis_model.eval()

    output1, output2 = seg_model(source_img)
    meta_map = dis_model(F.softmax(output2, dim=1)).cpu().data[0]
    source_like = torch.where(meta_map < 0.5)
    return source_like

def main():
    """Create the model and start the training."""
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.log_dir + '/result'):
        os.makedirs(args.log_dir + '/result')

    best_mIoU = 0
    mIoU = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    metaloader = data.DataLoader(
        Prostate(args.data_dir, args.data_list, mode='train', is_mirror=args.random_mirror,
                 is_pseudo=None, max_iter=args.num_steps * args.iter_size * args.batch_size),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    targetloader = data.DataLoader(nciisbiPseudo(args.data_dir_target, args.data_list_target,
                                                  mode='train', is_mirror=args.random_mirror,
                                                  is_pseudo=None,
                                                  max_iter=args.num_steps * args.iter_size * args.batch_size),
                                   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True)

    targetloader_iter = enumerate(targetloader)

    main_model = build_model(args)
    saved_state_dict = torch.load(args.restore_from)
    pretrained_dict = {k:v for k,v in saved_state_dict.items() if k in main_model.state_dict()} #k是name ，v是value。 打印k，v
    # pretrained_dict_list = ['layer5.conv2d_list.0.weight','layer5.conv2d_list.1.weight',
    # 'layer5.conv2d_list.2.weight','layer5.conv2d_list.3.weight']
    # for k in pretrained_dict_list:
    #     pretrained_dict[k] = pretrained_dict[k][:, :1024]
    main_model.load_state_dict(pretrained_dict,False)
# 为什么优化有问题? 因为模型太大了，一张卡加载不进来
    optimizer = optim.SGD(main_model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    interp = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        writer = SummaryWriter(args.log_dir)
    for i_iter in range(args.num_steps):
        if args.is_meta:
            main_model.train()
            l_f_meta = 0
            l_g_meta = 0
            l_f = 0
            optimizer.zero_grad()
            adjust_learning_rate(optimizer, i_iter)

            meta_net = Res_Deeplab(num_classes=args.num_classes)
            meta_net.load_state_dict(main_model.state_dict())
            meta_net.cuda()
            # optimizer_metanet = optim.SGD(meta_net.optim_parameters(args),
            #                      lr=1e-3, momentum=0.9, weight_decay=0.0005)
            # optimizer_metanet.zero_grad()

            _, batch = targetloader_iter.__next__()
            image, label, _, _ = batch
            image = to_var(image, requires_grad=False) #上传到cuda
            label = to_var(label, requires_grad=False)
            T1 = to_var(torch.eye(3, 3))
            T2 = to_var(torch.eye(3, 3))
            y_f_hat1, y_f_hat2 = meta_net(image) # 还是用resnet50 做的分割结果
            y_f_hat1 = torch.softmax(interp_target(y_f_hat1), dim=1).permute(0, 2, 3, 1).contiguous().view(-1, args.num_classes)
            y_f_hat2 = torch.softmax(interp_target(y_f_hat2), dim=1).permute(0, 2, 3, 1).contiguous().view(-1, args.num_classes)

            pre1 = torch.mm(y_f_hat1, T1).view(args.batch_size, h, w, args.num_classes).permute(0, 3, 1, 2)
            pre2 = torch.mm(y_f_hat2, T2).view(args.batch_size, h, w, args.num_classes).permute(0, 3, 1, 2)
            l_f_meta = loss_calc(pre2, label) + 0.1 * loss_calc(pre1, label)

            meta_net.zero_grad()
            # l_f_meta.backward()
            # torch.nn.utils.clip_grad_value_(meta_net.parameters(), clip_value=2)
            # optimizer_metanet.step()

            grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
            # torch.nn.utils.clip_grad_value_(grads, clip_value=2)
            meta_net.update_params(1e-3, source_params=grads)


            x_val, y_val, _, _ = next(iter(metaloader))
            x_val = to_var(x_val, requires_grad=False)
            y_val = to_var(y_val, requires_grad=False)
            meta_source = obtain_meta(x_val) # 训练的判别器，用于判别source domain中像target domain的pixel
            y_val[meta_source] = 255

            y_g_hat1, y_g_hat2 = meta_net(x_val)
            y_g_hat1 = torch.softmax(interp(y_g_hat1), dim=1)
            y_g_hat2 = torch.softmax(interp(y_g_hat2), dim=1)

            l_g_meta = loss_calc(y_g_hat2, y_val) + 0.1 * loss_calc(y_g_hat1, y_val)
            grad_eps1 = torch.autograd.grad(l_g_meta, T1, only_inputs=True, retain_graph=True,allow_unused=True)[0]
            grad_eps2 = torch.autograd.grad(l_g_meta, T2, only_inputs=True)[0]

            grad_eps1 = grad_eps1 / torch.max(grad_eps1)
            T1 = torch.clamp(T1-0.11*grad_eps1,min=0)
            # T1 = torch.softmax(T1, 1)
            norm_c = torch.sum(T1, 1)

            for j in range(args.num_classes):
                if norm_c[j] != 0:
                    T1[j, :] /= norm_c[j]

            grad_eps2 = grad_eps2 / torch.max(grad_eps2)
            T2 = torch.clamp(T2-0.11*grad_eps2,min=0)

            norm_c = torch.sum(T2, 1)

            for j in range(args.num_classes):
                if norm_c[j] != 0:
                    T2[j, :] /= norm_c[j]

            y_f_hat1, y_f_hat2 = main_model(image)
            y_f_hat1 = torch.softmax(interp_target(y_f_hat1), dim=1).permute(0, 2, 3, 1).contiguous().view(-1, args.num_classes)
            y_f_hat2 = torch.softmax(interp_target(y_f_hat2), dim=1).permute(0, 2, 3, 1).contiguous().view(-1, args.num_classes)
            pre1 = torch.mm(y_f_hat1, T1).view(args.batch_size, h, w, args.num_classes).permute(0, 3, 1, 2)
            pre2 = torch.mm(y_f_hat2, T2).view(args.batch_size, h, w, args.num_classes).permute(0, 3, 1, 2)

            l_f = loss_calc(pre2, label) + 0.1 * loss_calc(pre1, label)
            optimizer.zero_grad()
            l_f.backward()
            optimizer.step()

            if args.tensorboard:
                scalar_info = {
                    'loss_g_meta': l_g_meta.item(),
                    'loss_f_meta': l_f_meta.item(),
                    'loss_f': l_f.item(),
                }

                if i_iter % 10 == 0:
                    for key, val in scalar_info.items():
                        writer.add_scalar(key, val, i_iter)

            print('exp = {}'.format(args.log_dir))
            print(
            'iter = {0:8d}/{1:8d}, loss_g_meta = {2:.3f} loss_f_meta = {3:.3f} loss_f = {4:.3f}'.format(
                i_iter, args.num_steps, l_g_meta.item(), l_f_meta.item(), l_f.item()))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(main_model.state_dict(), osp.join(args.log_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            break
        if i_iter % args.save_pred_every == 0 and i_iter > 0:
            now = datetime.datetime.now()
            print (now.strftime("%Y-%m-%d %H:%M:%S"), '  Begin evaluation on iter {0:8d}/{1:8d}  '.format(i_iter, args.num_steps))
            mIoU = evaluate(main_model, pred_dir=args.log_dir + '/result')
            writer.add_scalar('mIoU', mIoU, i_iter)
            print('Finish Evaluation: '+time.asctime(time.localtime(time.time())))
            if mIoU > best_mIoU:
                best_mIoU = mIoU
                torch.save(main_model.state_dict(), osp.join(args.log_dir, 'MetaCorrection_best.pth'))

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()