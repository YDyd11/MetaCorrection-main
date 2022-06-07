import argparse
import torch.nn as nn
from torch.utils import data, model_zoo
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
from tensorboardX import SummaryWriter
import _init_paths
from evaluate_cityscapes import evaluate
sys.path.append('/data12T/ydaugust/code/domain_adaptation/MetaCorrection-main')
from nets.deeplab_multi import DeeplabMulti
from nets.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from datasets.dataset_prostate import *
from datasets.nci_isbi13_dataset import *
import datetime
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '/data12T/ydaugust/data/MRI_Prostate/Source/'
DATA_LIST_PATH = '/data12T/ydaugust/code/domain_adaptation/MetaCorrection-main/datasets/decathlon_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '256,256'
DATA_DIRECTORY_TARGET = '/data12T/ydaugust/data/MRI_Prostate/Target/train/'
DATA_LIST_PATH_TARGET = '/data12T/ydaugust/code/domain_adaptation/MetaCorrection-main/datasets/nci_list/train.txt'
INPUT_SIZE_TARGET = '400,400'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 3
NUM_STEPS = 250000
NUM_STEPS_STOP = 100000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
#RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = '/data12T/ydaugust/code/domain_adaptation/MetaCorrection-main/snapshots_medical'
WEIGHT_DECAY = 0.0005
LOG_DIR = '/data12T/ydaugust/code/domain_adaptation/MetaCorrection-main/log_medical'

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
GAN = 'Vanilla'

TARGET = 'Target'
SET = 'train'

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
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
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
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action='store_true', default=True, help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
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


def main():
    """Create the model and start the training."""
    best_mIoU = 0
    mIoU = 0
    device = torch.device("cuda" if not args.cpu else "cpu")

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True

    # Create network
    if args.model == 'DeepLab':
        model = DeeplabMulti(num_classes=args.num_classes)  # num_classes=19,GTA5-->Cityscapes=19
        if args.restore_from[:4] == 'http' :  # model 中的3 4 23 3 是101的数据
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                # print i_parts
        model.load_state_dict(new_params, False)

    model.train()
    model.to(device)

    cudnn.benchmark = True

    # init D
    model_D1 = FCDiscriminator(num_classes=args.num_classes).to(device)  #鉴别器
    model_D2 = FCDiscriminator(num_classes=args.num_classes).to(device)  #鉴别器

    model_D1.train()
    model_D1.to(device)

    model_D2.train()
    model_D2.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    # trainloader = data.DataLoader(
    #      GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
    #                  crop_size=input_size,
    #                  scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
    #      batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    trainloader = data.DataLoader(
         Prostate(args.data_dir, args.data_list, mode='train', is_mirror=args.random_mirror,
                  is_pseudo=None, max_iter=args.num_steps * args.iter_size * args.batch_size),
         batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    # targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
    #                                                   max_iters=args.num_steps * args.iter_size * args.batch_size,
    #                                                   crop_size=input_size_target,
    #                                                   scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
    #                                                  set=args.set),
    #                                 batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    #                                 pin_memory=True)
    targetloader = data.DataLoader(nciisbiDataSet(args.data_dir_target, args.data_list_target,
                                                   mode='train', is_mirror=args.random_mirror,
                                                   is_pseudo=None, max_iter=args.num_steps * args.iter_size * args.batch_size),
                                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                    pin_memory=True)


    targetloader_iter = enumerate(targetloader)
    print("here")

    # implement model.optim_parameters(args) to handle different models' lr setting
# 把构建的卷积网络的参数优化
    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    # 用SGD优化方法优化
    optimizer.zero_grad() # 把模型参数梯度设置为0，清空过往梯度

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99)) #用adam算法
    optimizer_D2.zero_grad()

    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss() #多分类 需要把预测到的概率值通过sigmoid映射到0-1之间。 Σ[y_t * log(f(x_t))]
    elif args.gan == 'LS':
        bce_loss = torch.nn.MSELoss()
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255) # -Σ[p(x) * log(p(x)) + (1-p(x) * log(1-p(x)))]
    # 实际输出与期望输出的距离，当值越小，两个输出的分布越接近。

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True) # 上采样到 (720,1280)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)#上采样到（512,1024）

    # labels for adversarial training 训练一个domain predictor
    source_label = 0
    target_label = 1

    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        writer = SummaryWriter(args.log_dir)

    for i_iter in range(args.num_steps):
        print(i_iter)
        if i_iter==376:
            print("aaa")
        model.train() # 启用batch normalization 和 dropout，模型有BN层，所以需要加上
        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        optimizer.zero_grad() # 清空之前累计的梯度
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D1.zero_grad() # adam 清空过往的D1的梯度
        optimizer_D2.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter)
        adjust_learning_rate_D(optimizer_D2, i_iter)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False

            for param in model_D2.parameters():
                param.requires_grad = False

            # train with source
            _, batch = trainloader_iter.__next__()
            images, labels, _, _ = batch
            images = images.to(device) #把数据上传到gpu上运行
            labels = labels.long().to(device)

            pred1, pred2 = model(images) # 为什么要预测两个pred
            pred1 = interp(pred1)
            pred2 = interp(pred2)

            loss_seg1 = seg_loss(pred1, labels)  # input : pred1 1 X 3 X 512 X 512
            # target : 1 X 256 X 256
            loss_seg2 = seg_loss(pred2, labels)
            loss = loss_seg2 + args.lambda_seg * loss_seg1  # 为什么这个loss是这么计算的

            # proper normalization
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_value1 += loss_seg1.item() / args.iter_size
            loss_seg_value2 += loss_seg2.item() / args.iter_size

            # train with target
            _, batch = targetloader_iter.__next__()
            images, _, _ = batch
            images = images.to(device)

            pred_target1, pred_target2 = model(images)
            pred_target1 = interp_target(pred_target1)
            pred_target2 = interp_target(pred_target2)

            D_out1 = model_D1(F.softmax(pred_target1, dim=1))  # 用于训练domain predictor
            D_out2 = model_D2(F.softmax(pred_target2, dim=1))

            loss_adv_target1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))

            loss_adv_target2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))

            loss = args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target2 * loss_adv_target2
            loss = loss / args.iter_size
            loss.backward()
            loss_adv_target_value1 += loss_adv_target1.item() / args.iter_size
            loss_adv_target_value2 += loss_adv_target2.item() / args.iter_size

            # train D

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            for param in model_D2.parameters():
                param.requires_grad = True

            # train with source
            pred1 = pred1.detach()
            pred2 = pred2.detach() #共用数据

            D_out1 = model_D1(F.softmax(pred1, dim=1))
            D_out2 = model_D2(F.softmax(pred2, dim=1))

            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))

            loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))

            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.item()
            loss_D_value2 += loss_D2.item()

            # train with target
            pred_target1 = pred_target1.detach()
            pred_target2 = pred_target2.detach()

            D_out1 = model_D1(F.softmax(pred_target1, dim=1))
            D_out2 = model_D2(F.softmax(pred_target2, dim=1))

            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(target_label).to(device))

            loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(target_label).to(device))

            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.item()
            loss_D_value2 += loss_D2.item()

        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()

        if args.tensorboard:
            scalar_info = {
                'loss_seg1': loss_seg_value1,
                'loss_seg2': loss_seg_value2,
                'loss_adv_target1': loss_adv_target_value1,
                'loss_adv_target2': loss_adv_target_value2,
                'loss_D1': loss_D_value1,
                'loss_D2': loss_D_value2,
            }

            if i_iter % 10 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)

        # print('exp = {}'.format(args.snapshot_dir))
        # print(
        # 'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_adv1 = {4:.3f}, loss_adv2 = {5:.3f} loss_D1 = {6:.3f} loss_D2 = {7:.3f}'.format(
        #     i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D1.pth'))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D2.pth'))
            break
        #print(i_iter)
        if i_iter % args.save_pred_every == 0:
            now = datetime.datetime.now()
            print(now.strftime("%Y-%m-%d %H:%M:%S"), '  Begin evaluation on iter {0:8d}/{1:8d}  '.format(i_iter, args.num_steps))
            mIoU = evaluate(model)
            writer.add_scalar('mIoU', mIoU, i_iter)
            print('Finish Evaluation: '+time.asctime(time.localtime(time.time())))
            if mIoU > best_mIoU:
                best_mIoU = mIoU
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_best.pth'))
                torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_best_D1.pth'))
                torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_best_D2.pth'))

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()