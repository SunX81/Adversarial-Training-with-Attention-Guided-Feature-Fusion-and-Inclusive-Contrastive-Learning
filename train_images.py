import torch.nn as nn
import torch.nn.functional as F
import tqdm
import random
import numpy as np
from functools import partial
import timm
from timm.loss import SoftTargetCrossEntropy
from timm.data import Mixup
import joblib
from parser_images import get_args
from utils import *
from losses import *
import math

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

args = get_args()

args.out_dir = args.out_dir + "_" + args.dataset + "_" + args.model + "_" + args.method
args.out_dir = args.out_dir + "_kind_{}/".format(args.kind)
args.out_dir = args.out_dir + "att+con/"
print(args.out_dir)
os.makedirs(args.out_dir, exist_ok=True)

logfile = os.path.join(args.out_dir, 'log_{}.log'.format(args.kind))
file_handler = logging.FileHandler(logfile)
file_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)-12s %(message)s'))
logger.addHandler(file_handler)
logger.info(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

resize_size = args.resize
crop_size = args.crop

num_classes = args.num_classes

train_loader, train_patch_loader, test_loader, test_patch_loader = get_loaders(args)
print(args.model)


from models.vit import AFICL
model = AFICL(num_classes=num_classes, mlp_hidden=args.mlp_hidden, out_dim=args.out_dim).cuda()
model = nn.DataParallel(model)
logger.info('Model{}'.format(model))

model.train()

ICL_criterion = SupConLoss(temperature=args.temp)


# evaluate
def evaluate_natural(model, test_loader):
    model.eval()
    with torch.no_grad():
        meter = MultiAverageMeter()

        def test_step(step, X_batch, y_batch):
            X, y = X_batch.cuda(), y_batch.cuda()

            if args.dataset == "imagenet" or args.dataset == "imagenet-un":
                y[y == 0] = 407
                y[y == 1] = 436
                y[y == 2] = 444
                y[y == 3] = 468
                y[y == 4] = 479
                y[y == 5] = 511
                y[y == 6] = 555
                y[y == 7] = 561
                y[y == 8] = 569
                y[y == 9] = 609
                y[y == 10] = 612
                y[y == 11] = 627
                y[y == 12] = 654
                y[y == 13] = 656
                y[y == 14] = 660
                y[y == 15] = 661
                y[y == 16] = 665
                y[y == 17] = 670
                y[y == 18] = 671
                y[y == 19] = 675
                y[y == 20] = 705
                y[y == 21] = 717
                y[y == 22] = 734
                y[y == 23] = 751
                y[y == 24] = 757
                y[y == 25] = 779
                y[y == 26] = 803
                y[y == 27] = 817
                y[y == 28] = 829
                y[y == 29] = 864
                y[y == 30] = 867
                y[y == 31] = 870
                y[y == 32] = 874
                y[y == 33] = 880
                y[y == 34] = 919
                y[y == 35] = 920

            feat, output = model(X)
            meter.update('test_acc', (output.max(1)[1] == y).float().mean(), y.size(0))

        for step, (X_batch, y_batch) in enumerate(test_loader):
            test_step(step, X_batch, y_batch)
        logger.info('Evaluation {}'.format(meter))


# evaluate cifar
def evaluate_natural_cifar(model, test_loader, kind):
    model.eval()
    dataset = args.dataset.replace("-un", "")
    with torch.no_grad():
        meter = MultiAverageMeter()

        def test_step(step, X_batch, y_batch):
            X, y = X_batch.cuda(), y_batch.cuda()

            with open("./dump/patch_adv_vit_base_patch16_224_" + dataset + "/val/" + str(step * 20) + kind, 'rb') as f:
                imgs = torch.from_numpy(joblib.load(f))

            for i in range(1, math.ceil(X.shape[0]/48)):
                with open("./dump/patch_adv_vit_base_patch16_224_" + dataset + "/val/" + str(step * 20 + i) + kind, 'rb') as f:
                    img = joblib.load(f)
                imgs = torch.cat([imgs, torch.from_numpy(img)], dim=0)

            feat, output = model(imgs.cuda())
            meter.update('test_acc', (output.max(1)[1] == y).float().mean(), y.size(0))

        for step, (X_batch, y_batch) in enumerate(test_loader):
            test_step(step, X_batch, y_batch)
        logger.info('Evaluation {}'.format(meter))


# train
def train_adv(args, model, ds_train, adv_train, ds_test, adv_test, logger):

    train_loader, train_patch_loader, test_loader, test_patch_loader = ds_train, adv_train, ds_test, adv_test

    mixup_fn = None

    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.labelsmoothvalue, num_classes=num_classes)
    
    if mixup_active:
        AT_criterion = SoftTargetCrossEntropy()
    else:
        AT_criterion = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    def lr_schedule(t):
        if t < args.epochs - 5:
            return args.lr_max
        elif t < args.epochs - 2:
            return args.lr_max * 0.1
        else:
            return args.lr_max * 0.01

    start_epoch = 0

    if args.resume:
        path_checkpoint = args.resume
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        model.load_state_dict(checkpoint['state_dict'])  # 加载模型可学习参数
        opt.load_state_dict(checkpoint['opt'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch

    for epoch in tqdm.tqdm(range(start_epoch + 1, args.epochs + 1)):
        train_loss = 0
        train_acc = 0
        train_n = 0

        def train_step(X, Y, t, mixup_fn):
            model.train()

            X = X.cuda()
            Y = Y.cuda()
            y = Y
            if mixup_fn is not None:
                X, Y = mixup_fn(X, Y)

            bsz = int(y.shape[0] / 2)

            feat, output = model(X, con=True, att=True)

            AT_loss = AT_criterion(output, Y)

            f1, f2 = torch.split(feat, [bsz, bsz], dim=0)
            feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # 8, 2, 128
            y, _ = torch.split(y, [bsz, bsz], dim=0)
            ICL_loss = ICL_criterion(feat, y)

            loss = 0.5 * (ICL_loss + AT_loss)

            opt.zero_grad()
            (loss / args.accum_steps).backward()

            acc = (output.max(1)[1] == Y.max(1)[1]).float().mean()

            return loss, acc, Y

        if args.dataset == "imagenet-un" or args.dataset == "imagenette-un":
            X_adv_list = [X for step, (X, y) in enumerate(train_patch_loader)]

        for step, (X, y) in enumerate(train_loader):

            if args.dataset == "imagenet-un" or args.dataset == "imagenette-un":
                X_adv = X_adv_list[step]
                X = torch.cat([X, X_adv], dim=0)

            elif args.dataset == "cifar-un" or args.dataset == "cifar100-un":
                dataset = args.dataset.replace("-un", "")
                with open(
                        "./dump/patch_adv_vit_base_patch16_224_" + dataset + "/train/" + str(
                            step * 3) + "_patch_clean_list_55.z",
                        'rb') as f:
                    imgs_cle = torch.from_numpy(joblib.load(f))

                for i in range(1, math.ceil(X.shape[0] / 48)):
                    with open("./dump/patch_adv_vit_base_patch16_224_" + dataset + "/train/" + str(
                            step * 3 + i) + "_patch_clean_list_55.z", 'rb') as f:
                        img = joblib.load(f)
                    imgs_cle = torch.cat([imgs_cle, torch.from_numpy(img)], dim=0)

                with open(
                        "./dump/patch_adv_vit_base_patch16_224_" + dataset + "/train/" + str(
                            step * 3) + "_patch_adv_list_55.z",
                        'rb') as f:
                    imgs_adv = torch.from_numpy(joblib.load(f))

                for i in range(1, math.ceil(X.shape[0] / 48)):
                    with open("./dump/patch_adv_vit_base_patch16_224_" + dataset + "/train/" + str(
                            step * 3 + i) + "_patch_adv_list_55.z", 'rb') as f:
                        img = joblib.load(f)
                    imgs_adv = torch.cat([imgs_adv, torch.from_numpy(img)], dim=0)

                X = torch.cat([imgs_cle, imgs_adv], dim=0)

            else:
                X = torch.cat([X[0], X[1]], dim=0)

            if args.dataset == "imagenet" or args.dataset == "imagenet-un":
                y[y == 0] = 407
                y[y == 1] = 436
                y[y == 2] = 444
                y[y == 3] = 468
                y[y == 4] = 479
                y[y == 5] = 511
                y[y == 6] = 555
                y[y == 7] = 561
                y[y == 8] = 569
                y[y == 9] = 609
                y[y == 10] = 612
                y[y == 11] = 627
                y[y == 12] = 654
                y[y == 13] = 656
                y[y == 14] = 660
                y[y == 15] = 661
                y[y == 16] = 665
                y[y == 17] = 670
                y[y == 18] = 671
                y[y == 19] = 675
                y[y == 20] = 705
                y[y == 21] = 717
                y[y == 22] = 734
                y[y == 23] = 751
                y[y == 24] = 757
                y[y == 25] = 779
                y[y == 26] = 803
                y[y == 27] = 817
                y[y == 28] = 829
                y[y == 29] = 864
                y[y == 30] = 867
                y[y == 31] = 870
                y[y == 32] = 874
                y[y == 33] = 880
                y[y == 34] = 919
                y[y == 35] = 920

            Y = torch.cat([y, y], dim=0)

            batch_size = args.batch_size // args.accum_steps
            epoch_now = epoch - 1 + (step + 1) / len(train_loader)

            for t in range(args.accum_steps):
                X_ = X[t * batch_size:(t + 1) * batch_size].cuda()
                y_ = Y[t * batch_size:(t + 1) * batch_size].cuda()
                if len(X_) == 0:
                    break

                loss, acc, y = train_step(X, Y, (step + 1) / len(train_loader), mixup_fn)

                train_loss += loss.item() * y_.size(0)
                train_acc += acc.item() * y_.size(0)
                train_n += y_.size(0)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad()

            if (step + 1) % args.log_interval == 0 or step + 1 == steps_per_epoch:
                logger.info('Training epoch {} step {}/{}, lr {:.4f} loss {:.4f} acc {:.4f}'.format(
                    epoch, step + 1, len(train_loader),
                    opt.param_groups[0]['lr'],
                           train_loss / train_n, train_acc / train_n
                ))

            lr = lr_schedule(epoch_now)
            opt.param_groups[0].update(lr=lr)

        if args.dataset == "imagenet-un" or args.dataset == "imagenette-un":
            del X_adv_list

        # evaluate
        if args.dataset == "cifar-un" or args.dataset == "cifar100-un":
            kind = "_patch_clean_list_55.z"
            evaluate_natural_cifar(model, test_loader, kind)
            kind = "_patch_adv_list_55.z"
            evaluate_natural_cifar(model, test_loader, kind)
        else:
            evaluate_natural(model, test_loader)
            evaluate_natural(model, test_patch_loader)

        path = os.path.join(args.out_dir, 'checkpoint_{}'.format(epoch))
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'opt': opt.state_dict()}, path)
        logger.info('Checkpoint saved to {}'.format(path))


# train
train_adv(args, model, train_loader, train_patch_loader, test_loader, test_patch_loader, logger)

logger.info(args.out_dir)

# evaluate
if args.dataset == "cifar-un" or args.dataset == "cifar100-un":
    kind = "_patch_clean_list_55.z"
    evaluate_natural_cifar(model, test_loader, kind)
    kind = "_patch_adv_list_55.z"
    evaluate_natural_cifar(model, test_loader, kind)
else:
    evaluate_natural(model, test_loader)
    evaluate_natural(model, test_patch_loader)
