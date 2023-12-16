from parser_images import get_args
from utils_test import *
import torch.nn as nn
import joblib
import math

args = get_args()

# Original data, 6% adversarial patch, 7% adversarial patch, 8% adversarial patch, 6% LaVAN patch
test_loader, test_patch6_loader, test_patch7_loader, test_patch8_loader, test_untarget_loader = get_loaders(args)

from models.vit import AFICL
from models.vit import vit_base_patch16_224


if args.dataset == "imagenet":
    # Vanilla ViT
    model0 = vit_base_patch16_224(num_classes=1000).cuda()
    model0 = nn.DataParallel(model0)

    # The proposed model trained on 6% adversarial patch
    model1 = AFICL(num_classes=1000).cuda()
    model1 = nn.DataParallel(model1)
    checkpoint = torch.load("./checkpoints/imagetnet/target/patch6/best")
    model1.load_state_dict(checkpoint['state_dict'])

    # The proposed model trained on 7% adversarial patch
    model2 = AFICL(num_classes=1000).cuda()
    model2 = nn.DataParallel(model2)
    checkpoint = torch.load("./checkpoints/imagetnet/target/patch7/best")
    model2.load_state_dict(checkpoint['state_dict'])

    # The proposed model trained on 8% adversarial patch
    model3 = AFICL(num_classes=1000).cuda()
    model3 = nn.DataParallel(model3)
    checkpoint = torch.load("./checkpoints/imagetnet/target/patch8/best")
    model3.load_state_dict(checkpoint['state_dict'])

    # The proposed model trained on 6% LaVAN patch
    model4 = AFICL(num_classes=1000).cuda()
    model4 = nn.DataParallel(model4)
    checkpoint = torch.load("./checkpoints/imagetnet/untarget/best")
    model4.load_state_dict(checkpoint['state_dict'])

elif args.dataset == "imagenette":
    # Vanilla ViT
    model0 = vit_base_patch16_224(num_classes=10).cuda()
    checkpoint = torch.load("./base_models/vit_base_patch16_224_imagenette.pth")
    model0.load_state_dict(checkpoint['state_dict'])
    model0 = nn.DataParallel(model0)

    # The proposed model trained on 6% adversarial patch
    model1 = AFICL(num_classes=10).cuda()
    model1 = nn.DataParallel(model1)
    checkpoint = torch.load("./checkpoints/imagetnette/target/patch6/best")
    model1.load_state_dict(checkpoint['state_dict'])

    # The proposed model trained on 7% adversarial patch
    model2 = AFICL(num_classes=10).cuda()
    model2 = nn.DataParallel(model2)
    checkpoint = torch.load("./checkpoints/imagetnette/target/patch7/best")
    model2.load_state_dict(checkpoint['state_dict'])

    # The proposed model trained on 8% adversarial patch
    model3 = AFICL(num_classes=10).cuda()
    model3 = nn.DataParallel(model3)
    checkpoint = torch.load("./checkpoints/imagetnette/target/patch8/best")
    model3.load_state_dict(checkpoint['state_dict'])

    # The proposed model trained on 6% LaVAN patch
    model4 = AFICL(num_classes=10).cuda()
    model4 = nn.DataParallel(model4)
    checkpoint = torch.load("./checkpoints/imagetnette/untarget/best")
    model4.load_state_dict(checkpoint['state_dict'])

elif args.dataset == "cifar":
    # Vanilla ViT
    model0 = vit_base_patch16_224(num_classes=10).cuda()
    checkpoint = torch.load("./base_models/vit_base_patch16_224_cifar.pth")
    model0.load_state_dict(checkpoint['state_dict'])
    model0 = nn.DataParallel(model0)

    # The proposed model trained on 6% adversarial patch
    model1 = AFICL(num_classes=10).cuda()
    model1 = nn.DataParallel(model1)
    checkpoint = torch.load("./checkpoints/cifar10/target/patch6/best")
    model1.load_state_dict(checkpoint['state_dict'])

    # The proposed model trained on 7% adversarial patch
    model2 = AFICL(num_classes=10).cuda()
    model2 = nn.DataParallel(model2)
    checkpoint = torch.load("./checkpoints/cifar10/target/patch7/best")
    model2.load_state_dict(checkpoint['state_dict'])

    # The proposed model trained on 8% adversarial patch
    model3 = AFICL(num_classes=10).cuda()
    model3 = nn.DataParallel(model3)
    checkpoint = torch.load("./checkpoints/cifar10/target/patch8/best")
    model3.load_state_dict(checkpoint['state_dict'])

    # The proposed model trained on 6% LaVAN patch
    model4 = AFICL(num_classes=10).cuda()
    model4 = nn.DataParallel(model4)
    checkpoint = torch.load("./checkpoints/cifar10/untarget/best")
    model4.load_state_dict(checkpoint['state_dict'])

elif args.dataset == "cifar100":
    # Vanilla ViT
    model0 = vit_base_patch16_224(num_classes=100).cuda()
    checkpoint = torch.load("./base_models/vit_base_patch16_224_cifar100.pth")
    model0.load_state_dict(checkpoint['state_dict'])
    model0 = nn.DataParallel(model0)

    # The proposed model trained on 6% adversarial patch
    model1 = AFICL(num_classes=100).cuda()
    model1 = nn.DataParallel(model1)
    checkpoint = torch.load("./checkpoints/cifar100/target/patch6/best")
    model1.load_state_dict(checkpoint['state_dict'])

    # The proposed model trained on 7% adversarial patch
    model2 = AFICL(num_classes=100).cuda()
    model2 = nn.DataParallel(model2)
    checkpoint = torch.load("./checkpoints/cifar100/target/patch7/best")
    model2.load_state_dict(checkpoint['state_dict'])

    # The proposed model trained on 8% adversarial patch
    model3 = AFICL(num_classes=100).cuda()
    model3 = nn.DataParallel(model3)
    checkpoint = torch.load("./checkpoints/cifar100/target/patch8/best")
    model3.load_state_dict(checkpoint['state_dict'])

    # The proposed model trained on 6% LaVAN patch
    model4 = AFICL(num_classes=100).cuda()
    model4 = nn.DataParallel(model4)
    checkpoint = torch.load("./checkpoints/cifar100/untarget/best")
    model4.load_state_dict(checkpoint['state_dict'])

else:
    raise ValueError("Dataset doesn't exist！")


def evaluate_natural_cifar(model, test_loader):
    model.eval()

    with torch.no_grad():
        meter = MultiAverageMeter()

        def test_step(step, X_batch, y_batch):
            X, y = X_batch.cuda(), y_batch.cuda()

            with open("./dump/patch_adv_vit_base_patch16_224_" + args.dataset + "/val/" + str(step * 20) + "_patch_adv_list_55.z", 'rb') as f:
                imgs = torch.from_numpy(joblib.load(f))

            for i in range(1, math.ceil(X.shape[0]/48)):
                with open("./dump/patch_adv_vit_base_patch16_224_" + args.dataset + "/val/" + str(
                        step * 20 + i) + "_patch_adv_list_55.z", 'rb') as f:
                    img = joblib.load(f)
                imgs = torch.cat([imgs, torch.from_numpy(img)], dim=0)

            _, output = model(imgs.cuda())
            meter.update('test_acc', (output.max(1)[1] == y).float().mean(), y.size(0))

        for step, (X_batch, y_batch) in enumerate(test_loader):
            test_step(step, X_batch, y_batch)
        print('Evaluation {}'.format(meter))


def evaluate_natural(model, test_loader):
    model.eval()

    with torch.no_grad():
        meter = MultiAverageMeter()

        def test_step(step, X_batch, y_batch):
            X, y = X_batch.cuda(), y_batch.cuda()
            if args.dataset == "imagenet":
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

            _, output = model(X)
            meter.update('test_acc', (output.max(1)[1] == y).float().mean(), y.size(0))

        for step, (X_batch, y_batch) in enumerate(test_loader):
            test_step(step, X_batch, y_batch)
        print('Evaluation {}'.format(meter))


evaluate_natural(model0, test_loader)
evaluate_natural(model0, test_patch6_loader)
evaluate_natural(model0, test_patch7_loader)
evaluate_natural(model0, test_patch8_loader)
evaluate_natural(model0, test_untarget_loader)
print("\n")

evaluate_natural(model1, test_loader)
evaluate_natural(model1, test_patch6_loader)
print("\n")

evaluate_natural(model2, test_loader)
evaluate_natural(model2, test_patch7_loader)
print("\n")

evaluate_natural(model3, test_loader)
evaluate_natural(model3, test_patch8_loader)
print("\n")

evaluate_natural(model4, test_loader)
if args.dataset == "cifar100" or args.dataset == "cifar":
    evaluate_natural_cifar(model4, test_untarget_loader)
else:
    evaluate_natural(model4, test_untarget_loader)



