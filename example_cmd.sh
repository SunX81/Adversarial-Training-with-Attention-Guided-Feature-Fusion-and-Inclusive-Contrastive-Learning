# ImageNet
# Adversarial Patch
python train_images.py --dataset imagenet --data-dir ./data/ImageNet/ --kind patch6 --num_classes 1000
python train_images.py --dataset imagenet --data-dir ./data/ImageNet/ --kind patch7 --num_classes 1000
python train_images.py --dataset imagenet --data-dir ./data/ImageNet/ --kind patch8 --num_classes 1000
# LaVAN
python train_images.py --dataset imagenet-un --data-dir ./data/ImageNet/ --kind patch6 --num_classes 1000

python test_images.py --dataset imagenet --data-dir ./data/ImageNet/


# ImageNette
# Adversarial Patch
python train_images.py --dataset imagenette --data-dir ./data/ImageNette/ --kind patch6 --num_classes 10
python train_images.py --dataset imagenette --data-dir ./data/ImageNette/ --kind patch7 --num_classes 10
python train_images.py --dataset imagenette --data-dir ./data/ImageNette/ --kind patch8 --num_classes 10
# LaVAN
python train_images.py --dataset imagenette-un --data-dir ./data/ImageNette/ --kind patch6 --num_classes 10

python test_images.py --dataset imagenette --data-dir ./data/ImageNette/


# CIFAR-10
# Adversarial Patch
python train_images.py --dataset cifar --data-dir ./data/ --kind patch6 --num_classes 10
python train_images.py --dataset cifar --data-dir ./data/ --kind patch7 --num_classes 10
python train_images.py --dataset cifar --data-dir ./data/ --kind patch8 --num_classes 10
# LaVAN
python train_images.py --dataset cifar --data-dir ./data/ --kind patch6 --num_classes 10

python test_images.py --dataset cifar --data-dir ./data/


# CIFAR-100
# Adversarial Patch
python train_images.py --dataset cifar100 --data-dir ./data/ --kind patch6 --num_classes 100
python train_images.py --dataset cifar100 --data-dir ./data/ --kind patch7 --num_classes 100
python train_images.py --dataset cifar100 --data-dir ./data/ --kind patch8 --num_classes 100
# LaVAN
python train_images.py --dataset cifar100 --data-dir ./data/ --kind patch6 --num_classes 100

python test_images.py --dataset cifar100 --data-dir ./data/
