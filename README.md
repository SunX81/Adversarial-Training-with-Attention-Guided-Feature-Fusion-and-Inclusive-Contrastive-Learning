# Adversarial Training with Attention-Guided Feature Fusion and Inclusive Contrastive Learning
By Xiao Sun, Song Wang and Jucheng Yang

Code for "Adversarial Training with Attention-Guided Feature Fusion and Inclusive Contrastive Learning"

<img width="1040" alt="architecture" src="https://github.com/SunX81/Adversarial-Training-with-Attention-Guided-Feature-Fusion-and-Inclusive-Contrastive-Learning/assets/152178254/1cd7d75d-8711-48b0-bea9-b00db048eab8">

# Requirements
Experiments were done with PyTorch 1.11.0 and timm 0.5.4. 

The complete list of required packages are available in [requirement.txt](requirement.txt), and can be installed with pip install -r requirement.txt. The code should be compatible with newer versions of packages. 

# Datasets
[ImageNet](https://image-net.org/download.php)(ILSVRC2012) 

[ImageNette](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz)(Full size) 

[CIFAR-10/CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)

# Usage
Download data in Datasets to data/.

 [BaiduYun Drive(code: 9nah)](https://pan.baidu.com/s/1oG2c2fNU9-Xzmzsjq-BGiA)
 
Download base models from the BaiduYun Drive link and move them to base_models/.

(optional) Download adversarial examples of ImageNet, ImageNette, CIFAR-10 and CIFAR-100 generated using untargeted attack (LaVAN) from the BaiduYun Drive link.
ImageNet and ImageNette to data/.
CIFAR-10 and CIFAR-100 to dump/.

(optional) Download checkpoints from BaiduYun Drive link and move them to checkpoints/.

See [example_cmd.sh](example_cmd.sh) for example commands for running the code.CIFAR-10 and CIFAR-100

# Acknowlegements
Related Repositories:

[https://github.com/mo666666/When-Adversarial-Training-Meets-Vision-Transformers](https://github.com/mo666666/When-Adversarial-Training-Meets-Vision-Transformers)

[https://github.com/HobbitLong/SupContrast](https://github.com/HobbitLong/SupContrast)

[https://github.com/A-LinCui/Adversarial_Patch_Attack](https://github.com/A-LinCui/Adversarial_Patch_Attack)

[https://github.com/Ping-C/certifiedpatchdefense](https://github.com/Ping-C/certifiedpatchdefense)
