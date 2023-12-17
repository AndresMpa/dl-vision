# Deep Learning Vision Algorithms

## What is this?

You're seeing a computer vision test field, under the presented architecture, you can easily
try some architecture such as AlexNet, VGGx16, VGGx19 and ResNetx18

## Usage

### To run virtual environments

#### Windows

```bath
python -m venv env
env\Scripts\activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib python-dotenv plyer
```

#### Linux

```bash
python -m venv env
source env/bin/activate

pip install torch torchvision torchaudio
pip install numpy matplotlib python-dotenv plyer
```

### Setting environment

.env file is required to setting the environment of this project, along with the virtual environment
this file sets each section of the project. Here is a example

```bash
# Architecture
NET_ARCH=AlexNet
USE_CUDA=1

# Using model
MODELS_PATH="models"
USE_MODEL=1

# Dataset
DATASET="CIFAR10"
DATA_PATH="./data"
BATCH_SIZE=8

# Image management
IMG_SIZE=224
IMG_START_INDEX=0

# Training
ITERATIONS=1
LEARNING_RATE=0.01
MOMENTUM_VALUE=0.8
CATCH_INTERVAL=5

# Loss
LOST_CRITERIA="CrossEntropyLoss"

# Management
RESULTS_PATH="results"
LOG_PATH="log"
AUTOCLEAR=0
```

#### Available architectures

Just write any of the following on the NET_ARCH env var

- AlexNet
- VGG16
- VGG19
- ResNet

> USE_CUDA=1 means that the host can and will use CUDA by default it uses the processor

### Available dataset

Data sets can be defined inside the .env file in the $DATASET$ variable, available dataset
work for image detection or segmentation from PyTorch documentation specification follow this
[link](https://pytorch.org/vision/stable/datasets.html#image-detection-or-segmentation) to see
other options

| Keyword    | Size | Dataset                                                     |
| ---------- | ---- | ----------------------------------------------------------- |
| "CelebA"   | 200K | [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| "CIFAR10"  | 60K  | [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)        |
| "CIFAR100" | 60K  | [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)        |

> Note: The difference between CIFAR10 and CIFAR100 is the amount of classes, CIFAR10 contains 10
> while CIFAR100 contains 100 see "The CIFAR-100 dataset" specifications

`LOST_CRITERIA` means the lost function, available options are "BCELoss" and "CrossEntropyLoss"