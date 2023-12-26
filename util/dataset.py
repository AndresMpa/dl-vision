from torchvision.datasets import CelebA, CIFAR10, CIFAR100
from torch.utils.data import DataLoader

from util.dirs import check_path

from config.vars import env_vars


def get_dataset(transform, isTrain=True):
    """
    Returns a data set from torchvision (Could be extended)

    Args:
        transform (Object): A transform PyTorch pipeline
        isTrain (Bool): A flag to define if it's a test or not

    Returns:
        An instance of a dataset
    """
    dataset = env_vars.dataset
    path = env_vars.data_path
    download = (not check_path(env_vars.data_path))

    if dataset == "CelebA":
        return CelebA(path, isTrain, transform=transform, download=download)
    elif dataset == "CIFAR10":
        return CIFAR10(path, isTrain, transform=transform, download=download)
    elif dataset == "CIFAR100":
        return CIFAR100(path, isTrain, transform=transform, download=download)


def get_loader(dataset):
    """
    Creates a DataLoader using some data set, for each train
    or test data set

    Args:
        dataset (Object): A slide of information for training or testing

    Returns:
        A DataLoader object
    """
    return DataLoader(
        dataset,
        batch_size=env_vars.batch_size,
        shuffle=True,
        num_workers=2)


def create_dataset(transform, get_sets=False):
    """
    Generates a train and test set using some data set
    specified on .env file

    Args:
        transform (Object): A transform PyTorch pipeline

    Returns:
        trainloader (Tensor): An slice of the dataset used for training
        testloader (Tensor): An slice of the dataset used for testing
    """

    trainset = get_dataset(transform)
    testset = get_dataset(transform, False)

    trainloader = get_loader(trainset)
    testloader = get_loader(testset)

    if get_sets:
        return trainloader, testloader, trainset, testset
    else:
        return trainloader, testloader
