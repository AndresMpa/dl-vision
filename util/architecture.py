import torch.nn as nn
from torch import save, load, device
from torch.cuda import is_available
import torch.optim as optim

import torchvision.transforms as transforms

from config.vars import env_vars

from architectures.vgg_16 import VGG16
from architectures.vgg_19 import VGG19
from architectures.alex_net import AlexNet
from architectures.resnet import ResNet, ResidualBlock

from util.dirs import get_current_path, create_path, check_path
from util.dirs import create_dir, list_files, extract_file_name_data
from util.logger import expected_time


def get_architecture(arch=env_vars.net_arch):
    """
    Return the selected architecture using .env file

    Returns:
        Architecture class to use
    """
    if arch == "AlexNet":
        estimation = env_vars.iterations * 0.0776
        msg = f"Using {arch}: \t \
            Estimated waiting time, around {estimation:0.2f} hrs"
        expected_time(msg)
        return AlexNet()
    elif arch == "VGG16":
        estimation = env_vars.iterations * 0.12011
        msg = f"Using {arch}: \t \
            Estimated waiting time, around {estimation:0.2f} hrs"
        expected_time(msg)
        return VGG16()
    elif arch == "VGG19":
        estimation = env_vars.iterations * 0.65135
        msg = f"Using {arch}: \t \
            Estimated waiting time, around {estimation:0.2f} hrs"
        expected_time(msg)
        return VGG19()
    elif arch == "ResNet":
        estimation = env_vars.iterations * 0.0416667
        msg = f"Using {arch}: \t \
            Estimated waiting time, around {estimation:0.2f} hrs"
        expected_time(msg)
        return ResNet(ResidualBlock, [2, 2, 2, 2])
    else:
        # Handle the case where the architecture is not recognized
        raise ValueError(f"Unsupported architecture: {arch}")


def get_device():
    """
    Define the available device to run the architecture, "CPU" or "GPU"

    Returns:
        A flag as a string "CPU" or "GPU" depending on which one is available
    """
    return ("cuda:0" if is_available() and env_vars.use_cuda else "cpu")


def create_device():
    """
    Creates an instance of a device using CPU or GPU depending on get_device()
    output

    Returns:
        An instance of device
    """
    print(f'Using {get_device()} for processing')
    return device(get_device())


def create_architecture():
    """
    Creates a new model using the given architecture
    using GPU when possible of CPU by default

    Returns:
        architecture (Object): An instance of the architecture and
        device (Object): The device used (GPU or CPU)
    """
    net_arch = get_architecture()

    if net_arch is None:
        # Handle the case where the architecture class is not found
        raise ValueError("Architecture class is None")

    arch_device = create_device()
    architecture = net_arch

    architecture.to(arch_device)

    return architecture, arch_device


def create_transform(arch=env_vars.net_arch):
    """
    Creates a transform using some specific pipeline

    Returns:
        A transforms pipeline composed
    """
    if arch == "ResNet":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(env_vars.img_size),
            transforms.ToTensor(),
        ])


def get_loss_function(criteria=env_vars.lost_criteria):
    """
    Create a new instance of a loss functions using some criteria from
    .env file

    Returns:
        A loss function using .env file vars to define
        on of them
    """
    if (criteria == "CrossEntropyLoss"):
        return nn.CrossEntropyLoss()
    elif (criteria == "BCELoss"):
        return nn.BCELoss()


def create_optimizer(architecture):
    """
    Creates an optimizer using hyper parameters from
    .env file

    Returns:
        An instance of an optimizer
    """
    return optim.SGD(
        architecture.parameters(),
        lr=env_vars.learning_rate,
        momentum=env_vars.momentum_value)


def save_arch(arch, timestamp):
    """
    Saves a model to handling with naming and paths creation

    Args:
        arch (Instance): Architecture instance or model to save
        timestamp: A time stamp used as id
    """
    net = env_vars.net_arch
    dataset = env_vars.dataset
    iterations = env_vars.iterations

    results = env_vars.models_path
    create_dir(results)

    file_name = f"{timestamp}_arch-{net}_{dataset}_iter-{iterations}"
    file_path = create_path(results, file_name + ".pth")

    print(f"Created file at: {file_path}")

    save(arch.state_dict(), file_path)


def get_model_to_use():
    """
    Create a path for a selected (By the console) model under ".pth" format

    Returns:
        A path to chosen model also it's index
    """
    models_path = env_vars.models_path

    if check_path(models_path):
        available_models = list_files(models_path, ".pth")

        if len(available_models) == 0:
            file_error = f"[ERROR]: Directory {models_path} is empty"
            raise Exception(file_error)

        print("Available Models:")
        for i, model in enumerate(available_models, 1):
            print(f"{i}. {model}")

        selected_model_index = int(
            input("Select a model (enter the corresponding number): ")) - 1

        return available_models, selected_model_index
    else:
        dir_error = f"[ERROR]: Directory {models_path} does not exist"
        raise Exception(dir_error)


def extract_model_data(available_models, selected_model_index):
    """
    Extracts information from a model to get its path, its name and its id

    Args:
        available_models (Array): A list of all found models at some directory
        selected_model_index (int): A number that represents chosen model id

    Returns:
        selected_model_path (str): Specific path to the choose model
        model_name (str): Architecture name
        model_identifier (str): Identifier as time stamp
    """
    models_path = env_vars.models_path

    selected_model_path = get_current_path(
        f"{models_path}" +
        f"/{available_models[selected_model_index]}")

    model_name = extract_file_name_data(
        available_models[selected_model_index], "arch-", "_")

    model_identifier = extract_file_name_data(
        available_models[selected_model_index], "", "_arch")

    return selected_model_path, model_name, model_identifier


def use_model(models_path, model_name, model_identifier):
    """
    Loads a model mapping it to an instance

    Args:
        selected_model_path (str): Specific path to the choose model
        model_name (str): Architecture name
        model_identifier (str): Identifier as time stamp

    Returns:
        model (Object): A instance of a pre-trained model
        model_name (Str): Model's name
        model_identifier (Str): A time stamp used as id
    """
    loaded_state_model = load(models_path, map_location=device(get_device()))

    architecture = get_architecture(model_name)
    model = architecture

    model.load_state_dict(loaded_state_model)

    return model, model_name, model_name + model_identifier


def get_single_model():
    """
    Creates a sample of a model
    """
    available, index = get_model_to_use()
    path, name, identifier = extract_model_data(available, index)

    return use_model(path, name, identifier)
