from config.vars import env_vars


def get_activation_hook(name, activations):
    """
    Returns a hook function that stores the activation
    output in a dictionary

    Args:
        name (str): Name associated with the activation
        activations (dict): Dictionary to store activation outputs

    Returns:
        hook: Hook function to be used with register_forward_hook
    """
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook


def global_activation_hook(module, input, output):
    """
    Global hook function that stores the activation output in a global variable

    Args:
        module: Module associated with the forward hook
        input: Input tensor to the module
        output: Output tensor from the module
    """
    global activation
    activation = output.detach()


def normalize_activation(activation):
    """
    Normalize an activation into a numpy array

    Args:
        activation: Should be hook in a layer like activations["conv1"]

    Returns:
        activation_normalized: A normalized activation using a mathematical
        criteria
    """
    # Dims to avoid excess glare
    activation = activation[env_vars.img_start_index, :, :, :].cpu().numpy()

    numerator = activation - activation.min()
    denominator = activation.max() - activation.min()

    activation_normalized = numerator / denominator

    return activation_normalized
