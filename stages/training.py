import time

from util.architecture import create_architecture, create_optimizer
from util.architecture import create_transform, get_loss_function, save_arch

from util.activation import get_activation_hook, global_activation_hook
from util.activation import normalize_activation

from util.draw import draw_views, draw_error

from util.dataset import create_dataset

from util.logger import create_log_entry, send_message_to_os

from config.vars import env_vars


def execute_training():
    start_time = time.time()

    """
    Architecture definition
    """
    architecture, device = create_architecture()
    transform = create_transform()

    """
    Hyper parameters
    """
    lost_criteria = get_loss_function()
    optimizator = create_optimizer(architecture)

    """
    Analytics
    """
    error = []

    """
    Dataset split
    """
    trainloader, _ = create_dataset(transform)

    """
    Hooks
    """
    activations = {}

    # Mapping between layer indices and names
    layer_mapping = {
        3: 'conv1',
    }

    # Register hooks based on the mapping
    for layer_index, layer_name in layer_mapping.items():
        hook_function = get_activation_hook(layer_name, activations)
        architecture.characteristic[layer_index].register_forward_hook(
            hook_function)

    # Register a global hook
    architecture.register_forward_hook(global_activation_hook)

    '''
    Trainig process
    '''
    for epochs in range(env_vars.iterations):
        iteration_lost = 0
        for j, i in enumerate(trainloader, 0):
            data_input, label = i
            data_input, label = data_input.to(device), label.to(device)

            optimizator.zero_grad()
            output = architecture(data_input)

            lost = lost_criteria(output, label)
            lost.backward()

            optimizator.step()
            iteration_lost += lost.item()

            if (epochs == 0 and j % env_vars.catch_interval == 0
                    or j % env_vars.catch_interval == 0 and j > 0):
                print(
                    f'Epoch: {epochs + 1} \t \
                    Iteration: {j:5d} \t \
                    Lost: {(iteration_lost / env_vars.catch_interval):.3f}')
                error.append(iteration_lost / env_vars.catch_interval)
                iteration_lost = 0

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60.0
    print(f"Took: {elapsed_time:0.2f} minutes")

    timestamp = time.time()

    # Move the model back to the CPU
    architecture.to("cpu")

    '''
    Visualization
    '''
    activation_normalized = normalize_activation(activations["conv1"])

    '''
    Plotting activations
    '''
    draw_views(activation_normalized, "conv1", timestamp)

    '''
    Plotting loss function
    '''
    # Move the error to the CPU before plotting
    error_cpu = [e for e in error]
    draw_error(error_cpu, timestamp)

    '''
    Saving model
    '''
    save_arch(architecture, timestamp)

    '''
    Logs
    '''
    create_log_entry(timestamp, elapsed_time)
    send_message_to_os(
        f"Process ended; took {elapsed_time} minutes",
        f"{env_vars.net_arch}"
    )
