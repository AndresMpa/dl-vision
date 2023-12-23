import time
import numpy as np

from torch import no_grad, max

from util.dataset import create_dataset

from util.architecture import get_single_model, get_loss_function
from util.architecture import create_device, create_transform

from util.draw import draw_error, draw_confusion_matrix

from util.logger import create_log_entry, send_message_to_os


def execute_eval():
    start_time = time.time()

    """
    Model definition
    """
    model, model_name, identifier = get_single_model()
    transform = create_transform(model_name)
    lost_criteria = get_loss_function()
    device = create_device()
    model.to(device)

    print(f"Using model ({identifier}) architecture:")
    print(model)

    """
    Dataset split
    """
    _, testloader, _, testset = create_dataset(transform, True)
    class_names = testset.classes

    """
    Setting eval function
    """
    model.eval()

    total = 0
    correct = 0
    total_loss = 0.0
    eval_losses = []
    true_labels = []
    predicted_labels = []

    with no_grad():
        for batch_idx, (input, target) in enumerate(testloader):
            input, target = input.to(device), target.to(device)

            outputs = model(input)

            loss = lost_criteria(outputs, target)
            total_loss += loss.item()
            eval_losses.append(loss.item())

            _, predicted = max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            true_labels.extend(target.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            # Print batch-level information
            if batch_idx % 10 == 0:
                batch = f"Batch [{batch_idx}/{len(testloader)}]"
                accuracy = f"Accuracy: {correct / total * 100:.2f}%"
                batch_loss = f"Batch Loss: {total_loss / (batch_idx + 1):.4f}"
                print(f"{batch} - {accuracy} {batch_loss}")

    # Print overall accuracy
    accuracy = correct / total
    overall_loss = total_loss / len(testloader)
    print(f"Accuracy on the sample dataset: {accuracy * 100:.2f}%")
    print(f"Overall Loss: {overall_loss:.4f}")

    """
    Measuring time
    """
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60.0
    print(f"Model {identifier} took: {elapsed_time:0.2f} minutes")

    timestamp = time.time()

    """
    Metrics
    """

    conf_matrix = np.zeros(
        (len(class_names), len(class_names)), dtype=np.int64)

    for label, prediction in zip(true_labels, predicted_labels):
        conf_matrix[label, prediction] += 1

    """
    Plotting
    """
    draw_confusion_matrix(conf_matrix, class_names, model_name, identifier)

    draw_error(eval_losses, identifier)

    '''
    Logs
    '''
    create_log_entry(timestamp, elapsed_time, model_name)
    send_message_to_os(
        f"Process ended; took {elapsed_time} minutes",
        f"{model_name}"
    )
