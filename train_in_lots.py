import subprocess
import platform

from lots.training import training_combinations
from util.dirs import get_current_path


def update_env_parameters(parameters):
    """
    Updates .env file with the corresponding parameters

    Args:
        - parameters (Dict): A dictionary of permutations for .env file
    """
    # Read the content of the .env file
    env_file_path = ".env"
    with open(env_file_path, 'r') as file:
        lines = file.readlines()

    # Iterate through each line and update the target parameters if found
    for i in range(len(lines)):
        for target_parameter, new_value in parameters.items():
            if lines[i].startswith(f"{target_parameter}="):
                lines[i] = f"{target_parameter}={new_value}\n"

    # Write the updated content back to the .env file
    with open(env_file_path, 'w') as file:
        file.writelines(lines)


# TODO: Check this
def initialize_env():
    """
    Initialize a virtual environment depending on current OS
    """
    env = get_current_path("env")
    os_name = platform.system()

    if os_name == "Windows":
        activate_cmd = f"{env}/Scripts/activate"
        subprocess.call(activate_cmd, shell=True)


initialize_env()
update_env_parameters({"USE_MODEL": "0"})

parameter_combinations = training_combinations
os_name = platform.system()

for parameters in parameter_combinations:
    update_env_parameters(parameters)

    subprocess.call(["python", "main.py"], shell=(os_name == "Windows"))
