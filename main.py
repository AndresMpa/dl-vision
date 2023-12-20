import multiprocessing

from stages.training import execute_training
from stages.eval import execute_eval

from config.vars import env_vars

from util.logger import clear_log

if __name__ == '__main__':
    clear_log()

    """
    Threads management
    """
    multiprocessing.freeze_support()

    if env_vars.use_model:
        execute_eval()
    else:
        execute_training()
