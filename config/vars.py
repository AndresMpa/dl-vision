from dotenv import load_dotenv
from types import SimpleNamespace
from torch.cuda import is_available

import os

load_dotenv()

env_vars = SimpleNamespace(
    # Net props
    net_arch=os.environ.get("NET_ARCH"),
    use_cuda=os.environ.get("USE_CUDA", is_available()) == "1",

    # Models
    models_path=os.environ.get("MODELS_PATH", "models"),
    use_model=os.environ.get("USE_MODEL", False) == "1",

    # Data
    dataset=os.environ.get("DATASET"),
    data_path=os.environ.get("DATA_PATH", "./data"),
    batch_size=int(os.environ.get("BATCH_SIZE")),

    # Image props
    img_size=int(os.environ.get("IMG_SIZE")),
    img_start_index=int(os.environ.get("IMG_START_INDEX")),

    # Hyper parameters
    iterations=int(os.environ.get("ITERATIONS")),
    learning_rate=float(os.environ.get("LEARNING_RATE")),
    momentum_value=float(os.environ.get("MOMENTUM_VALUE")),
    catch_interval=int(os.environ.get("CATCH_INTERVAL", 200)),

    # Loss
    lost_criteria=os.environ.get("LOST_CRITERIA"),

    # Results directory
    autoclear=os.environ.get("AUTOCLEAR") == "1",
    results_path=os.environ.get("RESULTS_PATH", "results"),
    log_path=os.environ.get("LOG_PATH", "log"),
)
