import os
import torch
import numpy as np
import random

SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 关键：确保卷积等算法使用确定性实现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/Users/cici/yan/dateset/CMAPSSData"
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_RUL = 125  #截断最大 RUL，防止极端值干扰训练和评估

BATCH_SIZE = 64
EPOCHS = 100
EARLY_STOP_PATIENCE = 30

WINDOW_SIZES = {"FD001": 30, "FD002": 30, "FD003": 30, "FD004": 30}
SETTINGS = ['setting_1', 'setting_2', 'setting_3']

FD001_SENSORS = [f's_{i}' for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
FD002_SENSORS = [f's_{i}' for i in [2,3,4,6,7,8,9,11,12,13,15,17,20,21]]
FD003_SENSORS = [f's_{i}' for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
FD004_SENSORS = [f's_{i}' for i in [2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]

DATASET_CONFIGS = {
    "FD001": {"sensors": FD001_SENSORS, "use_settings": False, "regime_norm": False},
    "FD003": {"sensors": FD003_SENSORS, "use_settings": False, "regime_norm": False},
    "FD002": {"sensors": FD002_SENSORS, "use_settings": True, "regime_norm": True},
    "FD004": {"sensors": FD004_SENSORS, "use_settings": True, "regime_norm": True}
}

LIGHTGBM_PARAMS = {
    'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05, 
    'n_jobs': -1, 'verbose': -1,
    'random_state': SEED  
}