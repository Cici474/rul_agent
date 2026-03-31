import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_rmse(y_true, y_pred):
    """均方根误差 """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    """平均绝对误差 """
    return mean_absolute_error(y_true, y_pred)

def calculate_r2(y_true, y_pred):
    """R2 决定系数"""
    return r2_score(y_true, y_pred)

def calc_score(y_true, y_pred):
    """
    C-MAPSS 官方非对称评分函数
    - 晚预测 (d > 0，即预测寿命 > 实际寿命) 的惩罚呈 exp(d/10) 指数增长，模拟机毁人亡的高昂代价。
    - 早预测 (d < 0，即预测寿命 < 实际寿命) 的惩罚呈 exp(-d/13) 指数增长，模拟维护成本增加。
    """
    score = 0
    for t, p in zip(y_true, y_pred):
        d = p - t
        if d < 0:
            score += np.exp(-d / 13) - 1
        else:
            score += np.exp(d / 10) - 1
    return score