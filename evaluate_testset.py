import os
import glob
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from config import DATASET_CONFIGS, WINDOW_SIZES, DATA_DIR, MODEL_SAVE_DIR, SETTINGS, MAX_RUL, DEVICE
from data_engine import CMAPSSDataEngine
from metrics import calculate_rmse, calc_score
from expert_models import Expert_TCN, Expert_BiLSTM, Expert_Transformer, Expert_LightGBM

def plot_true_vs_pred(y_true, y_pred, model_name, ds_name, run_dir):
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    sorted_indices = np.argsort(y_true)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    plt.figure(figsize=(14, 7))
    plt.plot(y_true_sorted, 'k-', label='True RUL', linewidth=2.5)
    plt.plot(y_pred_sorted, 'tab:red', linestyle='--', label='Predicted RUL', alpha=0.8, linewidth=1.5)
    plt.title(f'Prediction Results on Test Set - {model_name} ({ds_name})', fontsize=16)
    plt.xlabel('Engine Units (Sorted by True RUL Descending)', fontsize=14)
    plt.ylabel('Remaining Useful Life (Cycles)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{model_name}_{ds_name}_Test_Line.png"), dpi=300)
    plt.close()

def plot_error_heatmap(y_true, dict_preds, ds_name, run_dir):
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    sorted_idx = np.argsort(y_true)
    
    error_matrix = []
    model_names = []
    for name, y_pred in dict_preds.items():
        abs_error = np.abs(y_true[sorted_idx] - y_pred[sorted_idx])
        error_matrix.append(abs_error)
        model_names.append(name)
        
    error_matrix = np.array(error_matrix)
    
    plt.figure(figsize=(14, 6))
    sns.heatmap(error_matrix, cmap='viridis', cbar_kws={'label': 'Absolute Prediction Error (Cycles)'})
    plt.yticks(ticks=np.arange(len(model_names)) + 0.5, labels=model_names, rotation=0, fontsize=12)
    plt.xlabel('Test Engine Units (Sorted by True RUL Ascending)', fontsize=12)
    plt.title(f'Prediction Error Distribution Heatmap - {ds_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{ds_name}_Error_Heatmap.png"), dpi=300)
    plt.close()

def evaluate_all():
    print("="*80)
    print("启动 C-MAPSS 测试集评估 ")
    print("="*80)

    for ds_name in ["FD001", "FD002", "FD003", "FD004"]:
        ds_dir = os.path.join(MODEL_SAVE_DIR, ds_name)
        if not os.path.exists(ds_dir): continue
        runs = [os.path.join(ds_dir, d) for d in os.listdir(ds_dir) if d.startswith('run_')]
        if not runs: continue
        run_dir = max(runs, key=os.path.getctime)
            
        print(f"\n[{ds_name}] 评估最新模型批次: {os.path.basename(run_dir)}")
        
        test_file = os.path.join(DATA_DIR, f"test_{ds_name}.txt")
        rul_file = os.path.join(DATA_DIR, f"RUL_{ds_name}.txt")
        stats_file = os.path.join(run_dir, "train_stats.pkl")
        
        if not (os.path.exists(test_file) and os.path.exists(rul_file) and os.path.exists(stats_file)):
            continue

        cfg = DATASET_CONFIGS[ds_name]
        window_size = WINDOW_SIZES[ds_name]

        test_df = pd.read_csv(test_file, sep=r'\s+', header=None)
        test_df.columns = ['unit_nr', 'time_cycles'] + SETTINGS + [f's_{i}' for i in range(1, 22)]
        
        y_true_raw = np.loadtxt(rul_file)
        y_true = np.clip(y_true_raw, a_min=None, a_max=MAX_RUL)

        engine = CMAPSSDataEngine(test_df, cfg['sensors'], window_size, cfg['use_settings'], cfg['regime_norm'], max_rul=MAX_RUL)
        saved_stats = joblib.load(stats_file)
        
        # 恢复所有训练期的参数
        engine.global_scaler = saved_stats.get('global')
        engine.regime_scalers = saved_stats.get('regimes')
        engine.kmeans = saved_stats.get('kmeans')
        engine.setting_scaler = saved_stats.get('settings')
        engine.sensor_baselines = saved_stats.get('baselines') 
        
        # 获取用于动态集成的安全边界
        var_min = saved_stats.get('var_min', 0.0)
        var_max = saved_stats.get('var_max', 1.0)

        X_test, Macro_test, _ = engine.extract_test_features()
        input_dim = X_test.shape[2]
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)

        all_model_preds = {}

        lgb_path = os.path.join(run_dir, "expert_lgb.pkl")
        if os.path.exists(lgb_path):
            lgb_expert = Expert_LightGBM()
            lgb_expert.model = joblib.load(lgb_path)
            preds_lgb = np.clip(lgb_expert.forward(X_test, Macro_test), 0, MAX_RUL)
            all_model_preds["LightGBM"] = preds_lgb
            print(f"  --> LightGBM    | Test RMSE: {calculate_rmse(y_true, preds_lgb):.4f} | Score: {calc_score(y_true, preds_lgb):.2f}")
            plot_true_vs_pred(y_true, preds_lgb, "LightGBM", ds_name, run_dir)

        for name, ModelClass in {"TCN": Expert_TCN, "BiLSTM": Expert_BiLSTM, "Transformer": Expert_Transformer}.items():
            pattern = os.path.join(run_dir, f"expert_{name.lower()}_*.pt")
            weight_files = glob.glob(pattern)
            if weight_files:
                model = ModelClass(input_dim, window_size).to(DEVICE)
                model.load_state_dict(torch.load(weight_files[0], map_location=DEVICE))
                model.eval()
                with torch.no_grad():
                    preds_dl = np.clip(model(X_test_tensor).cpu().numpy(), 0, MAX_RUL)
                all_model_preds[name] = preds_dl
                print(f"  --> {name.ljust(11)} | Test RMSE: {calculate_rmse(y_true, preds_dl):.4f} | Score: {calc_score(y_true, preds_dl):.2f}")
                plot_true_vs_pred(y_true, preds_dl, name, ds_name, run_dir)
        
        if len(all_model_preds) > 0:
            plot_error_heatmap(y_true, all_model_preds, ds_name, run_dir)

        if len(all_model_preds) == 4:
            ensemble_pred = np.zeros(len(y_true))
            
            macro_variance = Macro_test[:, 1]
            # 必须使用从 train_stats 中读取的 var_min 和 var_max
            norm_variance = (macro_variance - var_min) / (var_max - var_min + 1e-8)
            norm_variance = np.clip(norm_variance, 0.0, 1.0) # 约束在0-1之间
            # 只是模拟融合测试
            for i in range(len(y_true)):
                degradation_factor = norm_variance[i]
                
                if degradation_factor > 0.5:
                    # 晚期高危：TCN 对抖动衰减的追踪最可靠
                    w_tcn, w_lstm, w_trans, w_lgb = 0.50, 0.30, 0.10, 0.10
                elif degradation_factor > 0.2:
                    # 中期波动：均衡发挥
                    w_tcn, w_lstm, w_trans, w_lgb = 0.30, 0.30, 0.25, 0.15
                else:
                    # 早期健康：Transformer 全局注意力和 LGBM 统计特征更稳健
                    w_tcn, w_lstm, w_trans, w_lgb = 0.10, 0.15, 0.45, 0.30
                
                ensemble_pred[i] = (
                    all_model_preds["TCN"][i] * w_tcn +
                    all_model_preds["BiLSTM"][i] * w_lstm +
                    all_model_preds["Transformer"][i] * w_trans +
                    all_model_preds["LightGBM"][i] * w_lgb
                )
                
            ensemble_pred = np.clip(ensemble_pred, 0, MAX_RUL)
            ens_rmse = calculate_rmse(y_true, ensemble_pred)
            ens_score = calc_score(y_true, ensemble_pred)
            print(f"   [归纳式动态集成] | Test RMSE: {ens_rmse:.4f} | Score: {ens_score:.2f}\n")
            
            plot_true_vs_pred(y_true, ensemble_pred, "Ensemble", ds_name, run_dir)

if __name__ == "__main__":
    evaluate_all()