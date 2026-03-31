import os
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

from config import WINDOW_SIZES, DATASET_CONFIGS, MODEL_SAVE_DIR, DATA_DIR, SETTINGS, BATCH_SIZE, EPOCHS, DEVICE, seed_everything, SEED, MAX_RUL
from data_engine import CMAPSSDataEngine
from metrics import calculate_rmse, calc_score
from expert_models import Expert_LightGBM, Expert_TCN, Expert_BiLSTM, Expert_Transformer

warnings.filterwarnings("ignore")

class OfficialScoreLoss(nn.Module):
    def __init__(self, max_clip=5.0):
        super().__init__()
        self.max_clip = max_clip

    def forward(self, pred, target):
        d = pred - target
        loss_early = torch.exp(torch.clamp(-d / 13.0, max=self.max_clip)) - 1.0
        loss_late = torch.exp(torch.clamp(d / 10.0, max=self.max_clip)) - 1.0
        loss = torch.where(d < 0, loss_early, loss_late)
        return loss.mean()

def train_nn_expert(model, X_train, y_train, X_val, y_val, base_save_path, name, ds_name="FD001"):
    print(f"\n    [模型训练] 正在启动专家: {name}")
    print(f"    {'-'*85}")
    print(f"    {'Epoch':<6} | {'Tr Loss':<8} | {'Val Loss':<8} | {'Tr RMSE':<8} | {'Val RMSE':<8} | {'Val Score':<10}")
    print(f"    {'-'*85}")
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), batch_size=BATCH_SIZE * 2, shuffle=False)

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    criterion = OfficialScoreLoss().to(DEVICE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

    patience = 25
    patience_counter = 0
    best_val_rmse = float('inf')
    best_val_score = float('inf')
    best_model_state = None
    
    history = {'train_loss': [], 'val_loss': [], 'train_rmse': [], 'val_rmse': [], 'val_score': []}
    
    for epoch in range(EPOCHS):
        model.train()
        train_preds, train_trues, train_losses = [], [], []
        
        for bx, by in train_loader:
            optimizer.zero_grad()
            preds = model(bx.to(DEVICE)).view(-1)
            loss = criterion(preds, by.view(-1).to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            
            train_losses.append(loss.item())
            preds_clipped = torch.clamp(preds, min=0, max=MAX_RUL)
            train_preds.extend(preds_clipped.detach().cpu().numpy())
            train_trues.extend(by.numpy())
            
        current_train_loss = np.mean(train_losses)
        current_train_rmse = calculate_rmse(np.array(train_trues), np.array(train_preds))

        model.eval()
        v_preds, v_trues, val_losses = [], [], []
        with torch.no_grad():
            for bx, by in val_loader:
                preds = model(bx.to(DEVICE)).view(-1)
                loss = criterion(preds, by.view(-1).to(DEVICE))
                val_losses.append(loss.item())
                
                preds_clipped = np.clip(preds.cpu().numpy(), a_min=0, a_max=MAX_RUL)
                v_preds.extend(preds_clipped)
                v_trues.extend(by.numpy())
        
        current_val_loss = np.mean(val_losses)
        current_val_rmse = calculate_rmse(np.array(v_trues), np.array(v_preds))
        current_val_score = calc_score(np.array(v_trues), np.array(v_preds))
        
        scheduler.step(current_val_rmse)
        
        history['train_loss'].append(current_train_loss)
        history['val_loss'].append(current_val_loss)
        history['train_rmse'].append(current_train_rmse)
        history['val_rmse'].append(current_val_rmse)
        history['val_score'].append(current_val_score)
        
        is_best = False
        if current_val_rmse < best_val_rmse:
            best_val_rmse = current_val_rmse
            best_val_score = current_val_score
            is_best = True
            patience_counter = 0  
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            
        mark = "[*]" if is_best else "   "
        print(f"    {mark} {epoch+1:03d}/{EPOCHS} | {current_train_loss:8.4f} | {current_val_loss:8.4f} | {current_train_rmse:8.4f} | {current_val_rmse:8.4f} | {current_val_score:8.2f}")
        
        if patience_counter >= patience:
            print(f"      [Early Stopping] 验证集 RMSE 连续 {patience} 轮未下降，提前结束训练。")
            break
            
    final_save_path = f"{base_save_path}_valRMSE_{best_val_rmse:.2f}_Score_{int(best_val_score)}.pt"
    if best_model_state is not None:
        torch.save(best_model_state, final_save_path)
    print(f"      [完成] 已保存最优模型 -> Val RMSE: {best_val_rmse:.4f}")

    plot_dir = os.path.join(os.path.dirname(base_save_path), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    epochs_range = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title(f'Official Score Loss Convergence - {name} ({ds_name})', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss Value (Log Scale)', fontsize=12)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{name}_{ds_name}_Loss_Curve.png"), dpi=300)
    plt.close()

def run_four_experts_pipeline():
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for ds_name in ["FD001", "FD002", "FD003", "FD004"]:
        print(f"\n" + "="*85 + f"\n启动系统: {ds_name} (跑批次: run_{run_timestamp})\n" + "="*85)
        cfg = DATASET_CONFIGS[ds_name]
        current_window_size = WINDOW_SIZES[ds_name] 
        
        save_dir = os.path.join(MODEL_SAVE_DIR, ds_name, f"run_{run_timestamp}")
        plot_dir = os.path.join(save_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        train_file = os.path.join(DATA_DIR, f"train_{ds_name}.txt")
        if not os.path.exists(train_file): continue

        df = pd.read_csv(train_file, sep=r'\s+', header=None)
        df.columns = ['unit_nr', 'time_cycles'] + SETTINGS + [f's_{i}' for i in range(1, 22)]
        
        unique_units = df['unit_nr'].unique()
        np.random.seed(SEED)  
        np.random.shuffle(unique_units)
        split_idx = int(len(unique_units) * 0.8) 
        train_units, val_units = unique_units[:split_idx], unique_units[split_idx:]
        
        engine = CMAPSSDataEngine(df, cfg['sensors'], current_window_size, cfg['use_settings'], cfg['regime_norm'], max_rul=MAX_RUL)
        engine.plot_sensor_correlation_heatmap(plot_dir)
        engine.fit_scaler(train_units) 
        
        X_t, Macro_t, y_t, _ = engine.extract_features_safe(target_units=train_units)
        X_v, Macro_v, y_v, _ = engine.extract_features_safe(target_units=val_units)
        
        # 提取训练集中方差特征的极值，提供给独立集成引擎
        train_macro_variance = Macro_t[:, 1]
        var_min = float(np.min(train_macro_variance))
        var_max = float(np.max(train_macro_variance))
        
        joblib.dump({
            'global': engine.global_scaler, 
            'regimes': engine.regime_scalers, 
            'kmeans': engine.kmeans, 
            'settings': engine.setting_scaler,
            'baselines': engine.sensor_baselines,
            'var_min': var_min,  # 记录训练集边界
            'var_max': var_max
        }, os.path.join(save_dir, "train_stats.pkl"))
        
        input_dim = X_t.shape[2]

        print(f"    [训练] 正在优化专家库成员: LightGBM (Input Dim: {input_dim})")
        lgb_expert = Expert_LightGBM()
        lgb_expert.fit(X_t, Macro_t, y_t)
        joblib.dump(lgb_expert.model, os.path.join(save_dir, "expert_lgb.pkl")) 
        
        v_preds = np.clip(lgb_expert.forward(X_v, Macro_v), a_min=0, a_max=MAX_RUL)
        print(f"      [*] LightGBM 训练完成 | Val RMSE: {calculate_rmse(y_v, v_preds):.4f} | Val Score: {calc_score(y_v, v_preds):.2f}")
        
        dl_experts = {
            "TCN": Expert_TCN(input_dim, current_window_size),
            "BiLSTM": Expert_BiLSTM(input_dim, current_window_size),
            "Transformer": Expert_Transformer(input_dim, current_window_size)
        }
        for name, model in dl_experts.items():
            train_nn_expert(model, X_t, y_t, X_v, y_v, os.path.join(save_dir, f"expert_{name.lower()}"), name, ds_name=ds_name)

if __name__ == "__main__":
    seed_everything(SEED) 
    run_four_experts_pipeline()