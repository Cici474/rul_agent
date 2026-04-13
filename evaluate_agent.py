import os
import glob
import torch
import joblib
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

from config import DATASET_CONFIGS, WINDOW_SIZES, DATA_DIR, MODEL_SAVE_DIR, SETTINGS, MAX_RUL, DEVICE
from data_engine import CMAPSSDataEngine
from metrics import calculate_rmse, calc_score
from expert_models import Expert_TCN, Expert_BiLSTM, Expert_Transformer, Expert_LightGBM
from agents.coordinator import MASCoordinator

load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")

def get_latest_run_dir(dataset_name):
    ds_dir = os.path.join(MODEL_SAVE_DIR, dataset_name)
    if not os.path.exists(ds_dir): return None
    runs = [os.path.join(ds_dir, d) for d in os.listdir(ds_dir) if d.startswith('run_')]
    return max(runs, key=os.path.getctime) if runs else None

def evaluate_all_with_agent():
    print("="*80)
    print(" [C-MAPSS Testing Engine: MAS Cognitive Consensus Evaluation]")
    print("="*80)

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    coordinator = MASCoordinator(client)

    for ds_name in ["FD001","FD002", "FD003", "FD004"]:
        run_dir = get_latest_run_dir(ds_name)
        if not run_dir: continue
            
        print(f"\n[{ds_name}] Evaluating checkpoint run: {os.path.basename(run_dir)}")
        test_file = os.path.join(DATA_DIR, f"test_{ds_name}.txt")
        rul_file = os.path.join(DATA_DIR, f"RUL_{ds_name}.txt")
        stats_file = os.path.join(run_dir, "train_stats.pkl")
        
        if not (os.path.exists(test_file) and os.path.exists(rul_file) and os.path.exists(stats_file)):
            continue

        cfg = DATASET_CONFIGS[ds_name]; window_size = WINDOW_SIZES[ds_name]
        test_df = pd.read_csv(test_file, sep=r'\s+', header=None)
        test_df.columns = ['unit_nr', 'time_cycles'] + SETTINGS + [f's_{i}' for i in range(1, 22)]
        y_true = np.clip(np.loadtxt(rul_file), a_min=None, a_max=MAX_RUL)

        engine = CMAPSSDataEngine(test_df, cfg['sensors'], window_size, cfg['use_settings'], cfg['regime_norm'], max_rul=MAX_RUL)
        saved_stats = joblib.load(stats_file)
        engine.global_scaler = saved_stats.get('global'); engine.regime_scalers = saved_stats.get('regimes')
        engine.kmeans = saved_stats.get('kmeans'); engine.setting_scaler = saved_stats.get('settings')
        engine.sensor_baselines = saved_stats.get('baselines') 

        X_test, Macro_test, _ = engine.extract_test_features()
        input_dim = X_test.shape[2]

        experts = {}
        lgb_path = os.path.join(run_dir, "expert_lgb.pkl")
        if os.path.exists(lgb_path):
            lgb_raw = joblib.load(lgb_path)
            experts['LightGBM'] = Expert_LightGBM(); experts['LightGBM'].model = lgb_raw

        for name, ModelClass in {"TCN": Expert_TCN, "BiLSTM": Expert_BiLSTM, "Transformer": Expert_Transformer}.items():
            pattern = os.path.join(run_dir, f"expert_{name.lower()}_*.pt")
            weight_files = glob.glob(pattern)
            if weight_files:
                model = ModelClass(input_dim, window_size).to(DEVICE)
                model.load_state_dict(torch.load(weight_files[0], map_location=DEVICE))
                model.eval(); experts[name] = model

        agent_preds = []; excel_data = []
        
        for i in tqdm(range(len(X_test)), desc=f"{ds_name} Inference", ncols=80):
            raw_x = X_test[i]; macro_f = Macro_test[i]
            
            with torch.no_grad():
                t_x = torch.FloatTensor(raw_x).unsqueeze(0).to(DEVICE)
                p_lgb = experts['LightGBM'].forward(raw_x[np.newaxis, ...], macro_f[np.newaxis, :11])[0]
                p_tcn = experts['TCN'](t_x).item()
                p_lstm = experts['BiLSTM'](t_x).item()
                p_trans = experts['Transformer'](t_x).item() 
            
            preds_dict = {"LightGBM": float(p_lgb), "TCN": float(p_tcn), "BiLSTM": float(p_lstm), "Transformer": float(p_trans)}
            
            # Agent进行会议和仲裁
            final_rul, logs = coordinator.conduct_debate_and_decide(preds_dict, macro_f)
            final_rul = np.clip(final_rul, 0, MAX_RUL); agent_preds.append(final_rul)
            
            latest_weights = coordinator.memory.cache[-1]['weights']
            
            excel_data.append({
                "Engine_Unit": i + 1, "True_RUL": y_true[i],
                "Slope": macro_f[0], "Variance": macro_f[1], "Kurtosis": macro_f[11], "RMS_Energy": macro_f[12],
                "Weight_LGB": latest_weights[0], "Weight_TCN": latest_weights[1], "Weight_LSTM": latest_weights[2], "Weight_Trans": latest_weights[3],
                "MAS_Final_RUL": final_rul, "Peer_Revision_Log": logs["Round2_Log"], "Arbitration_Logic": logs["Round3_Log"]
            })
            
        agent_preds = np.array(agent_preds)
        rmse = calculate_rmse(y_true, agent_preds)
        score = calc_score(y_true, agent_preds)
        
        print("\n" + "="*80)
        print(f"  [{ds_name} 子集评估完成]")
        print(f"    ├─ 最终 Test RMSE  : {rmse:.4f}")
        print(f"    └─ 最终 Test Score : {score:.2f}")
        print("="*80 + "\n")

        df_results = pd.DataFrame(excel_data)
        save_path = os.path.join(run_dir, f"{ds_name}_MAS_Scientific_Predictions.csv")
        df_results.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"  详细预测及各窗口的仲裁判词已导出至: {save_path}")

        # 保存指标报告文件
        metrics_save_path = os.path.join(run_dir, f"{ds_name}_MAS_Metrics.txt")
        with open(metrics_save_path, "w", encoding="utf-8") as f:
            f.write(f"=========================================\n")
            f.write(f"  Dataset: {ds_name}\n")
            f.write(f"  Evaluation Type: Multi-Agent System (MAS) Consensus Fusion\n")
            f.write(f"=========================================\n")
            f.write(f"  Test RMSE  : {rmse:.4f}\n")
            f.write(f"  Test Score : {score:.2f}\n")
            f.write(f"=========================================\n")

if __name__ == "__main__":
    evaluate_all_with_agent()