import os
import glob
import json
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

load_dotenv()
#  LLM 配置 
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = "https://api.deepseek.com/v1"

class DiagnosticAgentSystem:
    def __init__(self, experts, initial_weights=None):
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        self.experts = experts 
        
        # 状态记忆机制 
        if initial_weights is None:
            self.last_weights = [0.25, 0.25, 0.25, 0.25]
        else:
            self.last_weights = initial_weights

    def diagnose_engine_window(self, raw_x, macro_f):
        """语义感知的专家混合决策 (带平滑降级机制)"""
        with torch.no_grad():
            t_x = torch.FloatTensor(raw_x).unsqueeze(0).to(DEVICE)
            # 使用 Expert_LightGBM 包装器进行推理，正确传入 X 和 Macro_X，保持特征维度为 143
            p_lgb = self.experts['LightGBM'].forward(raw_x[np.newaxis, ...], macro_f[np.newaxis, ...])[0]
            p_tcn = self.experts['TCN'](t_x).item()
            p_lstm = self.experts['BiLSTM'](t_x).item()
            p_trans = self.experts['Transformer'](t_x).item() 
        
        preds = [p_lgb, p_tcn, p_lstm, p_trans]

        prompt = f"""
        [Aero-engine Diagnosis]
        Sensor Trend: {macro_f[0]:.6f}, Variance: {macro_f[1]:.6f}
        Expert Preds: [LGBM: {p_lgb:.1f}, TCN: {p_tcn:.1f}, BiLSTM: {p_lstm:.1f}, Transformer: {p_trans:.1f}]
        Task: Assign weights [w1, w2, w3, w4] based on trend stability.
        Return JSON: {{"weights": [w1, w2, w3, w4], "reasoning": "...", "stage": "..."}}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            res = json.loads(response.choices[0].message.content)
            weights = res.get("weights", self.last_weights)
            
            # 权重强制归一化处理
            weight_sum = sum(weights)
            if weight_sum > 0:
                weights = [w / weight_sum for w in weights]
            else:
                weights = self.last_weights
                
            self.last_weights = weights 
            final_rul = sum(w * p for w, p in zip(weights, preds))
            
            return final_rul, res.get("reasoning", "No reasoning provided.")
            
        except Exception as e:
            # 容灾降级，沿用之前的权重
            final_rul = sum(w * p for w, p in zip(self.last_weights, preds))
            fallback_msg = f"LLM Error: {str(e)} -> [Fallback] Using previous window weights: {self.last_weights}"
            return float(final_rul), fallback_msg

def get_latest_run_dir(dataset_name):
    ds_dir = os.path.join(MODEL_SAVE_DIR, dataset_name)
    if not os.path.exists(ds_dir): return None
    runs = [os.path.join(ds_dir, d) for d in os.listdir(ds_dir) if d.startswith('run_')]
    return max(runs, key=os.path.getctime) if runs else None

def evaluate_all_with_agent():
    print("="*80)
    print(" 启动 C-MAPSS 测试集引擎 (LLM Agent 专家融合评估)")
    print("="*80)

    for ds_name in ["FD001", "FD002", "FD003", "FD004"]:
        run_dir = get_latest_run_dir(ds_name)
        if not run_dir: continue
            
        print(f"\n[{ds_name}] 评估模型批次: {os.path.basename(run_dir)}")
        
        test_file = os.path.join(DATA_DIR, f"test_{ds_name}.txt")
        rul_file = os.path.join(DATA_DIR, f"RUL_{ds_name}.txt")
        stats_file = os.path.join(run_dir, "train_stats.pkl")
        
        if not (os.path.exists(test_file) and os.path.exists(rul_file) and os.path.exists(stats_file)):
            print(f"数据缺失，跳过 {ds_name}")
            continue

        cfg = DATASET_CONFIGS[ds_name]
        window_size = WINDOW_SIZES[ds_name]

        # 1. 读取并预处理数据
        test_df = pd.read_csv(test_file, sep=r'\s+', header=None)
        test_df.columns = ['unit_nr', 'time_cycles'] + SETTINGS + [f's_{i}' for i in range(1, 22)]
        
        y_true_raw = np.loadtxt(rul_file)
        y_true = np.clip(y_true_raw, a_min=None, a_max=MAX_RUL)

        engine = CMAPSSDataEngine(test_df, cfg['sensors'], window_size, cfg['use_settings'], cfg['regime_norm'], max_rul=MAX_RUL)
        saved_stats = joblib.load(stats_file)
        
        # 恢复归一化器、聚类器
        engine.global_scaler = saved_stats.get('global')
        engine.regime_scalers = saved_stats.get('regimes')
        engine.kmeans = saved_stats.get('kmeans')
        engine.setting_scaler = saved_stats.get('settings')
        # 防止 Test 时报错 NoneType
        engine.sensor_baselines = saved_stats.get('baselines') 

        X_test, Macro_test, _ = engine.extract_test_features()
        input_dim = X_test.shape[2]

        # 2. 加载四位专家模型
        experts = {}
        missing_models = False
        
        lgb_path = os.path.join(run_dir, "expert_lgb.pkl")
        if os.path.exists(lgb_path):
            # 使用包装器加载 LightGBM 权重
            lgb_raw_model = joblib.load(lgb_path)
            lgb_expert = Expert_LightGBM()
            lgb_expert.model = lgb_raw_model
            experts['LightGBM'] = lgb_expert
        else:
            print(f"  [Error] 缺少 LightGBM，无法进行融合评估。")
            continue

        for name, ModelClass in {"TCN": Expert_TCN, "BiLSTM": Expert_BiLSTM, "Transformer": Expert_Transformer}.items():
            pattern = os.path.join(run_dir, f"expert_{name.lower()}_*.pt")
            weight_files = glob.glob(pattern)
            if weight_files:
                model = ModelClass(input_dim, window_size).to(DEVICE)
                model.load_state_dict(torch.load(weight_files[0], map_location=DEVICE))
                model.eval()
                experts[name] = model
            else:
                missing_models = True
                print(f"  [Error] 缺少 {name} 模型，无法进行融合评估。")
                break
                
        if missing_models:
            continue

        # 3. 启动大模型 Agent 进行融合推理
        agent = DiagnosticAgentSystem(experts)
        agent_preds = []
        
        print(f"  --> 开始 LLM Agent 逐样本推理评估 (共 {len(X_test)} 个窗口)...")
        # 考虑到 API 网络请求耗时，引入进度条直观显示进度
        for i in tqdm(range(len(X_test)), desc=f"{ds_name} Inference", ncols=80):
            raw_x = X_test[i]
            macro_f = Macro_test[i]
            final_rul, reason = agent.diagnose_engine_window(raw_x, macro_f)
            agent_preds.append(final_rul)
            
        # 安全性截断 (防止超出最高阈值)
        agent_preds = np.clip(agent_preds, 0, MAX_RUL)
        
        # 4. 计算并打印评估指标
        rmse = calculate_rmse(y_true, agent_preds)
        score = calc_score(y_true, agent_preds)
        
        print("-" * 60)
        print(f" [Agent 融合] | Test RMSE: {rmse:.4f} | Test Score: {score:.2f}")
        print("-" * 60)

if __name__ == "__main__":
    evaluate_all_with_agent()