import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from config import *

warnings.filterwarnings('ignore')

class CMAPSSDataEngine:
    def __init__(self, df, sensors, window_size, use_settings=False, regime_norm=False, max_rul=125):
        self.df = df.copy()
        self.sensors = sensors
        self.window_size = window_size
        self.use_settings = use_settings
        self.regime_norm = regime_norm
        self.max_rul = max_rul  
        
        self.diff1_cols = [f"{s}_d1" for s in self.sensors]
        self.feature_cols = self.sensors + self.diff1_cols + (SETTINGS if use_settings else [])
        
        self.df[self.sensors] = self.df[self.sensors].astype(float)
        self.df['time_cycles'] = self.df['time_cycles'].astype(float)
        
        self.global_scaler = MinMaxScaler(feature_range=(0, 1))
        self.regime_scalers = {}
        self.setting_scaler = StandardScaler() if use_settings else None
        self.kmeans = None
        self.sensor_baselines = None 

        self._prepare_sota_features()

    def _prepare_sota_features(self):
        for unit_id in self.df['unit_nr'].unique():
            mask = self.df['unit_nr'] == unit_id
            
            #  按时间排序
            group_sorted = self.df.loc[mask].sort_values('time_cycles')
            df_sensor = pd.DataFrame(group_sorted[self.sensors].values)
            
            if self.regime_norm:
                smoothed = df_sensor.values
                self.df.loc[mask, self.sensors] = smoothed
                diff1 = df_sensor.ewm(span=10, adjust=False).mean().diff().ffill().fillna(0)
                self.df.loc[mask, self.diff1_cols] = diff1.values
            else:
                smoothed = df_sensor.ewm(span=10, adjust=False).mean().values
                # 将排序后的平滑值写回原 DataFrame (利用原始 index 对齐)
                self.df.loc[group_sorted.index, self.sensors] = smoothed
                df_smoothed = pd.DataFrame(smoothed)
                diff1 = df_smoothed.ewm(span=5, adjust=False).mean().diff().ffill().fillna(0)
                self.df.loc[group_sorted.index, self.diff1_cols] = diff1.values

    def plot_sensor_correlation_heatmap(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(12, 10))
        corr = self.df[self.sensors].corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
        plt.title('Sensor Correlation Heatmap (Train Set)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "Sensor_Correlation_Heatmap.png"), dpi=300)
        plt.close()

    def fit_scaler(self, train_units):
        train_df = self.df[self.df['unit_nr'].isin(train_units)].copy()
        
        baselines = []
        for unit_id in train_units:
            unit_data = train_df[train_df['unit_nr'] == unit_id].sort_values('time_cycles')[self.sensors].values
            baselines.append(unit_data[:10].mean(axis=0))
        self.sensor_baselines = np.nanmean(baselines, axis=0)
        
        if self.regime_norm and self.use_settings:
            scaled_settings = self.setting_scaler.fit_transform(train_df[SETTINGS].values)
            self.kmeans = KMeans(n_clusters=6, n_init=10, random_state=SEED).fit(scaled_settings)
            train_df['regime'] = self.kmeans.labels_
            for r in range(6):
                reg_data = train_df[train_df['regime'] == r][self.feature_cols]
                if not reg_data.empty:
                    self.regime_scalers[r] = StandardScaler().fit(reg_data.values) 
        else:
            self.global_scaler.fit(train_df[self.feature_cols].values)

    def extract_features_safe(self, target_units=None):
        X_tensors, Macro_features, Y_rul, IDs = [], [], [], []
        proc_df = self.df[self.df['unit_nr'].isin(target_units)].copy() if target_units is not None else self.df.copy()
        
        if self.regime_norm and self.use_settings:
            scaled_settings = self.setting_scaler.transform(proc_df[SETTINGS].values)
            proc_df['regime'] = self.kmeans.predict(scaled_settings)
            normalized_features = np.zeros((len(proc_df), len(self.feature_cols)))
            for r in range(6):
                mask = proc_df['regime'] == r
                if mask.any():
                    normalized_features[mask] = self.regime_scalers[r].transform(proc_df.loc[mask, self.feature_cols].values)
            proc_df[self.feature_cols] = normalized_features
        else:
            proc_df[self.feature_cols] = self.global_scaler.transform(proc_df[self.feature_cols].values)

        x_idx = np.arange(self.window_size)
        
        for unit_id in proc_df['unit_nr'].unique():
            group = proc_df[proc_df['unit_nr'] == unit_id].sort_values('time_cycles')
            vals_normalized = group[self.feature_cols].values
            smoothed_sensors = group[self.sensors].values
            
            max_c = group['time_cycles'].max()
            ruls = (max_c - group['time_cycles']).values
            piecewise_ruls = np.clip(ruls, a_min=None, a_max=self.max_rul)

            # 加上 +1，不丢弃 RUL=0 的最后一个窗口
            for i in range(len(group) - self.window_size + 1):
                window_normalized = vals_normalized[i : i + self.window_size].copy()
                window_sensors = smoothed_sensors[i : i + self.window_size, :]
                
                # 提取统计与物理特征
                mean_vals = window_sensors.mean(axis=0).mean()
                std_vals = window_sensors.std(axis=0).mean()
                max_vals = window_sensors.max(axis=0).mean()
                min_vals = window_sensors.min(axis=0).mean()
                last_vals = window_sensors[-1, :].mean()
                
                # 防 NaN 处理
                corr_mat = np.corrcoef(window_sensors.T)
                if np.any(np.isnan(corr_mat)):
                    corr_mat = np.nan_to_num(corr_mat, nan=0.0)
                mask_corr = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
                mean_corr = corr_mat[mask_corr].mean() if mask_corr.any() else 0.0
                
                # AUC (退化能量累积)
                auc = np.trapezoid(window_sensors, axis=0).mean()
                
                # 趋势斜率与方差 
                slopes = [np.polyfit(x_idx, window_sensors[:, s], 1)[0] for s in range(window_sensors.shape[1])]
                slope_std = np.std(slopes)
                global_trend_slope = np.mean(np.abs(slopes))
                global_variance = np.mean(np.var(window_sensors, axis=0))
                
                absolute_dev = window_sensors[-1, :] - self.sensor_baselines
                mean_abs_dev = np.mean(np.abs(absolute_dev))
                
                # 1. RMS Energy (信号退化能量)
                rms_energy = np.sqrt(np.mean(window_sensors**2))
                # 2. Kurtosis (瞬态冲击指数/峰度)
                var_vals = np.var(window_sensors, axis=0) + 1e-8
                kurtosis = np.mean(np.mean((window_sensors - np.mean(window_sensors, axis=0))**4, axis=0) / (var_vals**2))
                
                macro = [
                    global_trend_slope,   # 0: 全局斜率
                    global_variance,      # 1: 全局方差 
                    mean_abs_dev,         # 2: 绝对偏差
                    mean_vals,            # 3: 均值
                    std_vals,             # 4: 标准差
                    max_vals,             # 5: 最大值
                    min_vals,             # 6: 最小值
                    last_vals,            # 7: 最新值
                    mean_corr,            # 8: 传感器相关性
                    auc,                  # 9: 曲线下面积
                    slope_std,            # 10: 斜率标准差
                    kurtosis,             # 11: 峰度 
                    rms_energy            # 12: RMS 能量 
                ]
                macro = np.nan_to_num(macro, nan=0.0, posinf=0.0, neginf=0.0)
                
                if np.any(np.isnan(window_normalized)):
                    window_normalized = np.nan_to_num(window_normalized)
                
                X_tensors.append(window_normalized)
                Macro_features.append(macro)
                Y_rul.append(piecewise_ruls[i + self.window_size - 1])
                IDs.append(unit_id)
                
        return np.array(X_tensors), np.array(Macro_features), np.array(Y_rul), np.array(IDs)

    def extract_test_features(self):
        X_tensors, Macro_features, IDs = [], [], []
        proc_df = self.df.copy()
        
        if self.regime_norm and self.use_settings:
            scaled_settings = self.setting_scaler.transform(proc_df[SETTINGS].values)
            proc_df['regime'] = self.kmeans.predict(scaled_settings)
            normalized_features = np.zeros((len(proc_df), len(self.feature_cols)))
            for r in range(6):
                mask = proc_df['regime'] == r
                if mask.any():
                    normalized_features[mask] = self.regime_scalers[r].transform(proc_df.loc[mask, self.feature_cols].values)
            proc_df[self.feature_cols] = normalized_features
        else:
            proc_df[self.feature_cols] = self.global_scaler.transform(proc_df[self.feature_cols].values)

        x_idx = np.arange(self.window_size)

        for unit_id in proc_df['unit_nr'].unique():
            group = proc_df[proc_df['unit_nr'] == unit_id].sort_values('time_cycles')
            vals_normalized = group[self.feature_cols].values
            smoothed_sensors = group[self.sensors].values
            
            if len(group) >= self.window_size:
                window_normalized = vals_normalized[-self.window_size:].copy()
                window_sensors = smoothed_sensors[-self.window_size:, :]
            else:
                pad_len = self.window_size - len(group)
                window_normalized = np.pad(vals_normalized, ((pad_len, 0), (0, 0)), mode='edge')
                window_sensors = np.pad(smoothed_sensors, ((pad_len, 0), (0, 0)), mode='edge')
            
            mean_vals = window_sensors.mean(axis=0).mean()
            std_vals = window_sensors.std(axis=0).mean()
            max_vals = window_sensors.max(axis=0).mean()
            min_vals = window_sensors.min(axis=0).mean()
            last_vals = window_sensors[-1, :].mean()
            
            corr_mat = np.corrcoef(window_sensors.T)
            if np.any(np.isnan(corr_mat)):
                corr_mat = np.nan_to_num(corr_mat, nan=0.0)
            mask_corr = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
            mean_corr = corr_mat[mask_corr].mean() if mask_corr.any() else 0.0
            
            auc = np.trapezoid(window_sensors, axis=0).mean()
            
            slopes = [np.polyfit(x_idx, window_sensors[:, s], 1)[0] for s in range(window_sensors.shape[1])]
            slope_std = np.std(slopes)
            global_trend_slope = np.mean(np.abs(slopes))
            global_variance = np.mean(np.var(window_sensors, axis=0))
            absolute_dev = window_sensors[-1, :] - self.sensor_baselines
            mean_abs_dev = np.mean(np.abs(absolute_dev))
            
            rms_energy = np.sqrt(np.mean(window_sensors**2))
            var_vals = np.var(window_sensors, axis=0) + 1e-8
            kurtosis = np.mean(np.mean((window_sensors - np.mean(window_sensors, axis=0))**4, axis=0) / (var_vals**2))
            
            macro = [
                global_trend_slope, global_variance, mean_abs_dev, mean_vals, 
                std_vals, max_vals, min_vals, last_vals, mean_corr, auc, slope_std,
                kurtosis, rms_energy
            ]
            macro = np.nan_to_num(macro, nan=0.0, posinf=0.0, neginf=0.0)
            
            if np.any(np.isnan(window_normalized)):
                window_normalized = np.nan_to_num(window_normalized)
            
            X_tensors.append(window_normalized)
            Macro_features.append(macro)
            IDs.append(unit_id)
            
        return np.array(X_tensors), np.array(Macro_features), np.array(IDs)