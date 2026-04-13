import numpy as np

class StateMemoryCache:
    def __init__(self, similarity_threshold=0.05, max_size=1000):
        self.cache = [] 
        # 阈值设得更低，要求很小的相似度才会命中 (0.05)
        self.threshold = similarity_threshold
        self.max_size = max_size

    def search_similar_state(self, current_macro, current_preds_dict=None):
        """
        检索高维状态空间中是否存在相似的“认知状态”。
        不仅比对物理特征，还可以比对底层基模型的输出分布。
        """
        if not self.cache:
            return None
        
        # 将当前的 11 维宏观特征进行 L2 范数归一化，消除量纲影响
        curr_norm = current_macro / (np.linalg.norm(current_macro) + 1e-8)
        
        for record in self.cache:
            cached_macro = record['macro_features']
            cache_norm = cached_macro / (np.linalg.norm(cached_macro) + 1e-8)
            
            # 1. 计算 11 维物理特征的欧氏距离（全面覆盖斜率、方差、极值、能量AUC、相关性等）
            macro_dist = np.linalg.norm(curr_norm - cache_norm)
            
            # 2. 计算底层 4 位专家给出的数值差异
            preds_dist = 0
            if current_preds_dict and 'base_preds' in record and record['base_preds'] is not None:
                curr_preds = np.array(list(current_preds_dict.values()))
                cached_preds = np.array(list(record['base_preds'].values()))
                # 预测值差异占比
                preds_dist = np.mean(np.abs(curr_preds - cached_preds) / (curr_preds + 1e-8))

            # 综合物理特征距离与模型认知距离
            total_dist = macro_dist + (0.5 * preds_dist if current_preds_dict else 0)

            # 只有当多维特征的综合差异小于阈值时，才判定为相同
            if total_dist < self.threshold:
                return record
                
        return None

    def add_memory(self, macro_features, weights, final_rul, base_preds_dict=None):
        """将完整的状态切片（物理特征 + 决策权重 + 底层预测值）存入记忆体"""
        if len(self.cache) >= self.max_size:
            self.cache.pop(0) 
            
        self.cache.append({
            'macro_features': macro_features.copy(),
            'weights': weights,
            'rul': final_rul,
            'base_preds': base_preds_dict # 将当时 4 个专家的预测值也存下来用于严谨校验
        })