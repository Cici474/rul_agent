from .base_agent import BaseExpertAgent

class AgentLightGBM(BaseExpertAgent):
    def __init__(self, client):
        super().__init__(
            name="LightGBM", 
            persona_desc="【基于梯度提升决策树的非线性统计交互模型】。由于缺乏时序记忆机制，其推理完全依赖截面特征的空间分布。优势在于对高频机械噪声与异常值（Outliers）极具鲁棒性，能够在复杂工况下提供具备极高安全下限的保守预测。适用于作为防止深度学习模型发生严重梯度崩塌或数值突变的基准锚点（Anchor）。", 
            client=client
        )