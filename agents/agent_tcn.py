from .base_agent import BaseExpertAgent

class AgentTCN(BaseExpertAgent):
    def __init__(self, client):
        super().__init__(
            name="TCN", 
            persona_desc="【基于膨胀因果卷积（Dilated Causal Convolutions）的时间序列网络】。其算法机理依赖于扩大的局部感受野，对序列中的高频瞬态突变极其敏感。优势在于处理晚期剧烈退化阶段（高系统方差、大斜率）的急剧恶化信号。但在早期平稳阶段容易因过度捕捉微小波动而产生过拟合现象。", 
            client=client
        )