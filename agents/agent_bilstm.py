from .base_agent import BaseExpertAgent

class AgentBiLSTM(BaseExpertAgent):
    def __init__(self, client):
        super().__init__(
            name="BiLSTM", 
            persona_desc="【具备双向记忆门控机制的循环神经网络】。其理论基础契合了航空发动机内部组件材料的马尔可夫累积疲劳损伤过程（Cumulative Fatigue Damage Process）。优势在于追踪长期的、具有时间连续性的稳定退化趋势。在捕捉中期渐进式磨损阶段（方差适中、斜率稳定增加）具有不可替代的平滑追踪能力。", 
            client=client
        )