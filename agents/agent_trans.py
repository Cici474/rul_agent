from .base_agent import BaseExpertAgent

class AgentTransformer(BaseExpertAgent):
    def __init__(self, client):
        super().__init__(
            name="Transformer", 
            persona_desc="【基于多尺度卷积与全局自注意力机制（Global Self-Attention）的深层网络】。其核心优势在于能够跨越极长的时间步直接计算特征向量之间的点积相似度。极其擅长在退化极早期（低方差、近乎零斜率）从看似健康、平稳的信号中，挖掘出隐蔽的长程时空衰退依赖关系。但在极晚期容易被高频噪声干扰注意力权重的分配。", 
            client=client
        )