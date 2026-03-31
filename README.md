# rul_agent
航空发动机rul预测

RMSE：航空发动机剩余寿命的预测值和真实值的均方根误差（越小越好）

Score：例如航空发动机剩余寿命30，预测到25（低预测）只会损失更换设备成本，预测到40（高预测）可能会导致飞机损坏。故对于高预测的惩罚更大。（score越小越好）

.env:换成自己的密钥

data_engine:数据预处理

项目流程：

python train_pipeline.py   预训练模型，采用早停机制，保存最好的.pt

python evaluate_testset.py  测试单体模型的RMSE以及Score

python evaluate_agent.py。  进行agent融合
