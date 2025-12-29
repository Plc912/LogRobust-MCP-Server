# LogRobust MCP Server

MCP封装作者：庞力铖

github地址：https://github.com/Plc912/LogRobust-MCP-Server.git

邮箱：3522236586@qq.com

原代码获取：OpenAIops

## 简介

这是一个基于FastMCP的日志异常检测工具服务器，将LogRobust项目的联邦学习和单机训练功能封装为MCP（Model Context Protocol）工具。

# **注意：这个工具仅支持检测aliyun(阿里云)和cmcc(中国移动)的数据集，其他数据集均不支持检测。**

## 功能特性

1. **联邦学习模型训练** - 支持多客户端协作训练
2. **单机模型训练** - 传统集中式训练
3. **模型预测** - 使用已训练模型进行日志异常预测
4. **模型评估** - 评估已训练模型的性能
5. **训练状态查询** - 实时查看训练进度和状态
6. **训练结果查询** - 获取训练历史结果和指标

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：

- torch>=1.8.0
- numpy>=1.19.0
- scikit-learn>=0.24.0
- fastapi>=0.68.0
- uvicorn[standard]>=0.15.0
- pydantic>=1.8.0
- fastmcp>=0.1.0 (如果不可用，会自动使用FastAPI备用方案)

## 启动服务器

```bash
python mcp_server.py
```

服务器将在 `127.0.0.1:4002` 上启动，使用SSE（Server-Sent Events）传输协议。

## API接口说明

### 1. 训练联邦学习模型

**工具名**: `train_federated_model`

**参数**:

- `data_type` (str): 数据类型选择，可选 "aliyun" 或 "cmcc"，默认 "aliyun"
- `learning_rate` (float): 学习率，默认 0.001
- `batch_size` (int): 批次大小，默认 64
- `global_epochs` (int): 全局训练轮数，默认 10
- `local_epochs` (int): 每个客户端的本地训练轮数，默认 10
- `num_clients` (int): 客户端数量（world_size），默认 6
- `gpu_id` (int): GPU设备ID，默认 0
- `seed` (int): 随机种子，默认 2020
- `output_dir` (str, optional): 输出目录，如果不指定会自动生成

**返回**: 包含以下字段的字典

- `model_save_dir`: 模型保存目录
- `loss_history`: 损失值历史记录
- `accuracy_history`: 准确率历史记录
- `final_accuracy`: 最终准确率
- `client_reports`: 各客户端的评估报告（包含precision, recall, f1-score等）
- `best_scores`: 各客户端的最佳分数

### 2. 训练单机模型

**工具名**: `train_centralized_model`

**参数**:

- `data_type` (str): 数据类型选择，目前支持 "aliyun"，默认 "aliyun"
- `learning_rate` (float): 学习率，默认 0.001
- `batch_size` (int): 批次大小，默认 64
- `epochs` (int): 训练轮数，默认 10
- `gpu_id` (int): GPU设备ID，默认 0
- `seed` (int): 随机种子，默认 2020
- `output_dir` (str, optional): 输出目录，如果不指定会自动生成

**返回**: 包含以下字段的字典

- `model_save_path`: 模型保存路径
- `loss_history`: 损失值历史记录
- `accuracy_history`: 准确率历史记录
- `final_accuracy`: 最终准确率
- `metrics`: 评估指标（包含precision, recall, f1-score等）

### 3. 模型预测

**工具名**: `predict_logs`

**参数**:

- `model_path` (str): 已训练模型文件路径
- `log_data` (List[List[List[float]]]): 日志数据，形状为 [batch_size, sequence_length, feature_dim]
- `gpu_id` (int): GPU设备ID，默认 0

**返回**: 包含以下字段的字典

- `predictions`: 预测结果列表
- `probabilities`: 预测概率列表
- `anomaly_flags`: 异常标志列表（1表示异常，0表示正常）

### 4. 获取训练状态

**工具名**: `get_training_status`

**参数**:

- `model_type` (str): 模型类型，可选 "federated"、"centralized" 或 "both"，默认 "both"

**返回**: 训练状态字典，包含：

- `status`: 训练状态（idle/training/completed/error）
- `progress`: 训练进度（0-100）
- `current_epoch`: 当前轮次
- `loss`: 损失值列表
- `accuracy`: 准确率列表
- `f1_score`: F1分数列表

### 5. 获取训练结果

**工具名**: `get_training_results`

**参数**:

- `model_type` (str): 模型类型，可选 "federated"、"centralized" 或 "both"，默认 "both"

**返回**: 训练结果字典，包含完整的训练历史数据

### 6. 评估模型

**工具名**: `evaluate_model`

**参数**:

- `model_path` (str): 已训练模型文件路径
- `data_type` (str): 数据类型，默认 "aliyun"
- `batch_size` (int): 批次大小，默认 64
- `gpu_id` (int): GPU设备ID，默认 0
- `client_id` (int, optional): 客户端ID（仅用于联邦学习模型）

**返回**: 评估结果字典，包含：

- `test_loss`: 测试损失
- `accuracy`: 准确率
- `precision`: 精确率
- `recall`: 召回率
- `f1_score`: F1分数
- `classification_report`: 详细的分类报告

## 使用示例

### 使用FastMCP客户端

```python
from mcp_client import MCPClient

# 连接到服务器
client = MCPClient("http://127.0.0.1:4002")

# 训练联邦学习模型
result = await client.call_tool(
    "train_federated_model",
    data_type="aliyun",
    learning_rate=0.001,
    batch_size=64,
    global_epochs=10,
    local_epochs=10,
    num_clients=6,
    gpu_id=0
)
print(f"训练完成，模型保存在: {result['model_save_dir']}")

# 查询训练状态
status = await client.call_tool("get_training_status", model_type="federated")
print(f"训练状态: {status['federated']['status']}")
print(f"训练进度: {status['federated']['progress']}%")

# 使用模型进行预测
predictions = await client.call_tool(
    "predict_logs",
    model_path="output/fedlog_aliyun_logs_10e_10locEpoch_0.001lr_64bs_6ws/model_save/model_param0.pkl",
    log_data=[[[0.1, 0.2, ...], [...], ...]],  # 你的日志数据
    gpu_id=0
)
print(f"预测结果: {predictions['predictions']}")
```

### 使用HTTP API（备用方案）

如果fastmcp不可用，服务器会自动使用FastAPI提供HTTP接口：

```bash
# 训练联邦学习模型
curl -X POST "http://127.0.0.1:4002/tools/train_federated_model" \
  -H "Content-Type: application/json" \
  -d '{
    "data_type": "aliyun",
    "learning_rate": 0.001,
    "batch_size": 64,
    "global_epochs": 10,
    "local_epochs": 10,
    "num_clients": 6,
    "gpu_id": 0
  }'

# 查询训练状态
curl "http://127.0.0.1:4002/tools/get_training_status?model_type=federated"

# 使用模型进行预测
curl -X POST "http://127.0.0.1:4002/tools/predict_logs" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "output/.../model_param0.pkl",
    "log_data": [[[0.1, 0.2, ...], [...], ...]],
    "gpu_id": 0
  }'
```

## 注意事项

1. **数据文件路径**: 确保数据文件（.npy和.pickle文件）在正确的路径下。默认情况下：

   - Aliyun数据: `../Fedlog/data/data_{rank-1}.npy` 和 `../Fedlog/data/label_{rank-1}.npy`
   - CMCC数据: `/home/zhangshenglin/chezeyu/log/cmcc_0929/data/eventIndex_{rank-1}.npy`
   - 词向量文件: `EventId2WordVecter.pickle` 或 `EventId2WordVecter_cmcc.pickle`
2. **GPU支持**: 如果没有可用的GPU，会自动使用CPU，但训练速度会较慢。
3. **内存使用**: 联邦学习会创建多个客户端模型，注意内存使用情况。
4. **训练时间**: 训练时间取决于数据大小、模型复杂度和硬件配置。可以通过设置较少的epochs来快速测试。

## 故障排除

1. **ImportError: No module named 'fastmcp'**:

   - 服务器会自动使用FastAPI备用方案，功能不受影响
2. **CUDA out of memory**:

   - 减小batch_size
   - 减少num_clients
   - 使用CPU训练（设置gpu_id=-1）
3. **文件未找到错误**:

   - 检查数据文件路径是否正确
   - 确保pickle文件存在
4. **端口被占用**:

   - 修改mcp_server.py中的端口号（默认4002）

## 技术架构

- **模型**: 基于双向LSTM + Attention机制的robustlog模型
- **训练方式**: 支持联邦学习（FedAvg）和集中式训练
- **传输协议**: SSE (Server-Sent Events)
- **服务器框架**: FastMCP / FastAPI（备用）
- **异步处理**: 使用asyncio实现异步训练和推理

## 许可证

请参考项目根目录的LICENSE.txt文件。
