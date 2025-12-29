"""
FastMCP服务器 - LogRobust日志异常检测工具
使用SSE传输在127.0.0.1:4002上提供服务
"""
import asyncio
import os
import json
import logging
import copy
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import time

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# 导入项目模块
from data_loader import AliyunDataLoaderForFed, CMCCDataLoaderForFed, AliyunDataLoader
from model import robustlog

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量用于存储训练状态和结果
training_status: Dict[str, Any] = {
    "federated": {"status": "idle", "progress": 0, "current_epoch": 0, "loss": [], "accuracy": [], "f1_score": []},
    "centralized": {"status": "idle", "progress": 0, "current_epoch": 0, "loss": [], "accuracy": [], "f1_score": []}
}

training_results: Dict[str, Any] = {
    "federated": None,
    "centralized": None
}

# FastMCP服务器实现
try:
    from fastmcp import FastMCP
    mcp = FastMCP("LogRobust MCP Server")
except ImportError:
    # 如果fastmcp不存在，使用基于标准库的简单实现
    logger.warning("fastmcp库未找到，使用简化实现")
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from sse_starlette.sse import EventSourceResponse
    import uvicorn
    
    app = FastAPI()
    mcp = None  # 我们将使用FastAPI代替


def torch_seed(seed: int = 2020):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def FedAvg(w: List[Dict]) -> Dict:
    """联邦平均算法"""
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def cal_f1_metrics(label_list: List[int], pred_list: List[int]) -> Dict[str, Any]:
    """计算F1分数和分类报告"""
    label_arr = np.array(label_list)
    pred_arr = np.array(pred_list)
    ad_label = np.where(label_arr > 0, 1, 0)
    ad_pred = np.where(pred_arr > 0, 1, 0)
    
    report = classification_report(ad_label, ad_pred, output_dict=True)
    precision, recall, f1, support = precision_recall_fscore_support(
        ad_label, ad_pred, average='weighted', zero_division=0
    )
    
    return {
        "classification_report": report,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "support": int(np.sum(support))
    }


async def train_federated_model_impl(
    datatype: str,
    learning_rate: float,
    batch_size: int,
    global_epochs: int,
    local_epochs: int,
    world_size: int,
    gpu_id: int,
    seed: int,
    out_dir: str
) -> Dict[str, Any]:
    """联邦学习训练实现"""
    torch_seed(seed)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id >= 0 else 'cpu')
    
    logger.info(f"开始联邦学习训练: datatype={datatype}, epochs={global_epochs}, clients={world_size-1}")
    
    # 加载数据集
    if datatype == 'aliyun':
        def load_datasets_aliyun(rank, world_size):
            train_db = AliyunDataLoaderForFed(mode='train', semi=False, rank=rank, world_size=world_size)
            test_db = AliyunDataLoaderForFed(mode='test', semi=False, rank=rank, world_size=world_size)
            train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, drop_last=True)
            test_loader = DataLoader(test_db, batch_size=batch_size)
            val_loader = test_loader
            return train_loader, val_loader, test_loader
        loader_list = [load_datasets_aliyun(i, world_size) for i in range(1, world_size)]
    else:  # cmcc
        def load_datasets_cmcc(rank, world_size):
            train_db = CMCCDataLoaderForFed(mode='train', semi=False, rank=rank, world_size=world_size)
            test_db = CMCCDataLoaderForFed(mode='test', semi=False, rank=rank, world_size=world_size)
            train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, drop_last=True)
            test_loader = DataLoader(test_db, batch_size=batch_size)
            val_loader = test_loader
            return train_loader, val_loader, test_loader
        loader_list = [load_datasets_cmcc(i, world_size) for i in range(1, world_size)]
    
    # 创建客户端模型
    client_list = [robustlog(300, 10, 2, device=device).to(device) for i in range(1, world_size)]
    best_score_list = [0.0 for _ in range(world_size - 1)]
    
    # 创建输出目录
    save_dir = os.path.join(out_dir, "model_save")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    criteon = nn.CrossEntropyLoss().to(device)
    
    # 保存初始模型参数
    for i, model in enumerate(client_list):
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_param{i}.pkl"))
    
    all_losses = []
    all_accuracies = []
    
    # 训练循环
    for it in range(global_epochs):
        logger.info(f"全局轮次 {it+1}/{global_epochs}")
        training_status["federated"]["current_epoch"] = it + 1
        training_status["federated"]["progress"] = int((it + 1) / global_epochs * 100)
        
        # 客户端本地训练
        for i in range(world_size - 1):
            client_model = client_list[i]
            optimizer_client = optim.SGD(client_model.parameters(), lr=learning_rate)
            client_model.train()
            
            train_loader = loader_list[i][0]
            epoch_losses = []
            for e in range(local_epochs):
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    target[target != 0] = 1
                    
                    logits = client_model(data)
                    loss = criteon(logits, target)
                    
                    optimizer_client.zero_grad()
                    loss.backward()
                    optimizer_client.step()
                    
                    epoch_losses.append(loss.item())
        
        # 联邦平均
        client_model = FedAvg([x.state_dict() for x in client_list])
        for i in range(world_size - 1):
            client_list[i].load_state_dict(client_model)
        
        # 验证
        epoch_accuracies = []
        for i in range(world_size - 1):
            client_model = client_list[i]
            client_model.eval()
            
            val_loader = loader_list[i][1]
            test_loss = 0
            y_true = []
            y_pred = []
            
            with torch.no_grad():
                for data, target in val_loader:
                    target[target != 0] = 1
                    y_true.extend(target.numpy())
                    data, target = data.to(device), target.to(device)
                    
                    logits = client_model(data)
                    test_loss += criteon(logits, target).item()
                    pred = logits.data.topk(1)[1].flatten().cpu().numpy()
                    y_pred.extend(pred)
            
            test_loss /= len(val_loader.dataset)
            acc = accuracy_score(y_true, y_pred)
            epoch_accuracies.append(float(acc))
            
            if acc > best_score_list[i]:
                best_score_list[i] = float(acc)
                torch.save(client_model.state_dict(), os.path.join(save_dir, f"model_param{i}.pkl"))
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_acc = np.mean(epoch_accuracies) if epoch_accuracies else 0.0
        
        all_losses.append(avg_loss)
        all_accuracies.append(avg_acc)
        training_status["federated"]["loss"].append(avg_loss)
        training_status["federated"]["accuracy"].append(avg_acc)
    
    # 最终评估
    all_reports = []
    for i in range(world_size - 1):
        client_model = client_list[i]
        client_model.load_state_dict(torch.load(os.path.join(save_dir, f"model_param{i}.pkl"), map_location=device))
        client_model.eval()
        
        test_loader = loader_list[i][2]
        pred_list = []
        label_list = []
        
        with torch.no_grad():
            for data, target in test_loader:
                target[target != 0] = 1
                label_list.extend(target.numpy())
                data, target = data.to(device), target.to(device)
                
                logits = client_model(data)
                pred = logits.data.topk(1)[1].flatten().cpu().numpy()
                pred_list.extend(pred)
        
        metrics = cal_f1_metrics(label_list, pred_list)
        all_reports.append(metrics)
    
    result = {
        "model_save_dir": save_dir,
        "loss_history": all_losses,
        "accuracy_history": all_accuracies,
        "final_accuracy": float(np.mean(all_accuracies)) if all_accuracies else 0.0,
        "client_reports": all_reports,
        "best_scores": [float(x) for x in best_score_list]
    }
    
    training_status["federated"]["status"] = "completed"
    training_results["federated"] = result
    
    return result


async def train_centralized_model_impl(
    datatype: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    gpu_id: int,
    seed: int,
    out_dir: str
) -> Dict[str, Any]:
    """单机训练实现"""
    torch_seed(seed)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id >= 0 else 'cpu')
    
    logger.info(f"开始单机训练: datatype={datatype}, epochs={epochs}")
    
    # 加载数据集
    if datatype == 'aliyun':
        train_db = AliyunDataLoader(mode='train')
        test_db = AliyunDataLoader(mode='test')
    else:
        raise ValueError(f"单机训练目前只支持aliyun数据集，收到: {datatype}")
    
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_db, batch_size=batch_size)
    
    # 创建模型
    model = robustlog(300, 10, 2, device=device).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criteon = nn.CrossEntropyLoss().to(device)
    
    # 创建输出目录
    save_dir = os.path.join(out_dir, "model_save")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    best_score = 0.0
    all_losses = []
    all_accuracies = []
    
    # 训练循环
    for e in range(epochs):
        logger.info(f"训练轮次 {e+1}/{epochs}")
        training_status["centralized"]["current_epoch"] = e + 1
        training_status["centralized"]["progress"] = int((e + 1) / epochs * 100)
        
        # 训练
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = criteon(logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        all_losses.append(avg_loss)
        training_status["centralized"]["loss"].append(avg_loss)
        
        # 验证
        model.eval()
        y_true = []
        y_pred = []
        test_loss = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                y_true.extend(target.numpy())
                data, target = data.to(device), target.to(device)
                logits = model(data)
                test_loss += criteon(logits, target).item()
                pred = logits.data.topk(1)[1].flatten().cpu().numpy()
                y_pred.extend(pred)
        
        acc = accuracy_score(y_true, y_pred)
        all_accuracies.append(float(acc))
        training_status["centralized"]["accuracy"].append(float(acc))
        
        if acc > best_score:
            best_score = float(acc)
            torch.save(model.state_dict(), os.path.join(save_dir, "model_param.pkl"))
    
    # 最终评估
    model.load_state_dict(torch.load(os.path.join(save_dir, "model_param.pkl"), map_location=device))
    model.eval()
    
    pred_list = []
    label_list = []
    
    with torch.no_grad():
        for data, target in test_loader:
            label_list.extend(target.numpy())
            data, target = data.to(device), target.to(device)
            logits = model(data)
            pred = logits.data.topk(1)[1].flatten().cpu().numpy()
            pred_list.extend(pred)
    
    metrics = cal_f1_metrics(label_list, pred_list)
    
    result = {
        "model_save_path": os.path.join(save_dir, "model_param.pkl"),
        "loss_history": all_losses,
        "accuracy_history": all_accuracies,
        "final_accuracy": best_score,
        "metrics": metrics
    }
    
    training_status["centralized"]["status"] = "completed"
    training_results["centralized"] = result
    
    return result


# MCP工具定义
if mcp is not None:
    @mcp.tool()
    async def train_federated_model(
        data_type: str = "aliyun",
        learning_rate: float = 0.001,
        batch_size: int = 64,
        global_epochs: int = 10,
        local_epochs: int = 10,
        num_clients: int = 6,
        gpu_id: int = 0,
        seed: int = 2020,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        训练联邦学习模型
        
        Args:
            data_type: 数据类型选择 (aliyun/cmcc)
            learning_rate: 学习率
            batch_size: 批次大小
            global_epochs: 全局训练轮数
            local_epochs: 每个客户端的本地训练轮数
            num_clients: 客户端数量（world_size）
            gpu_id: GPU设备ID
            seed: 随机种子
            output_dir: 输出目录（可选）
            
        Returns:
            包含训练结果的字典
        """
        if output_dir is None:
            output_dir = f"output/fedlog_{data_type}_logs_{global_epochs}e_{local_epochs}locEpoch_{learning_rate}lr_{batch_size}bs_{num_clients}ws"
        
        training_status["federated"]["status"] = "training"
        try:
            result = await train_federated_model_impl(
                datatype=data_type,
                learning_rate=learning_rate,
                batch_size=batch_size,
                global_epochs=global_epochs,
                local_epochs=local_epochs,
                world_size=num_clients,
                gpu_id=gpu_id,
                seed=seed,
                out_dir=output_dir
            )
            return result
        except Exception as e:
            training_status["federated"]["status"] = "error"
            logger.error(f"联邦学习训练错误: {str(e)}")
            raise
    
    @mcp.tool()
    async def train_centralized_model(
        data_type: str = "aliyun",
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 10,
        gpu_id: int = 0,
        seed: int = 2020,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        训练单机模型
        
        Args:
            data_type: 数据类型选择 (aliyun)
            learning_rate: 学习率
            batch_size: 批次大小
            epochs: 训练轮数
            gpu_id: GPU设备ID
            seed: 随机种子
            output_dir: 输出目录（可选）
            
        Returns:
            包含训练结果的字典
        """
        if output_dir is None:
            output_dir = f"output/centralized_{data_type}_{epochs}e_{learning_rate}lr_{batch_size}bs"
        
        training_status["centralized"]["status"] = "training"
        try:
            result = await train_centralized_model_impl(
                datatype=data_type,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                gpu_id=gpu_id,
                seed=seed,
                out_dir=output_dir
            )
            return result
        except Exception as e:
            training_status["centralized"]["status"] = "error"
            logger.error(f"单机训练错误: {str(e)}")
            raise
    
    @mcp.tool()
    async def predict_logs(
        model_path: str,
        log_data: List[List[List[float]]],
        gpu_id: int = 0
    ) -> Dict[str, Any]:
        """
        使用已训练模型进行日志异常预测
        
        Args:
            model_path: 模型文件路径
            log_data: 日志数据，形状为 [batch_size, sequence_length, feature_dim]
            gpu_id: GPU设备ID
            
        Returns:
            预测结果字典
        """
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id >= 0 else 'cpu')
        
        # 加载模型
        model = robustlog(300, 10, 2, device=device).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # 转换为tensor
        data_tensor = torch.tensor(log_data, dtype=torch.float32).to(device)
        
        # 预测
        with torch.no_grad():
            logits = model(data_tensor)
            predictions = logits.data.topk(1)[1].flatten().cpu().numpy().tolist()
            probabilities = torch.softmax(logits, dim=1).cpu().numpy().tolist()
        
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "anomaly_flags": [1 if p > 0 else 0 for p in predictions]
        }
    
    @mcp.tool()
    async def get_training_status(
        model_type: str = "both"
    ) -> Dict[str, Any]:
        """
        获取训练状态和进度
        
        Args:
            model_type: 模型类型 (federated/centralized/both)
            
        Returns:
            训练状态字典
        """
        if model_type == "federated":
            return {"federated": training_status["federated"]}
        elif model_type == "centralized":
            return {"centralized": training_status["centralized"]}
        else:
            return training_status
    
    @mcp.tool()
    async def get_training_results(
        model_type: str = "both"
    ) -> Dict[str, Any]:
        """
        获取训练结果
        
        Args:
            model_type: 模型类型 (federated/centralized/both)
            
        Returns:
            训练结果字典
        """
        if model_type == "federated":
            return {"federated": training_results["federated"]}
        elif model_type == "centralized":
            return {"centralized": training_results["centralized"]}
        else:
            return training_results
    
    @mcp.tool()
    async def evaluate_model(
        model_path: str,
        data_type: str = "aliyun",
        batch_size: int = 64,
        gpu_id: int = 0,
        client_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        加载已训练模型进行评估
        
        Args:
            model_path: 模型文件路径
            data_type: 数据类型 (aliyun/cmcc)
            batch_size: 批次大小
            gpu_id: GPU设备ID
            client_id: 客户端ID（仅用于联邦学习模型）
            
        Returns:
            评估结果字典
        """
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id >= 0 else 'cpu')
        
        # 加载模型
        model = robustlog(300, 10, 2, device=device).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # 加载测试数据
        if data_type == 'aliyun':
            test_db = AliyunDataLoader(mode='test')
        else:
            raise ValueError(f"评估目前只支持aliyun数据集，收到: {data_type}")
        
        test_loader = DataLoader(test_db, batch_size=batch_size)
        
        # 评估
        criteon = nn.CrossEntropyLoss().to(device)
        pred_list = []
        label_list = []
        test_loss = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                label_list.extend(target.numpy())
                data, target = data.to(device), target.to(device)
                logits = model(data)
                test_loss += criteon(logits, target).item()
                pred = logits.data.topk(1)[1].flatten().cpu().numpy()
                pred_list.extend(pred)
        
        test_loss /= len(test_loader.dataset)
        acc = accuracy_score(label_list, pred_list)
        metrics = cal_f1_metrics(label_list, pred_list)
        
        return {
            "test_loss": float(test_loss),
            "accuracy": float(acc),
            **metrics
        }

    # 运行服务器
    if __name__ == "__main__":
        try:
            # 尝试使用FastMCP的run方法
            mcp.run(transport="sse", host="127.0.0.1", port=4002)
        except Exception as e:
            logger.error(f"无法使用FastMCP运行服务器: {e}")
            logger.info("请确保已安装fastmcp库: pip install fastmcp")

else:
    # 使用标准库实现的备用方案（不依赖fastmcp）
    logger.warning("fastmcp库未安装，使用标准库实现")
    
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        from typing import Optional as Opt
        import uvicorn
        
        app = FastAPI(title="LogRobust MCP Server")
        app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
        
        class TrainFederatedRequest(BaseModel):
            data_type: str = "aliyun"
            learning_rate: float = 0.001
            batch_size: int = 64
            global_epochs: int = 10
            local_epochs: int = 10
            num_clients: int = 6
            gpu_id: int = 0
            seed: int = 2020
            output_dir: Opt[str] = None
        
        class TrainCentralizedRequest(BaseModel):
            data_type: str = "aliyun"
            learning_rate: float = 0.001
            batch_size: int = 64
            epochs: int = 10
            gpu_id: int = 0
            seed: int = 2020
            output_dir: Opt[str] = None
        
        class PredictRequest(BaseModel):
            model_path: str
            log_data: List[List[List[float]]]
            gpu_id: int = 0
        
        class EvaluateRequest(BaseModel):
            model_path: str
            data_type: str = "aliyun"
            batch_size: int = 64
            gpu_id: int = 0
            client_id: Opt[int] = None
        
        @app.post("/tools/train_federated_model")
        async def train_federated_model_api(request: TrainFederatedRequest):
            """联邦学习训练API"""
            training_status["federated"]["status"] = "training"
            try:
                output_dir = request.output_dir or f"output/fedlog_{request.data_type}_logs_{request.global_epochs}e_{request.local_epochs}locEpoch_{request.learning_rate}lr_{request.batch_size}bs_{request.num_clients}ws"
                result = await train_federated_model_impl(
                    datatype=request.data_type,
                    learning_rate=request.learning_rate,
                    batch_size=request.batch_size,
                    global_epochs=request.global_epochs,
                    local_epochs=request.local_epochs,
                    world_size=request.num_clients,
                    gpu_id=request.gpu_id,
                    seed=request.seed,
                    out_dir=output_dir
                )
                return result
            except Exception as e:
                training_status["federated"]["status"] = "error"
                logger.error(f"联邦学习训练错误: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/tools/train_centralized_model")
        async def train_centralized_model_api(request: TrainCentralizedRequest):
            """单机训练API"""
            training_status["centralized"]["status"] = "training"
            try:
                output_dir = request.output_dir or f"output/centralized_{request.data_type}_{request.epochs}e_{request.learning_rate}lr_{request.batch_size}bs"
                result = await train_centralized_model_impl(
                    datatype=request.data_type,
                    learning_rate=request.learning_rate,
                    batch_size=request.batch_size,
                    epochs=request.epochs,
                    gpu_id=request.gpu_id,
                    seed=request.seed,
                    out_dir=output_dir
                )
                return result
            except Exception as e:
                training_status["centralized"]["status"] = "error"
                logger.error(f"单机训练错误: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/tools/predict_logs")
        async def predict_logs_api(request: PredictRequest):
            """模型预测API"""
            try:
                device = torch.device(f'cuda:{request.gpu_id}' if torch.cuda.is_available() and request.gpu_id >= 0 else 'cpu')
                model = robustlog(300, 10, 2, device=device).to(device)
                model.load_state_dict(torch.load(request.model_path, map_location=device))
                model.eval()
                data_tensor = torch.tensor(request.log_data, dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = model(data_tensor)
                    predictions = logits.data.topk(1)[1].flatten().cpu().numpy().tolist()
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy().tolist()
                return {
                    "predictions": predictions,
                    "probabilities": probabilities,
                    "anomaly_flags": [1 if p > 0 else 0 for p in predictions]
                }
            except Exception as e:
                logger.error(f"预测错误: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/tools/get_training_status")
        async def get_training_status_api(model_type: str = "both"):
            """获取训练状态API"""
            if model_type == "federated":
                return {"federated": training_status["federated"]}
            elif model_type == "centralized":
                return {"centralized": training_status["centralized"]}
            else:
                return training_status
        
        @app.get("/tools/get_training_results")
        async def get_training_results_api(model_type: str = "both"):
            """获取训练结果API"""
            if model_type == "federated":
                return {"federated": training_results["federated"]}
            elif model_type == "centralized":
                return {"centralized": training_results["centralized"]}
            else:
                return training_results
        
        @app.post("/tools/evaluate_model")
        async def evaluate_model_api(request: EvaluateRequest):
            """模型评估API"""
            try:
                device = torch.device(f'cuda:{request.gpu_id}' if torch.cuda.is_available() and request.gpu_id >= 0 else 'cpu')
                model = robustlog(300, 10, 2, device=device).to(device)
                model.load_state_dict(torch.load(request.model_path, map_location=device))
                model.eval()
                
                if request.data_type == 'aliyun':
                    test_db = AliyunDataLoader(mode='test')
                else:
                    raise ValueError(f"评估目前只支持aliyun数据集，收到: {request.data_type}")
                
                test_loader = DataLoader(test_db, batch_size=request.batch_size)
                criteon = nn.CrossEntropyLoss().to(device)
                pred_list = []
                label_list = []
                test_loss = 0
                
                with torch.no_grad():
                    for data, target in test_loader:
                        label_list.extend(target.numpy())
                        data, target = data.to(device), target.to(device)
                        logits = model(data)
                        test_loss += criteon(logits, target).item()
                        pred = logits.data.topk(1)[1].flatten().cpu().numpy()
                        pred_list.extend(pred)
                
                test_loss /= len(test_loader.dataset)
                acc = accuracy_score(label_list, pred_list)
                metrics = cal_f1_metrics(label_list, pred_list)
                
                return {
                    "test_loss": float(test_loss),
                    "accuracy": float(acc),
                    **metrics
                }
            except Exception as e:
                logger.error(f"评估错误: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/")
        async def root():
            return {
                "name": "LogRobust MCP Server",
                "version": "1.0.0",
                "endpoints": [
                    "/tools/train_federated_model",
                    "/tools/train_centralized_model",
                    "/tools/predict_logs",
                    "/tools/get_training_status",
                    "/tools/get_training_results",
                    "/tools/evaluate_model"
                ]
            }
        
        if __name__ == "__main__":
            logger.info("启动MCP服务器在 127.0.0.1:4002")
            uvicorn.run(app, host="127.0.0.1", port=4002)
    
    except ImportError as e:
        logger.error(f"缺少必要的依赖库: {e}")
        logger.info("请安装: pip install fastapi uvicorn pydantic")

