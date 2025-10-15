import os
import numpy as np
import torch
import torch_geometric
from CGAMP.Net.model_mol import Causal
import warnings
import json
from torch_geometric.data import DataLoader, Data
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, confusion_matrix,
                             roc_curve)
from collections import Counter
import argparse

# 忽略警告信息
warnings.filterwarnings('ignore')

# 设置计算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def num_graphs(data):
    """计算批次中的图数量"""
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def dict_to_pyg(data_dict):
    """将字典格式的数据转换为PyG图结构列表"""
    pyg_data_list = []
    sequence_names = [k for k in data_dict.keys() if k != 'labels']

    for i, seq_name in enumerate(sequence_names):
        seq_data = data_dict[seq_name]
        x = torch.tensor(seq_data['seq_feat'], dtype=torch.float)
        edge_index = torch.tensor(seq_data['edge_index'], dtype=torch.long)

        graph_data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([data_dict['labels'][i]], dtype=torch.long),
            sequence_name=seq_name  # 保存原始序列名称
        )

        if 'edge_weight' in seq_data:
            graph_data.edge_weight = torch.tensor(seq_data['edge_weight'], dtype=torch.float)

        pyg_data_list.append(graph_data)

    return pyg_data_list


def load_and_preprocess_test_data(test_data_path, target_class):
    """加载并预处理测试数据：先转换为PyG格式，再转换为二分类标签"""
    print(f"Loading test data from: {test_data_path}")

    # 1. 加载原始数据（字典格式）
    raw_data = torch.load(test_data_path)

    # 2. 转换为PyG图结构列表
    pyg_data_list = dict_to_pyg(raw_data)
    print(f"Converted {len(pyg_data_list)} graphs to PyG format")

    # 3. 转换为二分类标签（目标类别为1，其他为0）
    processed_data = []
    for graph in pyg_data_list:
        original_label = graph.y.item()
        binary_label = 1 if original_label == target_class else 0

        processed_graph = Data(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_weight=graph.edge_weight if hasattr(graph, 'edge_weight') else None,
            y=torch.tensor([binary_label], dtype=torch.long),
            sequence_name=graph.sequence_name
        )
        processed_data.append(processed_graph)

    # 打印类别分布
    labels = [d.y.item() for d in processed_data]
    dist = Counter(labels)
    print(f"Class distribution - Positive: {dist.get(1, 0)}, Negative: {dist.get(0, 0)}")
    return processed_data


def evaluate_model_performance(model, test_loader, device, target_class):
    """在测试集上评估模型性能，同时保存预测概率和真实标签用于ROC曲线"""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            if data.x.shape[0] == 1:
                continue

            pred = model.eval_forward(data)
            pred = torch.nn.functional.softmax(pred, dim=1)

            all_probs.append(pred[:, 1].cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    y_score = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # 处理单类别情况
    if len(np.unique(y_true)) < 2:
        print("Warning: Only one class present in test data")
        y_pred = np.zeros_like(y_true)
        best_th = 0.5
    else:
        # Youden指数选最佳阈值
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_th = thresholds[best_idx]
        print(f"Optimal threshold (Youden's index): {best_th:.4f}")
        y_pred = (y_score >= best_th).astype(int)

    # 计算指标
    metrics = {}
    try:
        metrics['auc'] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics['auc'] = 0.0
        print("Warning: Could not compute AUC (insufficient class samples)")

    try:
        metrics['aupr'] = average_precision_score(y_true, y_score)
    except ValueError:
        metrics['aupr'] = 0.0
        print("Warning: Could not compute AUPR (insufficient class samples)")

    metrics['f1'] = f1_score(y_true, y_pred, average='binary')
    metrics['precision'] = precision_score(y_true, y_pred, average='binary')
    metrics['recall'] = recall_score(y_true, y_pred, average='binary')

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        metrics['tp'] = int(tp)
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
    else:
        metrics['specificity'] = 0.0
        metrics['tp'] = 0
        metrics['tn'] = 0
        metrics['fp'] = 0
        metrics['fn'] = 0

    metrics['accuracy'] = np.mean(y_pred == y_true)
    metrics['best_threshold'] = float(best_th)

    return metrics


def load_trained_model(model_path, params, device):
    """加载已训练好的模型"""
    model = Causal(
        hidden_in=params['hidden_in'],
        hidden_out=2,  # 二分类任务（多分类拆分为11个二分类）
        hidden=params['hidden'],
        num_layer=params['layers'],
        cls_layer=params['cls_layer']
    ).to(device).float()

    # 加载模型权重（兼容多分类预测代码的保存格式）
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    best_acc = checkpoint.get('best_acc', 0.0)
    print(f"Loaded model with best training accuracy: {best_acc:.4f}")

    return model


def main(args):
    # 基础模型参数（与多分类预测代码保持一致）
    base_params = {
        'hidden_in': 33,
        'layers': 2,
        'cls_layer': 2
    }

    # 存储所有类别的评估结果
    all_evaluation_results = {}

    # 遍历11个类别进行评估（与多分类预测代码的类别数一致）
    for target_class in range(11):
        print(f"\n{'=' * 100}")
        print(f"Evaluating model for class {target_class} on independent test set")
        print(f"{'=' * 100}\n")

        # 构建路径（使用命令行参数或默认路径，对齐项目约定）
        model_path = os.path.join(args.model_dir, f"model_class_{target_class}.pt")
        params_file = os.path.join(args.params_dir, f"class_{target_class}.json")

        # 检查文件是否存在（提示用户从README下载对应文件）
        if not os.path.exists(model_path):
            print(f"Warning: Model file for class {target_class} not found at {model_path}\n"
                  f"Download from: [Antimicrobial Peptide Multi-class Classifier](https://drive.google.com/file/d/1ZzXc5aqXXvtilHDZSLr32dY8YiXSpbZe/view?usp=drive_link)")
            continue
        if not os.path.exists(params_file):
            print(f"Warning: Parameters file for class {target_class} not found at {params_file}\n"
                  f"Ensure params are saved in {args.params_dir} during training")
            continue

        # 加载最佳参数（与多分类预测代码的参数格式一致）
        print(f"Loading parameters from: {params_file}")
        with open(params_file, 'r') as f:
            best_params = json.load(f)
        full_params = {**base_params, **best_params}  # 合并基础参数和类别专属参数

        # 加载模型
        print(f"Loading model from: {model_path}")
        model = load_trained_model(model_path, full_params, device)

        # 加载预处理后的测试数据（与data_processing.py的输出路径一致）
        test_data = load_and_preprocess_test_data(args.test_data_path, target_class)

        # 创建测试集加载器（批次大小与多分类预测代码一致）
        test_loader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=torch_geometric.data.Batch.from_data_list
        )

        # 评估模型
        print("Starting evaluation...")
        metrics = evaluate_model_performance(model, test_loader, device, target_class=target_class)

        # 存储结果
        all_evaluation_results[target_class] = metrics

        # 打印当前类别的评估结果
        print("\nEvaluation metrics for this class:")
        print(f"  Best Threshold: {metrics['best_threshold']:.4f}")
        print(f"  Accuracy:       {metrics['accuracy']:.4f}")
        print(f"  AUC:            {metrics['auc']:.4f}")
        print(f"  AUPR:           {metrics['aupr']:.4f}")
        print(f"  F1 Score:       {metrics['f1']:.4f}")
        print(f"  Precision:      {metrics['precision']:.4f}")
        print(f"  Recall:         {metrics['recall']:.4f}")
        print(f"  Specificity:    {metrics['specificity']:.4f}")
        print(f"  Confusion Matrix: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")

    # 保存所有评估结果（路径放在模型目录，方便用户查找）
    output_file = os.path.join(args.model_dir, "multi_class_independent_test_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_evaluation_results, f, indent=4)

    print(f"\nAll evaluations completed. Results saved to: {output_file}")


if __name__ == "__main__":
    # 新增：命令行参数配置（默认路径对齐README约定，支持自定义）
    parser = argparse.ArgumentParser(description="CGAMP Multi-Class Classification Test (Align with README)")
    parser.add_argument('--test_data_path', type=str,
                        default='./data_processed/stage2-independent-dataset.pt',
                        help='Path to multi-class independent test .pt file (converted from FASTA, default: ./data_processed/stage2-independent-dataset.pt)')
    parser.add_argument('--model_dir', type=str,
                        default='./trained_models',
                        help='Directory of pre-trained multi-class models (download from README, default: ./trained_models)')
    parser.add_argument('--params_dir', type=str,
                        default='./best_params',
                        help='Directory of class-specific hyperparameters (JSON files, default: ./best_params)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation (same as training, default: 64)')

    args = parser.parse_args()
    main(args)