import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch.optim.lr_scheduler import CosineAnnealingLR
from ogb.graphproppred import Evaluator
from torch.optim import Adam
from sklearn.model_selection import train_test_split  # 新增：用于8:2划分
from CGAMP.utils.util import print_args, set_seed
from CGAMP.Net.model_mol import Causal
import time
import warnings
from torch_geometric.data import DataLoader as PyGDataLoader
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix
from collections import Counter
import argparse
warnings.filterwarnings('ignore')


def check_sample_distribution(loader):
    """检查数据集类别分布"""
    all_labels = []
    for data in loader:
        all_labels.extend(data.y.cpu().numpy())
    return Counter(all_labels)


def eval(model, loader, device, args, evaluator):
    """评估函数，返回测试集各项指标"""
    model.eval()
    all_probs = []  # 正类概率
    all_preds = []  # 预测标签
    all_labels = []
    total_loss = 0
    total_local_loss = 0
    total_global_loss = 0

    for data in loader:
        data = data.to(device)
        if data.x.shape[0] == 1:
            continue
        with torch.no_grad():
            pred = model.eval_forward(data)  # 输出logits
            # 计算概率
            if args.domain in ["size", "color"]:
                pred = torch.nn.functional.log_softmax(pred, dim=-1)
            else:
                pred = torch.nn.functional.softmax(pred, dim=1)
            prob = pred[:, 1]  # 正类概率
            all_probs.append(prob.cpu().numpy())
            all_preds.append(pred.argmax(dim=1).cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

            # 计算损失
            one_hot_target = data.y.view(-1).type(torch.int64)
            pred_loss = F.cross_entropy(pred, one_hot_target)
            total_loss += pred_loss.item() * num_graphs(data)

            # 对比损失（如果启用）
            if args.local or args.global_:
                causal = model.forward_causal(data)
                class_causal, lack_class = class_split(data.y, causal, args)
                prototype = prototype_update(
                    prototype=None,
                    num_classes=args.num_classes,
                    class_causal=class_causal,
                    lack_class=lack_class
                )
                memory_bank = torch.randn(args.num_classes, 10, args.hidden).to(device)
                if args.local:
                    local_loss, _ = local_ssl(prototype, memory_bank, args, class_causal, lack_class)
                    total_local_loss += local_loss.item() * num_graphs(data)
                if args.global_:
                    global_loss = global_ssl(prototype, class_causal, lack_class, args.num_classes)
                    total_global_loss += global_loss.item() * num_graphs(data)

    # 计算平均损失
    num = len(loader.dataset)
    avg_pred_loss = total_loss / num
    avg_local_loss = total_local_loss / num if args.local else 0
    avg_global_loss = total_global_loss / num if args.global_ else 0
    avg_loss = args.pred * avg_pred_loss + args.l * avg_local_loss + args.g * avg_global_loss

    # 拼接所有结果
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 计算各项指标
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    try:
        aupr = average_precision_score(all_labels, all_probs)
    except:
        aupr = 0.0

    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')

    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    else:
        specificity = 0.0  # 防止单类别时出错

    acc = np.mean(all_preds == all_labels)

    return auc, aupr, f1, precision, recall, specificity, avg_loss, acc


# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss().to(device)


def main(args):
    set_seed(args.seed)
    # 从 args 读取数据路径（支持默认/自定义）
    data_path = os.path.join(args.data_dir, 'stage-1 benchmark dataset.pt')
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found at {data_path}\n"
            "Run 'python data_processing.py' first, or specify --data_dir to your .pt file folder."
        )
    data_list = torch.load(data_path)
    new_data_list = []
    for graph_dict in data_list:
        graph_data = Data(
            x=graph_dict["x"],
            edge_index=graph_dict["edge_index"],
            edge_attr=graph_dict["edge_attr"],
            y=graph_dict["y"]
        )
        new_data_list.append(graph_data)
    data_list = new_data_list

    # 数据类型转换（保持不变）
    for data in data_list:
        data.x = data.x.float()  # 确保float32类型

    # 8:2划分训练集和测试集（分层抽样，保持不变）
    all_labels = torch.cat([data.y for data in data_list], dim=0).cpu().numpy()
    train_indices, test_indices = train_test_split(
        range(len(data_list)),
        test_size=0.2,  # 测试集占20%
        random_state=args.seed,
        stratify=all_labels  # 分层抽样，保证类别比例
    )
    train_dataset = [data_list[i] for i in train_indices]
    test_dataset = [data_list[i] for i in test_indices]

    # 数据加载器（保持不变）
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=torch_geometric.data.Batch.from_data_list
    )
    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=torch_geometric.data.Batch.from_data_list
    )

    # 打印数据分布（保持不变）
    print(f"训练集分布: {check_sample_distribution(train_loader)} (样本数: {len(train_dataset)})")
    print(f"测试集分布: {check_sample_distribution(test_loader)} (样本数: {len(test_dataset)})")

    # 初始化模型（保持不变）
    model = Causal(
        hidden_in=args.hidden_in,
        hidden_out=args.num_classes,
        hidden=args.hidden,
        num_layer=args.layers,
        cls_layer=args.cls_layer
    ).to(device).float()

    evaluator = Evaluator(args.eval_name)
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
        last_epoch=-1,
        verbose=False
    )

    # 记录最佳结果（保持不变）
    best_test_result = None
    best_test_acc = 0.0
    train_losses = []
    test_losses = []
    train_aucs = []
    test_aucs = []

    start_time = time.time()
    print("\n开始训练...")
    for epoch in range(1, args.epochs + 1):
        start_epoch = time.time()
        model.train()
        total_loss = 0.0
        total_loss_p = 0.0
        total_loss_global = 0.0
        total_loss_local = 0.0
        memory_bank = torch.randn(args.num_classes, 10, args.hidden).cuda()

        # 训练循环（保持不变）
        for step, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            data.x = data.x.float()
            causal = model.forward_causal(data)
            pred = model(causal)

            # 计算预测损失（保持不变）
            if args.domain in ["size", "color"]:
                pred = torch.nn.functional.log_softmax(pred, dim=-1)
            else:
                pred = torch.nn.functional.softmax(pred, dim=1)
            one_hot_target = data.y.view(-1).type(torch.int64)
            pred_loss = F.cross_entropy(pred, one_hot_target)

            # 对比损失（保持不变）
            global_loss = 0.0
            local_loss = 0.0
            if args.global_ or args.local:
                class_causal, lack_class = class_split(data.y, causal, args)
                prototype = prototype_update(None, args.num_classes, class_causal, lack_class)
                if args.global_:
                    global_loss = global_ssl(prototype, class_causal, lack_class, args.num_classes)
                if args.local:
                    local_loss, memory_bank = local_ssl(prototype, memory_bank, args, class_causal, lack_class)

            # 总损失（保持不变）
            loss = args.pred * pred_loss + args.g * global_loss + args.l * local_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            # 累积损失（保持不变）
            batch_size = num_graphs(data)
            total_loss += loss.item() * batch_size
            total_loss_p += pred_loss.item() * batch_size
            total_loss_global += global_loss.item() * batch_size
            total_loss_local += local_loss.item() * batch_size

        # 计算平均训练损失（保持不变）
        num_train = len(train_loader.dataset)
        avg_train_loss = total_loss / num_train
        avg_train_p = total_loss_p / num_train
        avg_train_g = total_loss_global / num_train
        avg_train_l = total_loss_local / num_train

        # 评估训练集和测试集（保持不变）
        train_result = eval(model, train_loader, device, args, evaluator)
        test_result = eval(model, test_loader, device, args, evaluator)
        train_auc, train_aupr, train_f1, train_precision, train_recall, train_specificity, train_loss, train_acc = train_result
        test_auc, test_aupr, test_f1, test_precision, test_recall, test_specificity, test_loss, test_acc = test_result

        # 记录指标（保持不变）
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        train_aucs.append(train_auc)
        test_aucs.append(test_auc)

        # 更新最佳结果（保持不变）
        if test_acc > best_test_acc and epoch > args.pretrain:
            best_test_acc = test_acc
            best_test_result = test_result

        # 打印epoch信息（保持不变）
        print(f"Epoch [{epoch}/{args.epochs}] | "
              f"Loss: {avg_train_loss:.4f} = pred({avg_train_p:.4f}) + global({avg_train_g:.4f}) + local({avg_train_l:.4f}) | "
              f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
              f"Best Test Acc: {best_test_acc:.4f} | "
              f"时间: {(time.time() - start_epoch) / 60:.2f}min")

        lr_scheduler.step()

    # 打印总训练时间（保持不变）
    total_time = time.time() - start_time
    print(f"\n总训练时间: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

    # 打印最佳测试集指标（详细，保持不变）
    if best_test_result is not None:
        auc, aupr, f1, precision, recall, specificity, loss, acc = best_test_result
        print("\n==================== 最佳测试集指标 ====================")
        print(f"AUC:         {auc:.4f}")
        print(f"AUPR:        {aupr:.4f}")
        print(f"F1分数:      {f1:.4f}")
        print(f"精确率(P):   {precision:.4f}")
        print(f"召回率(R):   {recall:.4f}")
        print(f"特异度:      {specificity:.4f}")
        print(f"损失:        {loss:.4f}")
        print(f"准确率(Acc): {acc:.4f}")
        print("=======================================================")

    return best_test_result


def config_and_run(args):
    """配置并运行一次实验（8:2划分），保持不变"""
    print_args(args)
    best_result = main(args)
    return best_result


# 辅助函数（保持不变）
def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def class_split(y, causal_feature, args):
    class_causal = {}
    lack_class = []
    for i in range(args.num_classes):
        k = np.where(y.view(-1).cpu() == i)
        if len(k[0]) == 0:
            class_causal[i] = torch.randn(1, int(args.hidden)).cuda()
            lack_class.append(i)
        else:
            class_idx = torch.tensor(k).view(-1)
            class_causal_feature = causal_feature[class_idx]
            class_causal[i] = class_causal_feature
    return class_causal, lack_class


def softmax_with_temperature(input, t=1, axis=-1):
    ex = torch.exp(input / t)
    sum = torch.sum(ex, axis=axis)
    return ex / sum


def prototype_update(prototype, num_classes, class_causal, lack_class):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    if prototype is None:
        prototype_ = [
            torch.mean(class_causal[key].to(torch.float32), dim=0, keepdim=True)
            for key in class_causal
        ]
        prototype = torch.cat(prototype_, dim=0).detach()
    else:
        for i in range(num_classes):
            if i in lack_class:
                continue
            cosine = cos(prototype[i].detach(), class_causal[i]).detach()
            weights_proto = softmax_with_temperature(cosine, t=5).reshape(1, -1).detach()
            prototype[i] = torch.mm(weights_proto, class_causal[i]).detach()
    return prototype


def global_ssl(prototype, class_causal, lack_class, num_classes):
    distance = None
    for i in range(num_classes):
        if i in lack_class:
            continue
        prototype_ = torch.cat(
            (prototype[i:i + 1].detach(), prototype[0:i].detach(), prototype[i + 1:].detach()), 0
        )
        distance_ = torch.einsum('nc,kc->nk', [
            nn.functional.normalize(class_causal[i], dim=1),
            nn.functional.normalize(prototype_, dim=1)
        ])
        if distance is None:
            distance = F.softmax(distance_, dim=1)
        else:
            distance = torch.cat((distance, F.softmax(distance_, dim=1)), 0)
    labels = torch.zeros(distance.shape[0], dtype=torch.long).cuda()
    loss = criterion(distance, labels)
    return loss


def local_ssl(prototype, memory_bank, args, class_causal, lack_class):
    nce = None
    for key in class_causal:
        class_causal[key] = args.constraint * class_causal[key].float() + (1 - args.constraint) * prototype[
            key].float().detach()
    for i in lack_class:
        _ = class_causal.pop(i)
    for i in range(args.num_classes):
        if i in lack_class:
            continue
        pos = class_causal[i][0:1]
        class_causal_ = class_causal.copy()
        _ = class_causal_.pop(i)
        if class_causal_ == {}:
            hard_neg = memory_bank[i].detach()
        else:
            neg = torch.cat(list(class_causal_.values()))
            distance = F.softmax(torch.einsum('kc,nc->kn', [prototype[i:i + 1].detach(), neg]), dim=1)
            dis, idx = torch.sort(distance)
            if len(idx[0]) < 10:
                hard_neg = torch.cat((memory_bank[i][0:(10 - len(distance[0])), :].detach(), neg), 0)
            else:
                hard_neg = neg[idx[0][0:10]]
            memory_bank[i] = hard_neg
        sample = torch.cat((pos, hard_neg), 0)
        nce_ = F.softmax(torch.einsum('nc,kc->nk', [class_causal[i], sample]), dim=1)
        if nce is None:
            nce = nce_
        else:
            nce = torch.cat((nce, nce_), 0)
    labels = torch.zeros(nce.shape[0], dtype=torch.long).cuda()
    loss = criterion(nce, labels)
    return loss, memory_bank


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CGAMP Binary Classification Prediction")

    # 1. 路径相关（仅保留数据路径，删除模型保存路径）
    parser.add_argument('--data_dir', type=str, default='./data_processed',
                        help='Folder of .pt graph data (default: ./data_processed)')
    parser.add_argument('--root', type=str, default='./data',
                        help='Root data folder (default: ./data)')

    # 2. 任务/数据相关
    parser.add_argument('--dataset', type=str, default='benchmark dataset', help='Dataset name')
    parser.add_argument('--domain', type=str, default='basis', help='Domain setting')
    parser.add_argument('--shift', type=str, default='concept', help='Shift type')
    parser.add_argument('--eval_metric', type=str, default='rocauc', help='Evaluation metric')
    parser.add_argument('--eval_name', type=str, default='ogbg-molhiv', help='OGB evaluator name')

    # 3. 训练相关参数
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=521, help='Training epochs (default: 521)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate (default: 1e-5)')
    parser.add_argument('--weight_decay', type=float, default=0.0002664653812470226, help='Weight decay')
    parser.add_argument('--pretrain', type=int, default=52, help='Start epoch for best result tracking')
    parser.add_argument('--trails', type=int, default=1, help='Number of trails')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--delta', type=float, default=0.001, help='Early stopping delta')

    # 4. 模型结构参数
    parser.add_argument('--layers', type=int, default=2, help='GNN layers (default: 2)')
    parser.add_argument('--hidden', type=int, default=396, help='Model hidden dimension (default: 396)')
    parser.add_argument('--hidden_in', type=int, default=33, help='Input feature dimension (default: 33)')
    parser.add_argument('--cls_layer', type=int, default=2, help='Classifier layers (default: 2)')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes (default: 2)')

    # 5. SSL 权重参数
    parser.add_argument('--constraint', type=float, default=0.6741441799446165, help='Constraint coefficient')
    parser.add_argument('--g', type=float, default=0.2342299235496516, help='Global SSL weight')
    parser.add_argument('--l', type=float, default=0.111417383510663, help='Local SSL weight')
    parser.add_argument('--pred', type=float, default=0.7617102814256653, help='Prediction loss weight')

    # 6. SSL 开关
    parser.add_argument('--global_', action='store_true', default=True, help='Enable global SSL (default: True)')
    parser.add_argument('--local', action='store_true', default=True, help='Enable local SSL (default: True)')

    args = parser.parse_args()

    # 移除模型保存目录创建代码（无需保存模型，删除该逻辑）

    config_and_run(args)
    print(f"\nExperiment finished! Data used from: {args.data_dir}")