import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from CGAMP.Net.model_mol import Causal
import warnings
from torch_geometric.data import DataLoader, Data
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix

warnings.filterwarnings('ignore')

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss().to(device)

def load_model(model, filename='./trained_models/binary_model.pth'):
    """加载训练好的模型"""
    try:
        state_dict = torch.load(filename, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Model loaded from {filename}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def class_split(y, causal_feature, args):
    class_causal = {}
    lack_class = []
    idx = 0
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
    if prototype == None:
        prototype_ = [
            torch.mean(class_causal[key].to(torch.float32),
                       dim=0,
                       keepdim=True) for key in class_causal
        ]
        prototype = torch.cat(prototype_, dim=0).detach()
    else:
        for i in range(num_classes):
            if i in lack_class:
                continue
            else:
                cosine = cos(prototype[i].detach(), class_causal[i]).detach()
                weights_proto = softmax_with_temperature(cosine, t=5).reshape(
                    1, -1).detach()
                prototype[i] = torch.mm(weights_proto,
                                        class_causal[i]).detach()
    return prototype

def global_ssl(prototype, class_causal, lack_class, num_classes):
    distance = None
    for i in range(num_classes):
        if i in lack_class:
            continue
        else:
            prototype_ = torch.cat(
                (prototype[i:i + 1].detach(), prototype[0:i].detach(),
                 prototype[i + 1:].detach()), 0)
            distance_ = torch.einsum('nc,kc->nk', [
                nn.functional.normalize(class_causal[i], dim=1),
                nn.functional.normalize(prototype_, dim=1)
            ])
            #distance_ /= 5
            if distance is None:
                distance = F.softmax(distance_, dim=1)
            else:
                distance = torch.cat((distance, F.softmax(distance_, dim=1)),
                                     0)
    labels = torch.zeros(distance.shape[0], dtype=torch.long).cuda()
    loss = criterion(distance, labels)
    return loss

def local_ssl(prototype, memory_bank, args, class_causal, lack_class):
    nce = None
    for key in class_causal:
        class_causal[key] = args.constraint * class_causal[key].float() + (
            1 - args.constraint) * prototype[key].float().detach()
    for i in lack_class:
        _ = class_causal.pop(i)
    for i in range(args.num_classes):
        if i in lack_class:
            continue
        else:
            pos = class_causal[i][0:1]
            class_causal_ = class_causal.copy()
            _ = class_causal_.pop(i)
            if class_causal_ == {}:
                hard_neg = memory_bank[i].detach()
            else:
                neg = torch.cat(list(class_causal_.values()))
                # prototype_=self.prototype.clone().detach()
                distance = F.softmax(torch.einsum(
                    'kc,nc->kn', [prototype[i:i + 1].detach(), neg]),
                                     dim=1)
                dis, idx = torch.sort(distance)
                if len(idx[0]) < 10:
                    hard_neg = torch.cat(
                        (memory_bank[i][0:(10 - len(distance[0])), :].detach(),
                         neg), 0)
                else:
                    hard_neg = neg[idx[0][0:10]]
                # 检查 hard_neg 的形状是否与 pos 一致
                if hard_neg.shape[1] != pos.shape[1]:
                    # 如果不一致，可以选择截断或填充
                    if hard_neg.shape[1] > pos.shape[1]:
                        hard_neg = hard_neg[:, :pos.shape[1]]
                    else:
                        padding = torch.zeros(hard_neg.shape[0], pos.shape[1] - hard_neg.shape[1], device=hard_neg.device)
                        hard_neg = torch.cat((hard_neg, padding), dim=1)

            memory_bank[i] = hard_neg
            sample = torch.cat((pos, hard_neg), 0)
            nce_ = F.softmax(torch.einsum('nc,kc->nk',
                                          [class_causal[i], sample]),
                             dim=1)
            if nce is None:
                nce = nce_
            else:
                nce = torch.cat((nce, nce_), 0)
    labels = torch.zeros(nce.shape[0], dtype=torch.long).cuda()
    loss = criterion(nce, labels)
    return loss, memory_bank

def eval(model, loader, device, args):
    model.eval()
    all_probs = []  # 收集正类概率
    all_preds = []  # 收集类别标签
    all_labels = []
    total_loss = 0
    total_local_loss = 0  # 新增：用于记录局部对比损失
    total_global_loss = 0  # 新增：用于记录全局对比损失
    for data in loader:
        data = data.to(device)
        if data.x.shape[0] == 1:
            continue
        with torch.no_grad():
            pred = model.eval_forward(data)  # 输出 logits
            # softmax 概率分布
            if args.domain in ["size", "color"]:
                pred = torch.nn.functional.log_softmax(pred, dim=-1)
            else:
                pred = torch.nn.functional.softmax(pred, dim=1)
            prob = pred[:, 1]  # 正类的概率
            all_probs.append(prob.cpu().numpy())
            all_preds.append(pred.argmax(dim=1).cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

            one_hot_target = data.y.view(-1).type(torch.int64)
            pred_loss = criterion(pred, one_hot_target)
            total_loss += pred_loss.item() * num_graphs(data)

            # 对比学习损失（局部和全局）
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

        # 样本数
    num = len(loader.dataset)
    avg_pred_loss = total_loss / num
    avg_local_loss = total_local_loss / num if args.local else 0
    avg_global_loss = total_global_loss / num if args.global_ else 0
    avg_loss = args.pred * avg_pred_loss + args.l * avg_local_loss + args.g * avg_global_loss

    # 拼接预测概率、标签
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 计算各项指标
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0

    try:
        aupr = average_precision_score(all_labels, all_probs)
    except:
        aupr = 0

    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')

    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    else:
        specificity = 0  # 防止全为一类时报错

    acc = np.mean(all_preds == all_labels)

    return auc, aupr, f1, precision, recall, specificity, acc


if __name__ == "__main__":
    # 最优参数配置
    best_params = {
        'lr': 0.0001,
        'min_lr': 1e-05,
        'weight_decay': 7.022021857867895e-05,
        'hidden': 264,
        'epochs': 521,
        'batch_size': 32,
        'pretrain': 10,
        'constraint': 0.6438765275781118,
        'layers': 3,
        'g': 0.7366938657887405,
        'l': 0.18429794975881822,
        'pred': 0.844044437643158
    }


    # 创建参数对象
    class Args:
        def __init__(self, params):
            self.hidden_in = 33  # 输入特征维度
            self.num_classes = 2  # 二分类
            self.hidden = params['hidden']
            self.layers = params['layers']
            self.cls_layer = 2  # 分类层数
            self.global_ = True
            self.local = True
            self.domain = "basis"
            self.constraint = params['constraint']
            self.pred = params['pred']
            self.g = params['g']
            self.l = params['l']
            self.batch_size = int(params['batch_size'])


    args = Args(best_params)

    # 初始化模型
    model = Causal(
        hidden_in=args.hidden_in,
        hidden_out=args.num_classes,
        hidden=args.hidden,
        num_layer=args.layers,
        cls_layer=args.cls_layer
    ).to(device).float()

    # 加载训练好的模型
    model = load_model(model)
    if model is None:
        exit(1)

    # 加载独立测试集并确保数据类型一致
    try:
        independent_data = torch.load('./data_processed/independent dataset.pt')
        print(f"Loaded independent test set with {len(independent_data)} samples")

        # 强制所有节点特征为Float类型
        for graph_dict in independent_data:
            graph_dict["x"] = graph_dict["x"].float()

    except Exception as e:
        print(f"Error loading independent test set: {e}")
        exit(1)

    # 转换数据格式
    test_data_list = []
    for graph_dict in independent_data:
        graph_data = Data(
            x=graph_dict["x"].float(),
            edge_index=graph_dict["edge_index"],
            edge_attr=graph_dict["edge_attr"],
            y=graph_dict["y"]
        )
        test_data_list.append(graph_data)

    # 创建测试集DataLoader
    test_loader = DataLoader(
        test_data_list,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=torch_geometric.data.Batch.from_data_list
    )

    # 评估模型
    auc, aupr, f1, precision, recall, specificity, acc = eval(model, test_loader, device, args)

    # 打印结果
    print("\n=== Independent Test Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"AUPR: {aupr:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")


