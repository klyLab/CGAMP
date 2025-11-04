import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch.optim.lr_scheduler import CosineAnnealingLR
from ogb.graphproppred import Evaluator
from torch.optim import Adam
from CGAMP.utils.util import set_seed
from CGAMP.Net.model_mol import Causal
import warnings
import json
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, Data
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, \
    confusion_matrix, precision_recall_curve, roc_curve
from collections import Counter
import argparse
warnings.filterwarnings('ignore')

class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


def check_sample_distribution(loader):
    all_labels = []
    for data in loader:
        all_labels.extend(data.y.cpu().numpy())
    return Counter(all_labels)


def eval(model, loader, device, args, evaluator, target_class):
    model.eval()
    all_probs = []  # Positive class probabilities
    all_preds = []  # Predicted labels
    all_labels = []  # True labels
    total_loss = 0
    total_local_loss = 0
    total_global_loss = 0

    for data in loader:
        data = data.to(device)
        if data.x.shape[0] == 1:
            continue
        with torch.no_grad():
            pred = model.eval_forward(data)
            batch_size = data.y.size(0)
            if pred.size(0) != batch_size:
                print(f"Warning: pred batch_size {pred.size(0)} != target batch_size {batch_size}, skipping batch")
                continue

            if args.domain in ["size", "color"]:
                pred = torch.nn.functional.log_softmax(pred, dim=-1)
            else:
                pred = torch.nn.functional.softmax(pred, dim=1)

            prob = pred[:, 1]  # Positive class probability
            all_probs.append(prob.cpu().numpy())
            all_preds.append(pred.argmax(dim=1).cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

            one_hot_target = data.y.view(-1).type(torch.int64)

            if pred.size(0) != one_hot_target.size(0):
                print(f"Warning: pred size {pred.size(0)} != target size {one_hot_target.size(0)}, skipping loss calculation")
                continue
                
            pred_loss = criterion(pred, one_hot_target)
            total_loss += pred_loss.item() * num_graphs(data)

            pred_loss = criterion(pred, one_hot_target)
            total_loss += pred_loss.item() * num_graphs(data)

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

    num = len(loader.dataset)
    avg_pred_loss = total_loss / num
    avg_local_loss = total_local_loss / num if args.local else 0
    avg_global_loss = total_global_loss / num if args.global_ else 0
    avg_loss = args.pred * avg_pred_loss + args.l * avg_local_loss + args.g * avg_global_loss

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    y_true = all_labels
    y_score = all_probs

    prec, rec, pr_thresh = precision_recall_curve(y_true, y_score)
    pr_thresh = pr_thresh[:-1]
    prec = prec[:-1]
    rec = rec[:-1]

    f1_scores = 2 * prec * rec / (prec + rec + 1e-8)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    best_th = thresholds[best_idx]
    print(f"Optimal threshold by Youden: {best_th:.4f}")

    y_pred = (y_score >= best_th).astype(int)

    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = 0

    try:
        aupr = average_precision_score(y_true, y_score)
    except:
        aupr = 0

    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    else:
        specificity = 0

    acc = np.mean(y_pred == y_true)

    return auc, aupr, f1, precision, recall, specificity, avg_loss, acc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss().to(device)


def train_with_best_params(args, target_class, best_params):
    data_path = os.path.join(args.data_dir, 'merged_multiclass.pt')
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found at {data_path}\n"
            "Run 'python data_processing.py' first, or specify --data_dir."
        )
    data_list = torch.load(data_path)
    new_data_list = []
    for graph_dict in data_list:
        y = 1 if graph_dict["y"] == target_class else 0
        graph_data = Data(
            x=graph_dict["x"].float(),
            edge_index=graph_dict["edge_index"],
            edge_attr=graph_dict["edge_attr"].float() if graph_dict["edge_attr"] is not None else None,
            y=torch.tensor([y], dtype=torch.long)
        )
        new_data_list.append(graph_data)

    train_size = int(0.8 * len(new_data_list))
    test_size = len(new_data_list) - train_size
    train_dataset, test_dataset = random_split(
        new_data_list,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=torch_geometric.data.Batch.from_data_list
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=torch_geometric.data.Batch.from_data_list
    )

    model = Causal(
        hidden_in=args.hidden_in,
        hidden_out=args.num_classes,
        hidden=args.hidden,
        num_layer=args.layers,
        cls_layer=args.cls_layer
    ).to(device).float()

    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )

    best_test_acc = 0.0
    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta)
    best_model_weights = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0

        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device)

            causal = model.forward_causal(data)
            pred = model(causal)

            if pred.size(0) != data.y.size(0):
                print(f"Warning: pred batch_size {pred.size(0)} != target batch_size {data.y.size(0)}, skipping batch")
                continue
            one_hot_target = data.y.view(-1).type(torch.int64)
            pred_loss = F.cross_entropy(pred, one_hot_target)

            global_loss, local_loss = 0.0, 0.0
            if args.global_ or args.local:
                class_causal, lack_class = class_split(data.y, causal, args)
                prototype = prototype_update(
                    prototype=None,
                    num_classes=args.num_classes,
                    class_causal=class_causal,
                    lack_class=lack_class
                )

                if args.global_:
                    global_loss = global_ssl(
                        prototype, class_causal, lack_class, args.num_classes
                    )

                if args.local:
                    memory_bank = torch.randn(
                        args.num_classes, 10, args.hidden
                    ).to(device)
                    local_loss, _ = local_ssl(
                        prototype, memory_bank, args, class_causal, lack_class
                    )

            total_loss = args.pred * pred_loss + args.g * global_loss + args.l * local_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += total_loss.item() * num_graphs(data)

        lr_scheduler.step()
        num_train_samples = len(train_loader.dataset)
        avg_train_loss = total_train_loss / num_train_samples
        print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {avg_train_loss:.4f}")
        train_metrics = eval(
            model,
            train_loader,
            device,
            args,
            Evaluator(args.eval_name),
            target_class=target_class
        )
        train_auc, train_aupr, train_f1, train_precision, train_recall, train_specificity, _, train_acc = train_metrics


        print(f"Epoch [{epoch}/{args.epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train AUC: {train_auc:.4f} | "
              f"Train F1: {train_f1:.4f} | "
              f"Train Acc: {train_acc:.4f}")

        test_metrics = eval(
            model,
            test_loader,
            device,
            args,
            Evaluator(args.eval_name),
            target_class=target_class
        )
        test_auc, test_aupr, test_f1, test_precision, test_recall, test_specificity, test_loss, test_acc = test_metrics

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_weights = model.state_dict()

        if early_stopping(test_acc):
            print(f"Early stopping at epoch {epoch}. Best test acc: {best_test_acc:.4f}")
            break

    save_file = os.path.join(args.model_save_dir, f"model_class_{target_class}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': best_model_weights,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_test_acc,
        'hyperparameters': best_params
    }, save_file)
    print(f"Class {target_class} model saved to: {save_file}")

    print(f"\nTest Metrics for Class {target_class}:")
    print(f"AUC: {test_auc:.4f}")
    print(f"AUPR: {test_aupr:.4f}")
    print(f"F1: {test_f1:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"Specificity: {test_specificity:.4f}")
    print(f"ACC: {test_acc:.4f}")

    return best_test_acc


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


def main(args):
    set_seed(args.seed)
    os.makedirs(args.params_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)

    for target_class in range(11):
        print(f"\n{'=' * 80}")
        print(f"Training model for class {target_class}")
        print(f"{'=' * 80}\n")

        params_file = os.path.join(args.params_dir, f"class_{target_class}.json")
        if not os.path.exists(params_file):
            print(f"Warning: No params for class {target_class}, skipping...")
            continue

        with open(params_file, 'r') as f:
            best_params = json.load(f)

        train_with_best_params(args, target_class, best_params)

    print("\nAll models training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CGAMP Multi-Class Classification Prediction")

    parser.add_argument('--data_dir', type=str, default='./data_processed/多分类',
                        help='Folder of .pt graph data (default: ./data_processed)')
    parser.add_argument('--params_dir', type=str, default='./trained_models/multiclass_model/params',
                        help='Folder of best hyperparameters (JSON files, default: ./best_params)')
    parser.add_argument('--model_save_dir', type=str, default='./trained_models',
                        help='Folder to save multi-class models (default: ./trained_models)')
    parser.add_argument('--root', type=str, default='./data',
                        help='Root data folder (default: ./data)')

    parser.add_argument('--domain', type=str, default='basis', help='Domain setting')
    parser.add_argument('--eval_name', type=str, default='ogbg-molhiv', help='OGB evaluator name')

    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=521, help='Training epochs (default: 521)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate (default: 1e-5)')
    parser.add_argument('--weight_decay', type=float, default=0.0002664653812470226, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--delta', type=float, default=0.001, help='Early stopping delta')

    parser.add_argument('--layers', type=int, default=2, help='GNN layers (default: 2)')
    parser.add_argument('--hidden', type=int, default=396, help='Model hidden dimension (default: 396)')
    parser.add_argument('--hidden_in', type=int, default=33, help='Input feature dimension (default: 33)')
    parser.add_argument('--cls_layer', type=int, default=2, help='Classifier layers (default: 2)')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes (default: 2)')

    parser.add_argument('--global_', action='store_true', default=True, help='Enable global SSL (default: True)')
    parser.add_argument('--local', action='store_true', default=True, help='Enable local SSL (default: True)')
    # Add missing SSL weight parameters to avoid KeyError
    parser.add_argument('--pred', type=float, default=0.7617102814256653, help='Prediction loss weight')
    parser.add_argument('--g', type=float, default=0.2342299235496516, help='Global SSL weight')
    parser.add_argument('--l', type=float, default=0.111417383510663, help='Local SSL weight')
    parser.add_argument('--constraint', type=float, default=0.6741441799446165, help='Constraint coefficient')

    args = parser.parse_args()

    main(args)