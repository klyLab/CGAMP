import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch.optim.lr_scheduler import CosineAnnealingLR
from ogb.graphproppred import Evaluator
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from CGAMP_model.utils.util import print_args, set_seed
from CGAMP_model.Net.model_mol import Causal
import time
import warnings
from torch_geometric.data import DataLoader as PyGDataLoader
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix
from collections import Counter
import argparse
warnings.filterwarnings('ignore')

def check_sample_distribution(loader):
    all_labels = []
    for data in loader:
        all_labels.extend(data.y.cpu().numpy())
    return Counter(all_labels)


def eval(model, loader, device, args, evaluator):
    model.eval()
    all_probs = []  # Positive class probabilities
    all_preds = []  # Predicted labels
    all_labels = []
    total_loss = 0
    total_local_loss = 0
    total_global_loss = 0
    total_graphs = 0

    for data in loader:
        data = data.to(device)

        original_y = data.y.clone()
        original_batch_size = num_graphs(data)

        if data.x.shape[0] == 1:
            continue
        with torch.no_grad():
            pred = model.eval_forward(data)  # Output logits

            pred_batch_size = pred.size(0)
            target_batch_size = original_batch_size

            if pred_batch_size != target_batch_size:
                print(f"Warning: pred batch_size ({pred_batch_size}) != target batch_size ({target_batch_size}), skipping loss calculation")
                if args.domain in ["size", "color"]:
                    pred = torch.nn.functional.log_softmax(pred, dim=-1)
                else:
                    pred = torch.nn.functional.softmax(pred, dim=1)
                prob = pred[:, 1]  # Positive class probability
                all_probs.append(prob.cpu().numpy())
                all_preds.append(pred.argmax(dim=1).cpu().numpy())
                all_labels.append(original_y.cpu().numpy())
                continue

            # Calculate probabilities
            if args.domain in ["size", "color"]:
                pred = torch.nn.functional.log_softmax(pred, dim=-1)
            else:
                pred = torch.nn.functional.softmax(pred, dim=1)
            prob = pred[:, 1]  # Positive class probability
            all_probs.append(prob.cpu().numpy())
            all_preds.append(pred.argmax(dim=1).cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

            # Calculate loss
            one_hot_target = data.y.view(-1).type(torch.int64)
            pred_loss = F.cross_entropy(pred, one_hot_target)
            total_loss += pred_loss.item() * num_graphs(data)

            num_graph = num_graphs(data)
            total_loss += pred_loss.item() * num_graph
            total_graphs += num_graph


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

    # Calculate average loss
    num = len(loader.dataset)
    avg_pred_loss = total_loss / num
    avg_local_loss = total_local_loss / num if args.local else 0
    avg_global_loss = total_global_loss / num if args.global_ else 0
    avg_loss = args.pred * avg_pred_loss + args.l * avg_local_loss + args.g * avg_global_loss

    # Concatenate all results
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate metrics
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
        specificity = 0.0

    acc = np.mean(all_preds == all_labels)

    return auc, aupr, f1, precision, recall, specificity, avg_loss, acc


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss().to(device)


def main(args):
    set_seed(args.seed)

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

    for data in data_list:
        data.x = data.x.float()  # Ensure float32 type

    filtered_data_list = []
    for data in data_list:
        data.x = data.x.float()  # Ensure float32 type
        if data.x.shape[0] > 1:
            filtered_data_list.append(data)
    
    print(f"Filtered out {len(data_list) - len(filtered_data_list)} single-node samples")
    data_list = filtered_data_list

    # Split into training (80%) and test (20%) sets
    all_labels = torch.cat([data.y for data in data_list], dim=0).cpu().numpy()
    train_indices, test_indices = train_test_split(
        range(len(data_list)),
        test_size=0.2,
        random_state=args.seed,
        stratify=all_labels
    )
    train_dataset = [data_list[i] for i in train_indices]
    test_dataset = [data_list[i] for i in test_indices]

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

    print(f"Training set distribution: {check_sample_distribution(train_loader)} (Number of samples: {len(train_dataset)})")
    print(f"Test set distribution: {check_sample_distribution(test_loader)} (Number of samples: {len(test_dataset)})")

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

    # Track best results
    best_test_result = None
    best_test_acc = 0.0
    train_losses = []
    test_losses = []
    train_aucs = []
    test_aucs = []

    start_time = time.time()
    print("\nStart training...")
    for epoch in range(1, args.epochs + 1):
        start_epoch = time.time()
        model.train()
        total_loss = 0.0
        total_loss_p = 0.0
        total_loss_global = 0.0
        total_loss_local = 0.0
        total_graphs = 0  
        memory_bank = torch.randn(args.num_classes, 10, args.hidden).cuda()

        for step, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            data.x = data.x.float()

            original_y = data.y.clone()
            original_batch_size = num_graphs(data)

            if data.x.shape[0] == 1 and original_batch_size == 1:
                continue
                        
            causal = model.forward_causal(data)
            pred = model(causal)

            pred_batch_size = pred.size(0)
            if pred_batch_size != original_batch_size:
                print(f"Warning: Training batch {step} - pred batch_size ({pred_batch_size}) != target batch_size ({original_batch_size}), skipping")
                continue

            # Calculate prediction loss
            if args.domain in ["size", "color"]:
                pred = torch.nn.functional.log_softmax(pred, dim=-1)
            else:
                pred = torch.nn.functional.softmax(pred, dim=1)
            one_hot_target = data.y.view(-1).type(torch.int64)
            pred_loss = F.cross_entropy(pred, one_hot_target)

            # Contrastive loss
            global_loss = 0.0
            local_loss = 0.0
            if args.global_ or args.local:
                class_causal, lack_class = class_split(data.y, causal, args)
                prototype = prototype_update(None, args.num_classes, class_causal, lack_class)
                if args.global_:
                    global_loss = global_ssl(prototype, class_causal, lack_class, args.num_classes)
                if args.local:
                    local_loss, memory_bank = local_ssl(prototype, memory_bank, args, class_causal, lack_class)

            # Total loss
            loss = args.pred * pred_loss + args.g * global_loss + args.l * local_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            # Accumulate loss
            batch_size = num_graphs(data)
            total_loss += loss.item() * batch_size
            total_loss_p += pred_loss.item() * batch_size
            total_loss_global += global_loss.item() * batch_size
            total_loss_local += local_loss.item() * batch_size

        # Calculate average training loss
        num_train = len(train_loader.dataset)
        avg_train_loss = total_loss / num_train
        avg_train_p = total_loss_p / num_train
        avg_train_g = total_loss_global / num_train
        avg_train_l = total_loss_local / num_train

        # Evaluate on training and test sets
        train_result = eval(model, train_loader, device, args, evaluator)
        test_result = eval(model, test_loader, device, args, evaluator)
        train_auc, train_aupr, train_f1, train_precision, train_recall, train_specificity, train_loss, train_acc = train_result
        test_auc, test_aupr, test_f1, test_precision, test_recall, test_specificity, test_loss, test_acc = test_result

        # Record metrics
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        train_aucs.append(train_auc)
        test_aucs.append(test_auc)

        # Update best results
        if test_acc > best_test_acc and epoch > args.pretrain:
            best_test_acc = test_acc
            best_test_result = test_result
            best_model_state = model.state_dict().copy()
            print(f"Epoch {epoch} Update best model (Test Acc: {test_acc:.4f})")

        # Print epoch information
        print(f"Epoch [{epoch}/{args.epochs}] | "
              f"Loss: {avg_train_loss:.4f} = pred({avg_train_p:.4f}) + global({avg_train_g:.4f}) + local({avg_train_l:.4f}) | "
              f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
              f"Best Test Acc: {best_test_acc:.4f} | "
              f"Time: {(time.time() - start_epoch) / 60:.2f}min")

        lr_scheduler.step()

    if best_model_state is not None:
        save_path = './trained_models/binary_model.pth'
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'best_test_acc': best_test_acc,
            'best_test_result': best_test_result,
            'args': args  
        }, save_path)
        print(f"\nBest model saved to: {save_path}")
    else:
        print("\nNo optimal model found (epoch > pretrain condition may not be satisfied)")
    # Print total training time
    total_time = time.time() - start_time
    print(f"\nTotal training time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

    # Print total training time
    total_time = time.time() - start_time
    print(f"\nTotal training time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

    # Print best test set metrics
    if best_test_result is not None:
        auc, aupr, f1, precision, recall, specificity, loss, acc = best_test_result
        print("\n==================== Best Test Set Metrics ====================")
        print(f"AUC:         {auc:.4f}")
        print(f"AUPR:        {aupr:.4f}")
        print(f"F1 Score:    {f1:.4f}")
        print(f"Precision (P):   {precision:.4f}")
        print(f"Recall (R):   {recall:.4f}")
        print(f"Specificity:      {specificity:.4f}")
        print(f"Loss:        {loss:.4f}")
        print(f"Accuracy (Acc): {acc:.4f}")
        print("=======================================================")

    return best_test_result


def config_and_run(args):
    print_args(args)
    best_result = main(args)
    return best_result


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

    # 1. Path-related
    parser.add_argument('--data_dir', type=str, default='./data_processed/binary_train',
                        help='Folder of .pt graph data (default: ./data_processed/binary_train)')

    # 2. Task/Data-related
    parser.add_argument('--domain', type=str, default='basis', help='Domain setting')
    parser.add_argument('--eval_name', type=str, default='ogbg-molhiv',
                        help='Eval name for OGB Evaluator (default: ogbg-molhiv)')  # Added missing parameter

    # 3. Training-related
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=521, help='Training epochs (default: 521)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate (default: 1e-5)')
    parser.add_argument('--weight_decay', type=float, default=0.0002664653812470226, help='Weight decay')
    parser.add_argument('--pretrain', type=int, default=52, help='Start epoch for best result tracking')

    # 4. Model structure
    parser.add_argument('--layers', type=int, default=2, help='GNN layers (default: 2)')
    parser.add_argument('--hidden', type=int, default=396, help='Model hidden dimension (default: 396)')
    parser.add_argument('--hidden_in', type=int, default=33, help='Input feature dimension (default: 33)')
    parser.add_argument('--cls_layer', type=int, default=2, help='Classifier layers (default: 2)')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes (default: 2)')

    # 5. SSL weight parameters
    parser.add_argument('--constraint', type=float, default=0.6741441799446165, help='Constraint coefficient')
    parser.add_argument('--g', type=float, default=0.2342299235496516, help='Global SSL weight')
    parser.add_argument('--l', type=float, default=0.111417383510663, help='Local SSL weight')
    parser.add_argument('--pred', type=float, default=0.7617102814256653, help='Prediction loss weight')

    # 6. SSL switches
    parser.add_argument('--global_', action='store_true', default=True, help='Enable global SSL (default: True)')
    parser.add_argument('--local', action='store_true', default=True, help='Enable local SSL (default: True)')

    args = parser.parse_args()

    config_and_run(args)
    print(f"\nExperiment finished! Data used from: {args.data_dir}")