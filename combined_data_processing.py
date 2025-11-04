import torch
import argparse
from sklearn.utils import shuffle
import os
from torch_geometric.data import Data
import ast
import glob


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Merge classification protein graph data (.pt files)')
    parser.add_argument('--task', type=str, required=True, choices=['binary', 'multiclass'], 
                       help='Task type (binary or multiclass classification)')
    parser.add_argument('--pos_pt_path', type=str, help='Path to positive sample .pt file (for binary classification)')
    parser.add_argument('--neg_pt_path', type=str, help='Path to negative sample .pt file (for binary classification)')
    parser.add_argument('--class_pt_paths', type=str, help='Comma-separated paths to .pt files for multiclass (e.g., "class1.pt,class2.pt,class3.pt")')
    parser.add_argument('--class_pt_dir', type=str, help='Directory containing .pt files for multiclass classification')
    parser.add_argument('--class_labels', type=str, help='Comma-separated labels for each class in multiclass (e.g., "0,1,2")')
    parser.add_argument('--class_names', type=str, help='Comma-separated class names for auto-labeling (e.g., "antibacterial,anticancer,antifungal")')
    parser.add_argument('--merged_save_path', type=str, required=True, help='Save path for merged .pt file')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for shuffling (default: 42)')
    args = parser.parse_args()

    # Check task-specific argument requirements
    if args.task == 'binary':
        if not args.pos_pt_path or not args.neg_pt_path:
            raise ValueError("For binary classification, both --pos_pt_path and --neg_pt_path must be provided")
    elif args.task == 'multiclass':
        if not args.class_pt_paths and not args.class_pt_dir:
            raise ValueError("For multiclass classification, either --class_pt_paths or --class_pt_dir must be provided")

    # Create output directory if needed
    save_dir = os.path.dirname(args.merged_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Load data from .pt files and convert to uniform dictionary format
    def load_data(pt_path):
        data = torch.load(pt_path)
        graphs = []
        for val in data.values():
            if isinstance(val, torch.Tensor):
                np_dtype = val.numpy().dtype
                if np_dtype.kind in ("S", "U"):
                    if np_dtype.kind == "S":
                        str_data = val.numpy().tobytes().decode("utf-8", errors="ignore").strip()
                    else:
                        str_data = "".join(val.numpy().tolist())
                    graphs.append(ast.literal_eval(str_data) if str_data else {})
                else:
                    graphs.append({"tensor": val})
            elif isinstance(val, dict):
                graphs.append(val)
            else:
                raise TypeError(f"Unsupported data type: {type(val)}")
        return graphs

    def format_graphs(graphs, label):
        formatted = []
        for g in graphs:
            g["y"] = torch.tensor([label])
            g["x"] = g.get("seq_feat", torch.tensor([])) 
            g["edge_attr"] = g.get("edge_weight", torch.tensor([])) 
            g.setdefault("edge_index", torch.tensor([], dtype=torch.long)) 
            formatted.append(Data(**g))
        return formatted

    all_graphs = []
    
    if args.task == 'binary':
        # Binary classification processing
        print(f"Loading positive data: {args.pos_pt_path}")
        pos_data = load_data(args.pos_pt_path)
        print(f"Loading negative data: {args.neg_pt_path}")
        neg_data = load_data(args.neg_pt_path)
        
        pos_formatted = format_graphs(pos_data, 1)
        neg_formatted = format_graphs(neg_data, 0)
        
        all_graphs = pos_formatted + neg_formatted
        print(f"Total samples: {len(all_graphs)} (pos: {len(pos_formatted)}, neg: {len(neg_formatted)})")
        
    elif args.task == 'multiclass':
        # Multiclass classification processing
        class_paths = []
        
        # Determine class paths from either directory or explicit list
        if args.class_pt_dir:
            if not os.path.exists(args.class_pt_dir):
                raise FileNotFoundError(f"Directory not found: {args.class_pt_dir}")
            
            # Get all .pt files in the directory
            class_paths = glob.glob(os.path.join(args.class_pt_dir, "*.pt"))
            if not class_paths:
                raise ValueError(f"No .pt files found in directory: {args.class_pt_dir}")
            
            class_paths.sort()  # Sort for consistent ordering
            print(f"Found {len(class_paths)} .pt files in directory: {args.class_pt_dir}")
            
            # Determine class labels
            if args.class_labels:
                class_labels = list(map(int, args.class_labels.split(',')))
                if len(class_paths) != len(class_labels):
                    raise ValueError(f"Number of .pt files ({len(class_paths)}) must match number of class labels ({len(class_labels)})")
            elif args.class_names:
                class_names = args.class_names.split(',')
                if len(class_paths) != len(class_names):
                    raise ValueError(f"Number of .pt files ({len(class_paths)}) must match number of class names ({len(class_names)})")
                
                # Create label mapping from class names
                unique_names = sorted(set(class_names))
                name_to_label = {name: i for i, name in enumerate(unique_names)}
                class_labels = [name_to_label[name] for name in class_names]
            else:
                # Auto-assign labels 0, 1, 2, ...
                class_labels = list(range(len(class_paths)))
                
        else:
            # Use explicit class paths
            class_paths = args.class_pt_paths.split(',')
            class_labels = list(map(int, args.class_labels.split(',')))
            
            if len(class_paths) != len(class_labels):
                raise ValueError(f"Number of class paths ({len(class_paths)}) must match number of class labels ({len(class_labels)})")
        
        # Load and format data for each class
        for i, (class_path, label) in enumerate(zip(class_paths, class_labels)):
            if not os.path.exists(class_path):
                raise FileNotFoundError(f"File not found: {class_path}")
            
            filename = os.path.basename(class_path)
            print(f"Loading class {label} data: {filename}")
            class_data = load_data(class_path)
            class_formatted = format_graphs(class_data, label)
            all_graphs.extend(class_formatted)
            print(f"Class {label}: {len(class_formatted)} samples")
        
        print(f"Total samples: {len(all_graphs)} from {len(class_paths)} classes")

    # Merge, shuffle and save
    print(f"Total samples before shuffling: {len(all_graphs)}")
    shuffled = shuffle(all_graphs, random_state=args.random_state)

    torch.save(shuffled, args.merged_save_path)
    print(f"Merged data saved to: {args.merged_save_path}")
    
    # Verify saved data
    verify = torch.load(args.merged_save_path)
    print(f"Verified sample count: {len(verify)}")
    
    # Count samples per class for verification
    if args.task == 'binary':
        pos_count = sum(1 for g in verify if g.y.item() == 1)
        neg_count = sum(1 for g in verify if g.y.item() == 0)
        print(f"Class distribution - Positive: {pos_count}, Negative: {neg_count}")
    else:
        class_counts = {}
        for g in verify:
            label = g.y.item()
            class_counts[label] = class_counts.get(label, 0) + 1
        print(f"Class distribution: {dict(sorted(class_counts.items()))}")
    
    print(f"First graph fields: {[k for k in verify[0].keys] if hasattr(verify[0], 'keys') else []}")


if __name__ == "__main__":
    main()
