import torch
import argparse
from sklearn.utils import shuffle
import os
from torch_geometric.data import Data
import ast


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Merge binary classification protein graph data (.pt files)')
    parser.add_argument('--task', type=str, required=True, choices=['binary'], help='Task type (binary classification)')
    parser.add_argument('--pos_pt_path', type=str, required=True, help='Path to positive sample .pt file (peptides)')
    parser.add_argument('--neg_pt_path', type=str, required=True,
                        help='Path to negative sample .pt file (non-peptides)')
    parser.add_argument('--merged_save_path', type=str, required=True, help='Save path for merged .pt file')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for shuffling (default: 42)')
    args = parser.parse_args()

    # Check input file existence
    for path in [args.pos_pt_path, args.neg_pt_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

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

    print(f"Loading positive data: {args.pos_pt_path}")
    pos_data = load_data(args.pos_pt_path)
    print(f"Loading negative data: {args.neg_pt_path}")
    neg_data = load_data(args.neg_pt_path)

    def format_graphs(graphs, label):
        formatted = []
        for g in graphs:
            g["y"] = torch.tensor([label])
            g["x"] = g.get("seq_feat", torch.tensor([]))
            g["edge_attr"] = g.get("edge_weight", torch.tensor([]))
            g.setdefault("edge_index", torch.tensor([], dtype=torch.long))
            formatted.append(Data(**g))
        return formatted

    pos_formatted = format_graphs(pos_data, 1)
    neg_formatted = format_graphs(neg_data, 0)

    # Merge, shuffle and save
    all_graphs = pos_formatted + neg_formatted
    print(f"Total samples before shuffling: {len(all_graphs)} (pos: {len(pos_formatted)}, neg: {len(neg_formatted)})")
    shuffled = shuffle(all_graphs, random_state=args.random_state)

    torch.save(shuffled, args.merged_save_path)
    print(f"Merged data saved to: {args.merged_save_path}")

    # Verify saved data
    verify = torch.load(args.merged_save_path)
    print(f"Verified sample count: {len(verify)}")
    print(f"First graph fields: {[k for k in verify[0].keys] if hasattr(verify[0], 'keys') else []}")


if __name__ == "__main__":
    main()