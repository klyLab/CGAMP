import torch
import argparse
from sklearn.utils import shuffle
import os


def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Merge binary classification protein graph data (.pt files)')
    parser.add_argument('--task', type=str, required=True, choices=['binary'],
                        help='Task type: currently only supports "binary" (binary classification)')
    parser.add_argument('--pos_pt_path', type=str, required=True,
                        help='Path to positive sample .pt file (peptides)')
    parser.add_argument('--neg_pt_path', type=str, required=True,
                        help='Path to negative sample .pt file (non-peptides)')
    parser.add_argument('--merged_save_path', type=str, required=True,
                        help='Save path for the merged .pt file')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for shuffling data, default is 42')

    # Parse arguments
    args = parser.parse_args()

    # Check if input files exist
    if not os.path.exists(args.pos_pt_path):
        raise FileNotFoundError(f"Positive sample file does not exist: {args.pos_pt_path}")
    if not os.path.exists(args.neg_pt_path):
        raise FileNotFoundError(f"Negative sample file does not exist: {args.neg_pt_path}")

    # Create save directory if it doesn't exist
    save_dir = os.path.dirname(args.merged_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Load positive and negative sample data
    print(f"Loading positive sample data: {args.pos_pt_path}")
    graph_peptides = torch.load(args.pos_pt_path)
    print(f"Loading negative sample data: {args.neg_pt_path}")
    graph_non_peptides = torch.load(args.neg_pt_path)

    # Assign labels to each graph
    for graph in graph_peptides:
        graph["y"] = torch.tensor([1])  # Label 1 for peptides
    for graph in graph_non_peptides:
        graph["y"] = torch.tensor([0])  # Label 0 for non-peptides

    # Merge the two datasets
    all_graphs = graph_peptides + graph_non_peptides

    # Shuffle the merged dataset
    shuffled_graphs = shuffle(all_graphs, random_state=args.random_state)

    # Save the shuffled merged data
    torch.save(shuffled_graphs, args.merged_save_path)
    print(f"Merged data saved to: {args.merged_save_path}")

    # Verify the saved data
    combined_graphs = torch.load(args.merged_save_path)
    print(f"Total number of graph data samples: {len(combined_graphs)}")
    print(f"Data structure of the first graph: {combined_graphs[0]}")


if __name__ == "__main__":
    main()