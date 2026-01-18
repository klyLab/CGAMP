from CGAMP import protein_init
from Bio import SeqIO
import torch
import os
import argparse
import ast

def tuple_type(s):
    try:
        value = ast.literal_eval(s)
        if not isinstance(value, tuple):
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid tuple value: {s}")
    return value

def list_type(s):
    try:
        value = ast.literal_eval(s)
        if not isinstance(value, list):
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid list value: {s}")
    return value

def read_fasta_with_labels(fasta_file_path, label=0):
    sequences = []
    labels = []
    if not os.path.exists(fasta_file_path):
        raise FileNotFoundError(f"FASTA file not found: {fasta_file_path}\nPlease check if the file is in {os.path.dirname(fasta_file_path)}")
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        sequences.append(str(record.seq))
        labels.append(label)
    print(f"Successfully read {len(sequences)} sequences from {fasta_file_path} (assigned label: {label})")
    return sequences, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FASTA to Protein Graph (.pt) for CGAMP Model")

    parser.add_argument('--fasta_dir', type=str, default='./raw_data',
                        help='Folder containing raw FASTA files (default: ./raw_data, as per Usage)')
    parser.add_argument('--fasta_filename', type=str, required=True,
                        help='Name of the FASTA file to process (e.g., binary_benchmark_pos.fasta)')
    parser.add_argument('--save_dir', type=str, default='./data_processed',
                        help='Folder to save generated .pt file (auto-created if missing, default: ./data_processed)')
    parser.add_argument('--label', type=int, required=True,
                        help='Label for all sequences in FASTA (binary: 0=non-AMP, 1=AMP; multi-class: 0-10)')

    ### 2. Basic Configuration Parameters (Only Essential Items Retained)
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for graph initialization (cuda:0 or cpu, default: cuda:0)')
    parser.add_argument('--config_path', type=str, default='config.json',
                        help='Path to config file (if needed, default: config.json)')

    args = parser.parse_args()

    full_fasta_path = os.path.join(args.fasta_dir, args.fasta_filename)
    pt_filename = os.path.splitext(args.fasta_filename)[0] + '.pt'
    full_pt_path = os.path.join(args.save_dir, pt_filename)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Output folder '{args.save_dir}' is ready (auto-created if missing)")

    # Read FASTA sequences
    protein_seqs, labels = read_fasta_with_labels(full_fasta_path, label=args.label)

    # Generate/Load Protein Graph Data
    if os.path.exists(full_pt_path):
        print(f"Loading existing protein graph from {full_pt_path}...")
        protein_dict = torch.load(full_pt_path, map_location=args.device)
    else:
        print(f"Initializing protein graph (device: {args.device})... This may take a moment.")
        protein_dict = protein_init(protein_seqs)

        protein_dict['labels'] = torch.tensor(labels, dtype=torch.long)
        torch.save(protein_dict, full_pt_path)
        print(f"Protein graph saved to {full_pt_path}")

    # ---------------------- Result Verification: Print Key Information ----------------------
    print("\n=== CGAMP FASTA Conversion Summary ===")
    print(f"Input FASTA: {full_fasta_path}")
    print(f"Output .pt File: {full_pt_path}")
    print(f"Number of Sequences: {len(protein_seqs)}")
    #print(f"Graph Keys: {list(protein_dict.keys())}")
    if 'labels' in protein_dict:
        print(f"Label Distribution: Unique values = {torch.unique(protein_dict['labels'])}")
    print("=== Conversion Completed Successfully ===")
