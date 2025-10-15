~~>📋  A  README.md for code accompanying our paper CGAMP

# Graph-Based Causal Learning and Protein Language Models for Multi-Class Prediction of Antimicrobial Peptides
![Overview](./CGAMP/docs/CGAMP_arch.png)



## Local Environment Setup for running the test

First, download the repository and create the environment.<br>

### Create an environment with conda
requirement
```bash
git clone https://github.com/klyLab/CGAMP.git
cd ./CGAMP
conda env create -f ./environment.yml
```

### Download Datasets

We provide the raw amino acid sequences in FASTA format for both binary classification and multi-class classification tasks.The datasets include benchmark sets (for model training) and independent test sets (for model evaluation).

#### Binary classification datasets
- [Binary classification benchmark dataset](https://drive.google.com/file/d/1VlSR_84WguKns87Or6ppd4lP5YYXFXWH/view?usp=drive_link)
- [Binary classification independent test dataset](https://drive.google.com/file/d/15-bVYhVDDW3yMMCmxHI_Num9Tfw3hFfV/view?usp=drive_link).<br>

#### Multi-class classification datasets
- [Multi-class classification benchmark dataset](https://drive.google.com/file/d/1oO3tno3dFAFrqeXjJR3dUWphrpqPasIF/view?usp=drive_link)
- [Multi-class classification independent test dataset](https://drive.google.com/file/d/18Tgbn9Dsu3dOAOOIpdQL3VShVA_EjLys/view?usp=drive_link).<br>

### Download trained predictive models
We provide pre-trained model files for the two-stage antimicrobial peptide classification task. You can download and load them directly in the prediction code without retraining.

#### Binary classification model (Stage 1)
- [Antimicrobial Peptide Binary Classifier](https://drive.google.com/file/d/1mlRHP3s6pLEOMVozVeQ8Ye1S7jG6Xndf/view?usp=drive_link).<br>

#### Multi-class classification model (Stage 2)
- [Antimicrobial Peptide Multi-class Classifier](https://drive.google.com/file/d/1ZzXc5aqXXvtilHDZSLr32dY8YiXSpbZe/view?usp=drive_link).<br>

## Usage
Put all the downloaded dataset files into the same folder.  
We recommend creating `./raw_data` in the project root directory and placing all FASTA files there.

Convert FASTA to Protein Graph
Before running predictions or testing, you need to convert FASTA sequences into graph-structured .pt files.
```bash
#Demo 1: Binary classification positive sample (label=1):
python data_processing.py \
  --fasta_dir ./raw_data \
  --fasta_filename binary_benchmark_pos.fasta \
  --label 1 \
  --save_dir ./data_processed # Output: binary_benchmark_pos.pt
```
```bash
# Demo 2: Binary classification negative sample (label=0):
python data_processing.py \
  --fasta_dir ./raw_data \
  --fasta_filename binary_benchmark_neg.fasta \
  --label 0 \
  --save_dir ./data_processed  # Output: binary_benchmark_neg.pt
```

Merge Processed Protein Graph (.pt) Files
After converting all FASTA files to .pt, use combined_data_processing.py to merge category-specific .pt files into a single file for model input.
```bash
# Merge binary benchmark positive and negative .pt files:
python combined_data_processing.py \
  --task binary \
  --pos_pt_path ./data_processed/binary_benchmark_pos.pt \  
  --neg_pt_path ./data_processed/binary_benchmark_neg.pt \  
  --merged_save_path ./data_processed/binary_benchmark_merged.pt  
```

Demo of CGAMP on binary classification:
```bash
# Predict
python binary_prediction.py 

# Test
python binary_test.py 
```

Demo of CGAMP on multi-class classification:
```bash
# Predict
python multi_prediction.py

# Test
python multi_test.py
```

 
