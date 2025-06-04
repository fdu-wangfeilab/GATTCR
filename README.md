# GATTCR

GATTCR: A Graph Attention Network with Multi-Feature Fusion for Peripheral Blood TCR Repertoire Classification
------------------------------

## Introduction
GATTCR is a graph-based deep learning framework designed for T-cell receptor (TCR) repertoire classification from peripheral blood samples. It integrates Graph Attention Networks (GATs) with multi-feature fusion—including sequence embeddings (via TCR-BERT), clonal frequency, and V gene usage—to model complex inter-clonal relationships and immune signatures. GATTCR offers a scalable, non-invasive approach to immune state classification and holds promise for applications in early disease detection and personalized immunotherapy.

<p float="left">
  <img src="./Results/4-13.png"/>
</p>

## Instructions

### Clone the repository
git clone https://github.com/juhengwei/GATTCR.git

### Install dependencies
pip install -r requirements.txt

### Usage

#### Data processing

First, we need to collect the raw TCR-sequencing data files in one directory, such as `./Data/RawData/Example_raw_file.tsv`, and use the Python script `./Codes/preprocess.py` to process them by this command:

```
python ./Codes/preprocess.py --sample_dir ./Data/RawData/ --info_index [-3,5,2] --aa_file ./Data/AAidx_PCA.txt --save_dir ./Results/ProcessedData/
```

After processing, TCRs with their information are saved in `./Results/ProcessedData/RawData/0/Example_raw_file.tsv_processed.tsv`:

```
amino_acid  v_gene  frequency
CASSLTRLGVYGYTF TRBV6-6*05  0.06351
CASSKREIHPTQYF  TRBV28*01(179.7)    0.043778
CASSLEGGAAMGEKLFF   TRBV28*01(194.7)    0.039882
CASSPPDRGAFF    TRBV28*01(179.5)    0.034422
CASSTGTAQYF TRBV19*03   0.028211
CASSEALQNYGYTF  TRBV2*01(255.6) 0.027918
CSARADRGQGYEQYF TRBV20-1*01 0.027427
CASSPWAATNEKLFF TRBV28*01(179.7)    0.023224
CAWGWTGGTYEQYF  TRBV30*05   0.019363
······
```

#### Prediction

Then, we can use the Python script `./Codes/caRepertoire_prediction.py` to make predictions on the sample set 0 `./Data/Geneplus/THCA/0/` using the pre-trained model by this command:

```
python ./Codes/caRepertoire_prediction.py --network GATTCR --mode 0 --sample_dir ./Data/Geneplus/THCA/0/ --aa_file ./Data/AAidx_PCA.txt --model_file ./Models/Geneplus/THCA/GATTCR_THCA_test0.pth --record_file ./Results/GATTCR_THCA_test0.tsv
```

The metric are calculated and printed as: 

```
ACC = 0.899 ± 0.009
RECALL = 0.884 ± 0.012
SPECIFICITY = 0.912 ± 0.015
MCC = 0.797 ± 0.017
AUC = 0.964 ± 0.005
AUPR = 0.955 ± 0.009
```

#### Training

Users can use the Python script `./Codes/caRepertoire_prediction.py` to train their own GATTCR models on their TCR-sequencing data samples for a better prediction performance. For example, we can train the model on the THCA sample sets 1, 2, 3 and 4, by this command:

```
python ./Codes/caRepertoire_prediction.py --network GATTCR --mode 1 --sample_dir ['./Data/Geneplus/THCA/1/','./Data/Geneplus/THCA/2/','./Data/Geneplus/THCA/3/','./Data/Geneplus/THCA/4/'] --aa_file ./Data/AAidx_PCA.txt --model_file ./Results/GATTCR_THCA_test0.pth
```

After the training process, the final model can be found in `./Results/Geneplus/THCA/GATTCR_THCA_test0.pth`


