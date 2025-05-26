# GATTCR

GATTCR: A Graph Attention Network with Multi-Feature Fusion for Peripheral Blood TCR Repertoire Classification
------------------------------

## Introduction


<p float="left">
  <img src="./Results/4-13.png"/>
</p>

## Instructions

### Clone the repository
git clone https://github.com/shiipl/AntiBinder.git

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
amino_acid  v_gene  frequency   target_seq
CASSLTRLGVYGYTF TRBV6-6*05  0.06351 CASSLTRLGVYGYTF
CASSKREIHPTQYF  TRBV28*01(179.7)    0.043778    CASSKREIHPTQYF
CASSLEGGAAMGEKLFF   TRBV28*01(194.7)    0.039882    CASSLEGGAAMGEKLFF
CASSPPDRGAFF    TRBV28*01(179.5)    0.034422    CASSPPDRGAFF
CASSTGTAQYF TRBV19*03   0.028211    CASSTGTAQYF
CASSEALQNYGYTF  TRBV2*01(255.6) 0.027918    CASSEALQNYGYTF
CSARADRGQGYEQYF TRBV20-1*01 0.027427    CSARADRGQGYEQYF
CASSPWAATNEKLFF TRBV28*01(179.7)    0.023224    CASSPWAATNEKLFF
CAWGWTGGTYEQYF  TRBV30*05   0.019363    CAWGWTGGTYEQYF
······
```

#### Prediction

Then, we can use the Python script `./Codes/caRepertoire_prediction.py` to make predictions on the sample set 0 `./Data/Geneplus/THCA/0/` using the pre-trained model by this command:

```
python ./Codes/caRepertoire_prediction.py --network GATTCR --mode 0 --sample_dir ./Data/Geneplus/THCA/0/ --aa_file ./Data/AAidx_PCA.txt --model_file ./Models/Geneplus/THCA/GATTCR_THCA_test0.pth --record_file ./Results/GATTCR_THCA_test0.tsv
```

The metrics, accuracy, sensitivity, specificity, Matthews correlation coefficient (MCC), area under the receiver operating characteristic (ROC) curve (AUC) and area under the precision-recall curve (AUPR) , are calculated and printed as: 

```
Accuracy =  0.7017543859649122
Sensitivity =  0.7222222222222222
Specificity =  0.6666666666666666
MCC =  0.37994771924604226
AUC =  0.701058201058201
AUPR =  0.6998565729439137
```

#### Training

Users can use the Python script `./Codes/caRepertoire_prediction.py` to train their own GATTCR models on their TCR-sequencing data samples for a better prediction performance. For example, we can train the model on the THCA sample sets 1, 2, 3 and 4, by this command:

```
python ./Codes/caRepertoire_prediction.py --network GATTCR --mode 1 --sample_dir ['./Data/Geneplus/THCA/1/','./Data/Geneplus/THCA/2/','./Data/Geneplus/THCA/3/','./Data/Geneplus/THCA/4/'] --aa_file ./Data/AAidx_PCA.txt --model_file ./Results/GATTCR_THCA_test0.pth
```

After the training process, the final model can be found in `./Results/Geneplus/THCA/GATTCR_THCA_test0.pth`


