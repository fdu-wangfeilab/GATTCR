import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import gc
from Bio.Align import substitution_matrices
from Bio import Align
import json
import pickle

import argparse
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


import utils
from network_test_fold import DeepLION, MINN_SA, TransMIL, BiFormer, DeepLION2, DeepLION2_bert, DeepLION2_GAT, DeepLION2_GCN, DeepLION2_GAT_div, DeepLION2_mulgat_div_fre, DeepLION2_mulgat_div_fre_vgene, DeepLION2_mulgat_fre
from network_test_fold import DeepLION2_mulgat_fre_vgene, Mulgat_fre_vgene_fusion, Mulgat_vgene_fusion_freq
from network_test_fold import Mulgat_vgene_fusion_freq_multi_task, Mulgat_vgene_fusion_freq_meanpooling, Mulgat_vgene_fusion_freq_Hierarchical

def create_parser():
    parser = argparse.ArgumentParser(
        description='Script to train and test DeepLION model for caRepertoire prediction, '
                    'and find key TCRs using the trained model.'
    )
    parser.add_argument(
        "--network",
        dest="network",
        type=str,
        help="The network used for caTCR prediction (DeepLION, MINN_SA, TransMIL, BiFormer or DeepLION2.",
        required=True,
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        type=int,
        help="The mode of script (0: model test; 1: model training; 2: key TCR detection).",
        required=True
    )
    parser.add_argument(
        "--sample_dir",
        dest="sample_dir",
        type=str,
        help="The file directory of samples.",
        required=True
    )
    parser.add_argument(
        "--valid_sample_dir",
        dest="valid_sample_dir",
        type=str,
        help="The file directory of samples used to valid the model performance in training process.",
        default=None
    )
    parser.add_argument(
        "--aa_file",
        dest="aa_file",
        type=str,
        help="The file recording animo acid vectors.",
        required=True
    )
    parser.add_argument(
        "--model_file",
        dest="model_file",
        type=str,
        help="The pre-trained DeepLION model file of for TCR prediction in .pth format (for mode 0 & 2) or "
             "the file path to save the trained model in .pth format (for mode 1).",
        required=True
    )
    parser.add_argument(
        "--tcr_num",
        dest="tcr_num",
        type=int,
        help="The number of TCRs in each sample.",
        default=300,
    )
    parser.add_argument(
        "--epoch",
        dest="epoch",
        type=int,
        help="The number of training epochs.",
        default=500,
    )
    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        type=float,
        help="The learning rate used to train DeepLION.",
        default=0.0001,
    )
    parser.add_argument(
        "--dropout",
        dest="dropout",
        type=float,
        help="The dropout value used to train DeepLION.",
        default=0.4,
    )
    parser.add_argument(
        "--log_interval",
        dest="log_interval",
        type=int,
        help="The fixed number of intervals to print training conditions",
        default=100,
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        help="The device used to make prediction.",
        default="cuda:1",
    )
    parser.add_argument(
        "--record_file",
        dest="record_file",
        type=str,
        help="Whether to record the prediction results.",
        default=None
    )
    parser.add_argument(
        "--loss",
        dest="loss",
        type=str,
        help="Which loss function used in model training (CE/SCE).",
        default="CE",
    )
    parser.add_argument(
        "--alpha",
        dest="alpha",
        type=float,
        help="The parameter alpha of SCE.",
        default=1.0,
    )
    parser.add_argument(
        "--beta",
        dest="beta",
        type=float,
        help="The parameter beta of SCE.",
        default=1.0,
    )
    parser.add_argument(
        "--gce_q",
        dest="gce_q",
        type=float,
        help="The parameter q of GCE.",
        default=0.7,
    )
    parser.add_argument(
        "--pretraining",
        dest="pretraining",
        type=int,
        help="The number of pretraining epoch.",
        default=20,
    )
    parser.add_argument(
        "--data_balance",
        dest="data_balance",
        type=str,
        help="Whether to balance the data.",
        default="",
    )
    parser.add_argument(
        "--filter_sequence",
        dest="filter_sequence",
        type=str,
        help="Whether to filter TCRs according to the scores.",
        default="False"
    )
    parser.add_argument(
        "--mask_ratio",
        dest="mask_ratio",
        type=float,
        help="The ratio of TCRs masked in the self-attention layers.",
        default=0
    )
    parser.add_argument(
        "--score_thres",
        dest="score_thres",
        type=float,
        help="The threshold of TCR scores for motif identification.",
        default=0.999
    )
    args = parser.parse_args()
    return args


def record_prediction(tcrs, probs, save_filename, sort_scores=False):
    tcr_scores = []
    for ind, tcr in enumerate(tcrs):
        tcr_scores.append([tcr, probs[ind]])
    if sort_scores:
        tcr_scores = sorted(tcr_scores, key=lambda x: float(x[1]), reverse=True)
    if os.path.exists(save_filename):
        save_filename = save_filename + "_overlap.tsv"
    with open(save_filename, "w", encoding="utf8") as save_f:
        for tcr in tcr_scores:
            save_f.write("{0}\t{1}\n".format(tcr[0], tcr[1]))
    print("The prediction results have been recorded to {}!".format(save_filename))
    return 0

def filter_sequence(tcrs, tcr_num, ind=-1):
    tcrs = sorted(tcrs, key=lambda x: float(x[ind]), reverse=True)
    if len(tcrs) > tcr_num:
        tcrs = tcrs[: tcr_num]
    return tcrs


def get_mask_matrix(tcrs, tcr_num, ratio, ind=-1):
    mask_matrix = []
    scores = []
    for tcr in tcrs:
        scores.append(tcr[ind])
    scores = sorted(scores)
    score = scores[int(ratio * len(scores))]
    temp = []
    for tcr in tcrs:
        if tcr[ind] < score:
            temp.append(True)
        else:
            temp.append(False)
    for tcr in range(tcr_num - len(tcrs)):
        temp.append(True)
    for tcr in range(tcr_num):
        mask_matrix.append(temp)
    return mask_matrix


def read_samples(sp_dir, tcr_num, filter_seq, mask_ratio):
    # Get data.
    sp_names = []
    sps = []
    labels = []
    # Read samples.
    jump_sum = 0
    if type(sp_dir) != list:
        sp_dir = [sp_dir]
    if mask_ratio > 0:
        mask_mat = []
    else:
        mask_mat = None
    for d in sp_dir:
        # print(f"d:{d}")
        for sp in os.listdir(d):
            # print(f"sp:{sp}")
            if sp.find("negative") != -1:
                labels.append(0)
            elif sp.find("positive") != -1:
                labels.append(1)
            else:
                jump_sum += 1
                continue
            sp_names.append(sp)
            # sp = d + sp
            sp = os.path.join(d, sp)
            # sp = utils.read_tsv(sp, [3, 1, 2, 4], True)
            sp = utils.read_tsv(sp, [0, 1, 2, 3], True)
            # sp = utils.read_csv(sp, [0, 2, 1, 3])
            # sp = sp[1:]
            # print(f"filter_seq: {filter_seq}")
            # print(sp[0])
            if filter_seq:
                sp = filter_sequence(sp, tcr_num) 
            sp = sorted(sp, key=lambda x: float(x[2]), reverse=True)
            if len(sp) > tcr_num:
                sp = sp[: tcr_num]
            if mask_ratio > 0:
                temp_mask_mat = get_mask_matrix(sp, tcr_num, mask_ratio)
                mask_mat.append(temp_mask_mat)
            sps.append(sp)
    # print("Jump {} files!".format(jump_sum))  # Debug #
    return sps, labels, sp_names, mask_mat

def read_samples_tyh(sp_dir):
    sps = []
    save_labels_index = []
    mask_mat = None
    sp_names = None
    labels = np.load("/beifen/Covid19/test_my_one/test_labels.npy")

    #print(sp_dir)

    #print(sp_dir)
    dirs=np.load(sp_dir[0])
    # dirs = os.listdir(sp_dir[0])

    for index,i in enumerate(dirs):
        tmp=[]
        count=0
        i=os.path.join(sp_dir[0],i)
        data=np.load(i)
        if data.shape[0]<100:
            continue
        #print(data)
        if i.split('/')[3]=='health':
            data=data[::-1]
            data=data[:,[0,1,-1]]
        else:
            data=data[:,[0,1,-1]]

        for k in data:
            
            if '*' in k[0] or '0' in k[0] or len(k[0])<5:
                #print(k)
                continue
            else:
                tmp.append(k)
                count+=1
            if count==100:
                break
        if len(tmp)==100:
        #print(tmp)
            sps.append(tmp)    
            save_labels_index.append(labels[index])

    return sps,save_labels_index,sp_names,mask_mat

from tqdm import tqdm

def cal_distmat(sps, chunk_size, chunk_num, distance_function, cores=16):
    """
    Calculate the distance matrix for a given list of sequences and specified distance function.
    
    Parameters:
    - sps: total
    - chunk_size: Size of each chunk to process separately.
    - chunk_num: The index of the chunk to process.
    - distance_option: The type of distance function to use.
    - cores: Number of cores to use for parallel processing.
    """
    start_overall_time = time.time()
    
    # Select the appropriate distance function based on the distance option
    # distance_function = choose_distance_function(distance_option)
    dis_mat = []
    total_samples = len(sps)
    for sp in tqdm(sps, desc="Processing samples", total=total_samples):
        seqs = []
        for i in range(len(sp)):
            seqs.append(sp[i][0])
			
        # Split data into chunks
        # print(f"######################chunk_num:{chunk_num}")
        # print(f"list:{list(chunks(seqs, chunk_size))}#########################")
        cdr3_chunk = list(chunks(seqs, chunk_size))[chunk_num]
        # print(cdr3_chunk)

        # Start parallel computation
        start_parallel_time = time.time()
    
        pool = mp.Pool(processes=cores, maxtasksperchild=1000)
        async_results = []

        # Apply the distance function in parallel
        for ii, cdr3_i in enumerate(cdr3_chunk):
            # print(cdr3_i)
            # print(data)
            process = pool.apply_async(dist_i_list_parallel, (cdr3_i, seqs, distance_function)) # 函数未定义
            async_results.append(process)

        # Collect results and write out
        results = []
        for process in async_results:
            function_output = process.get()
            results.append(function_output)
            del function_output

        # Clean up
        pool.close()
        pool.join()
        del async_results
        gc.collect()
		
        dis_mat.append(results)

    print("Parallel calculating time:", time.time() - start_parallel_time)
    print("Overall used time:", time.time() - start_overall_time)
    
    return dis_mat

def get_dist_mat(samples, file_path):
    if os.path.exists(file_path):
        print(f"Load dist mat...")
        with open(file_path, 'rb') as f:
            return np.load(f)
    else:
        dist_mat = cal_distmat(samples, 300, 0, distance_function, cores=4)
        with open(file_path, 'wb') as f:
            np.save(f, dist_mat)
        return dist_mat
def get_full_dist_mat(samples, tcr_num, file_path, force_recompute=False):
    if (not force_recompute) and os.path.exists(file_path):
        return np.load(file_path, allow_pickle=True)
    else:
        dist_mat = cal_distmat(samples, tcr_num, 0, distance_function, cores=24)
        np.save(file_path, dist_mat)
        return dist_mat

def cal_distmat_all(sps, chunk_size, chunk_num, distance_function, cores=4):
    """
    Calculate the distance matrix for a given list of sequences and specified distance function.
    
    Parameters:
    - sps: total
    - chunk_size: Size of each chunk to process separately.
    - chunk_num: The index of the chunk to process.
    - distance_option: The type of distance function to use.
    - cores: Number of cores to use for parallel processing.
    """
    start_overall_time = time.time()
    
    # Select the appropriate distance function based on the distance option
    # distance_function = choose_distance_function(distance_option)
    dis_mat = []

    for sp in sps:
        seqs = []
        for i in range(len(sp)):
            seqs.append(sp[i][0]) 
			
        # Split data into chunks
    
        cdr3_chunk = list(chunks(seqs, chunk_size))[chunk_num]
        # print(cdr3_chunk)

        # Start parallel computation
        start_parallel_time = time.time()
    
        pool = mp.Pool(processes=cores, maxtasksperchild=1000)
        async_results = []

        # Apply the distance function in parallel
        for ii, cdr3_i in enumerate(cdr3_chunk):
            print(cdr3_i)
            # print(data)
            process = pool.apply_async(dist_i_list_parallel, (cdr3_i, seqs, distance_function)) # 函数未定义
            async_results.append(process)

        # Collect results and write out
        results = []
        for process in async_results:
            function_output = process.get()
            results.append(function_output)
            del function_output

        # Clean up
        pool.close()
        pool.join()
        del async_results
        gc.collect()
		
        dis_mat.append(results)

    print("Parallel calculating time:", time.time() - start_parallel_time)
    print("Overall used time:", time.time() - start_overall_time)
    
    return dis_mat

def choose_distance_function(dist_option): 
	if dist_option == "AF_euclidean": 
		dist_func = Atchley_euclidean_dist
	elif dist_option == "Levenshtein":
		dist_func = levenshteinDistance
	elif dist_option == "BLOSUM45_score_dist":
		dist_func = BLOSUM45_score_dist
	return dist_func

def make_AA_to_Atchley_dict():
	"""
	REFERENCE: 
	Atchley et al. "Solving the protein sequence metric problem", 2005, PNAS, vol. 102, no. 18, pp: 6395-6400
	
	Atchley_factor_data order of the rows follows: 
	list_of_AAs=['A','C', 'D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','Z']
	each element of the list follows the format: ['AF1 AF2 AF3 AF4 AF5', ...]

	# AA_to_Atchley dict  has (letter symbol of AminoAcid  as sting, AF index as integer) as keys, and the corresponding AF as float as values
	"""
	Atchley_factor_data = ['-0.591 -1.302 -0.733 1.570 -0.146', 
	'-1.343 0.465 -0.862 -1.020 -0.255',
	'1.050 0.302 -3.656 -0.259 -3.242',
	'1.357 -1.453 1.477 0.113 -0.837',
	'-1.006 -0.590 1.891 -0.397 0.412',
	'-0.384 1.652 1.330 1.045 2.064',
	'0.336 -0.417 -1.673 -1.474 -0.078',
	'-1.239 -0.547 2.131 0.393 0.816',
	'1.831 -0.561 0.533 -0.277 1.648',
	'-1.019 -0.987 -1.505 1.266 -0.912',
	'-0.663 -1.524 2.219 -1.005 1.212',
	'0.945 0.828 1.299 -0.169 0.993',
	'0.189 2.081 -1.628 0.421 -1.392',
	'0.931 -0.179 -3.005 -0.503 -1.853',
	'1.538 -0.055 1.502 0.440 2.897',
	'-0.228 1.339 -4.760 0.670 -2.647',
	'-0.032 0.326 2.213 0.908 1.313',
	'-1.337 -0.279 -0.544 1.242 -1.262',
	'-0.595 0.009 0.672 -2.128 -0.184',
	'0.260 0.830 3.097 -0.838 1.512',
	'0.000 0.000 0.000 0.000 0.000']
	#
	list_of_AAs=['A','C', 'D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','Z']
	#
	AA_to_Atchley = dict()
	for (AA, row) in zip(list_of_AAs, Atchley_factor_data): 
		for (ii, entry) in enumerate(row.split(" "), start =1):
			AA_to_Atchley[AA, ii] = float(entry)
	#
	return AA_to_Atchley

def Atchley_euclidean_dist(s1,s2, AA_to_Atch = make_AA_to_Atchley_dict(), AF_list = [1,2,3,4,5]):
	"""
	Returns the distance calculated as euclidean distance of average AFs for each CDR3
	E.g. each AF is calculated as average AF value for the CDR3 resulting in a 5-tuple 
	corresponding to each AF. For 2 CDR3s the distance between them is calculated as 
	euleadian distance between the two 5-tuples.
	"""
	s1_list = np.array([sum(AA_to_Atch[AA, AF] for AA in list(s1))/float(len(s1)) for AF in AF_list])
	s2_list = np.array([sum(AA_to_Atch[AA, AF] for AA in list(s2))/float(len(s2)) for AF in AF_list])
	distance = np.sqrt(np.sum((np.array(s1_list)-np.array(s2_list))**2))
	del s1_list
	del s2_list
	return distance

def levenshteinDistance(s1, s2):
	if len(s1) > len(s2):
		s1, s2 = s2, s1
	#
	distances = range(len(s1) + 1)
	for i2, c2 in enumerate(s2):
		distances_ = [i2+1]
		for i1, c1 in enumerate(s1):
			if c1 == c2:
				distances_.append(distances[i1])
			else:
				distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
		distances = distances_
	return distances[-1]

def BLOSUM45_score_dist(s1,s2):
	aligner = Align.PairwiseAligner()
	aligner.open_gap_score = -10 
	aligner.substitution_matrix = substitution_matrices.load("BLOSUM45") 
	aligner.mode = "global" 
	score_s12 = aligner.score(s1,s2) 
	score11 = aligner.score(s1,s1)
	score22 = aligner.score(s2,s2) 
	distance = 1- score_s12/max(score11,score22)
	return distance

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
		
def dist_i_list_parallel(cdr3i, cdr3s_list, distance_function_used):
	dist_i = np.array([distance_function(cdr3i, cdr3j) for cdr3j in cdr3s_list])
	return dist_i

distance_function = choose_distance_function("BLOSUM45_score_dist")

def create_v_gene_mapping(v_gene_list):
    v_gene_dict = {v_gene: idx for idx, v_gene in enumerate(set(v_gene_list))}
    v_gene_dict['<UNK>'] = len(v_gene_dict)  # Add <UNK> token
    return v_gene_dict

# ****************************************************************************************************************

def test_model(net, sps_dir, aa_f, model_f, tcr_num, device, filter_seq, record_r, mask_ratio):
    # Get data.
    test_sps, test_labels, test_spnames, test_mask = read_samples(sps_dir, tcr_num, filter_seq, mask_ratio)

    test_dist_mat_file = '/home/juhengwei/GAT_all/tcr-bert/GATTCR/test_dist_mat/CMV_num_400_test_dist_mat_300.npy'
    # dist_mat_test = get_dist_mat(test_sps, test_dist_mat_file)
    dist_mat_test = get_full_dist_mat(test_sps, tcr_num=tcr_num, file_path=test_dist_mat_file)

    metric_lists = {
        'acc': [], 'recall': [], 'specificity': [], 'mcc': [], 'auc': [], 'aupr': []
    }

    v_gene_path = model_f.replace(".pth", "_v_gene_dict.pkl")

    for fold_idx in range(5):
        print(f"\n=== Fold {fold_idx + 1} ===")
        fold_model_path = model_f.replace(".pth", f"_fold{fold_idx}.pth")
        
        if not os.path.exists(v_gene_path):
            raise FileNotFoundError(f"v_gene_dict file not found at: {v_gene_path}")
        with open(v_gene_path, "rb") as f:
            v_gene_dict = pickle.load(f)
        print(f"[Info] Loaded v_gene_dict from {v_gene_path}, total {len(v_gene_dict)} entries")

        probs = Mulgat_vgene_fusion_freq.prediction_with_umap(test_sps, dist_mat_test, fold_model_path, tcr_num, device, v_gene_dict, test_labels)
        acc, recall, specificity, mcc, roc_auc, aupr = utils.evaluation(probs, test_labels)

        print(f"[Fold {fold_idx}]")
        print("Accuracy = ", acc)
        print("Sensitivity = ", recall)
        print("Specificity = ", specificity)
        print("MCC = ", mcc)
        print("AUC = ", roc_auc)
        print("AUPR = ", aupr)
        print()

        metric_lists['acc'].append(acc)
        metric_lists['recall'].append(recall)
        metric_lists['specificity'].append(specificity)
        metric_lists['mcc'].append(mcc)
        metric_lists['auc'].append(roc_auc)
        metric_lists['aupr'].append(aupr)

    print("=== Average Metrics Across Folds ===")
    for metric, values in metric_lists.items():
        values_np = np.array(values)
        print(f"{metric.upper()} = {values_np.mean():.3f} ± {values_np.std():.3f}")


def training_model(net, sps_dir, valid_sps_dir, tcr_num, lr, ep, dropout, log_inr, aa_f, model_f, device, loss, alpha,
                   beta, gce_q, pretraining, data_balance, filter_seq, mask_ratio):
    # Get data.
    training_sps, training_labels, training_spnames, training_mask = read_samples(sps_dir, tcr_num, filter_seq,
                                                                                  mask_ratio)
    full_dist_mat = get_full_dist_mat(training_sps, tcr_num=tcr_num, file_path="/home/juhengwei/GAT_all/tcr-bert/GATTCR/dist_mat_data/CMV_num_400_train_full.npy")

    if data_balance:
        training_seqs, training_labels = utils.data_balance(training_sps, training_labels) # 平衡正负样本数量

    v_gene_list = [tcr[1] for sample in training_sps for tcr in sample]
    v_gene_dict = create_v_gene_mapping(v_gene_list)
    v_gene_path = model_f.replace(".pth", "_v_gene_dict.pkl")
    if not os.path.exists(v_gene_path):
        with open(v_gene_path, "wb") as f:
            pickle.dump(v_gene_dict, f)
    print(f"[Info] v_gene_dict saved to {v_gene_path}")

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(training_sps)):
        print(f"\n=== Fold {fold_idx + 1} ===")
        
        train_sps = [training_sps[i] for i in train_idx]
        train_labels = [training_labels[i] for i in train_idx]
        val_sps = [training_sps[i] for i in val_idx]
        val_labels = [training_labels[i] for i in val_idx]

        train_dist_mat = [full_dist_mat[i] for i in train_idx]
        val_dist_mat = [full_dist_mat[i] for i in val_idx]
        
        val_freq = np.array([[float(tcr[2]) for tcr in sample] for sample in val_sps])
        val_vgene = np.array([[Mulgat_vgene_fusion_freq.v_gene_to_numeric(tcr[1], v_gene_dict) for tcr in sample] for sample in val_sps])

        loss, acc, auc, f1 = Mulgat_vgene_fusion_freq.training(
            sps=train_sps, 
            lbs=train_labels, 
            adjs=train_dist_mat, 
            tcr_num=tcr_num, 
            lr=lr, 
            ep=ep, 
            dropout=dropout, 
            log_inr=log_inr, 
            model_f=f"{model_f.replace('.pth', f'_fold{fold_idx}.pth')}", 
            aa_f=aa_f, 
            device=device, 
            v_gene_dict=v_gene_dict, 
            valid_sps=val_sps, 
            valid_lbs=val_labels, 
            valid_mat=val_dist_mat,
            valid_freq=val_freq,
            valid_vgene=val_vgene,
            shuffle=True
        )
        fold_metrics.append((acc, auc, f1))

def main():
    # Parse arguments.
    args = create_parser()
    args.sample_dir = utils.correct_path(args.sample_dir)
    if args.valid_sample_dir:
        args.valid_sample_dir = utils.correct_path(args.valid_sample_dir)
    args.aa_file = utils.correct_path(args.aa_file)
    args.model_file = utils.correct_path(args.model_file)
    if args.sample_dir.find("[") != -1:
        while type(args.sample_dir) == str:
            args.sample_dir = eval(args.sample_dir)
    args.filter_sequence = utils.check_bool(args.filter_sequence)

    # Execute the corresponding operation.
    if args.mode == 0:
        # Model test.
        test_model(args.network, args.sample_dir, args.aa_file, args.model_file, args.tcr_num,
                   args.device, args.filter_sequence, args.record_file, args.mask_ratio)
    elif args.mode == 1:
        # Model training.
        print(args.device)
        training_model(args.network, args.sample_dir, args.valid_sample_dir, args.tcr_num, args.learning_rate,
                       args.epoch, args.dropout, args.log_interval, args.aa_file, args.model_file, args.device,
                       args.loss, args.alpha, args.beta, args.gce_q, args.pretraining, args.data_balance,
                       args.filter_sequence, args.mask_ratio)
    else:
        print("Wrong parameter 'mode' set!")


if __name__ == "__main__":
    main()
