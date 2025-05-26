# -------------------------------------------------------------------------
# Name: caRepertoire_prediction_bert.py
# Coding: utf8
# Author: 
# Intro: Train and test the models for predicting caRepertoires.
# 将CNN换成Bert预训练模型，同时将MIL部分的注意力机制换成图注意力网络
# 输入：节点特征 + 距离矩阵
# 关键：GAT网络的搭建
# -------------------------------------------------------------------------
import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import gc
from Bio.Align import substitution_matrices
from Bio import Align
import json

import argparse
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


import utils
from network import DeepLION, MINN_SA, TransMIL, BiFormer, DeepLION2, DeepLION2_bert, DeepLION2_GAT, DeepLION2_GCN, DeepLION2_GAT_div, DeepLION2_mulgat_div_fre, DeepLION2_mulgat_div_fre_vgene, DeepLION2_mulgat_fre
from network import DeepLION2_mulgat_fre_vgene, Mulgat_fre_vgene_fusion, Mulgat_vgene_fusion_freq
from network import Mulgat_vgene_fusion_freq_multi_task, Mulgat_vgene_fusion_freq_meanpooling, Mulgat_vgene_fusion_freq_Hierarchical

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
        default="cuda:0",
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


# 对输入的TCR序列按照caTCR_score进行排序和过滤
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


# 从给定的目录（或目录列表）中读取样本文件，根据文件名确定样本的标签（正面或负面），并对数据进行预处理
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
                labels.append(0) # 根据文件名确定标签为0或1
            elif sp.find("positive") != -1:
                labels.append(1)
            else:
                jump_sum += 1
                continue
            sp_names.append(sp)
            # sp = d + sp # 每个样本文件的完整路径
            sp = os.path.join(d, sp)
            # sp = utils.read_tsv(sp, [3, 1, 2, 4], True)
            sp = utils.read_tsv(sp, [0, 1, 2, 3], True) # n x m的矩阵，其中n是文件中符合条件行的数量，m是inf_ind列表的长度
            # sp = utils.read_csv(sp, [0, 2, 1, 3]) # cmv数据
            # sp = sp[1:]  # 移除标题行，CMV数据
            # print(f"filter_seq: {filter_seq}")
            # print(sp[0])
            if filter_seq:
                sp = filter_sequence(sp, tcr_num) # 对输入的TCR序列按照caTCR_score进行排序和过滤
            sp = sorted(sp, key=lambda x: float(x[2]), reverse=True) # 按照频率进行排序
            if len(sp) > tcr_num:
                sp = sp[: tcr_num] # 如果没有按照caTCR_score进行过滤，这里就会按照频率进行过滤
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

# 基于多分类任务的数据读取
def read_samples_multitask(sp_dir, tcr_num, filter_seq, mask_ratio):
    # 初始化类别映射字典
    category_dict = {
        "negative": 0,
        "THCA": 1,
        "LUCA": 2,
        "GICA": 3,
        "LC": 4,
        "Melanoma": 5,
        "Cervical": 6
    }

    sp_names = []
    sps = []
    labels = []
    jump_sum = 0
    
    if type(sp_dir) != list:
        sp_dir = [sp_dir]
    mask_mat = [] if mask_ratio > 0 else None
    
    for d in sp_dir:
        for sp in os.listdir(d):
            # 根据文件名查找类别标签
            label = None
            for category, label_value in category_dict.items():
                if category in sp:
                    label = label_value
                    break
            
            if label is None:
                jump_sum += 1
                continue  # 如果没有找到匹配的类别则跳过

            labels.append(label)
            sp_names.append(sp)
            sp = os.path.join(d, sp)
            sp = utils.read_tsv(sp, [0, 1, 2, 3], True)  # 按指定列读取样本数据
            
            if filter_seq:
                sp = filter_sequence(sp, tcr_num)  # 根据caTCR_score过滤TCR序列
            sp = sorted(sp, key=lambda x: float(x[2]), reverse=True)  # 按频率排序
            if len(sp) > tcr_num:
                sp = sp[: tcr_num]  # 过滤多余的TCR
            
            if mask_ratio > 0:
                temp_mask_mat = get_mask_matrix(sp, tcr_num, mask_ratio)
                mask_mat.append(temp_mask_mat)
            
            sps.append(sp)

    return sps, labels, sp_names, mask_mat

from tqdm import tqdm
# 计算每个样本的距离矩阵，将在gat中用到
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
    start_overall_time = time.time() # 记录时间
    
    # Select the appropriate distance function based on the distance option
    # distance_function = choose_distance_function(distance_option) # 函数未定义
    dis_mat = []
    total_samples = len(sps)
    # 循环遍历每个样本
    for sp in tqdm(sps, desc="Processing samples", total=total_samples):
        seqs = []
        for i in range(len(sp)):
            seqs.append(sp[i][0]) # 取出样本中的每条序列
			
		# 计算每个样本的距离矩阵
        # Split data into chunks
        # print(f"######################chunk_num:{chunk_num}")
        # print(f"list:{list(chunks(seqs, chunk_size))}#########################")
        cdr3_chunk = list(chunks(seqs, chunk_size))[chunk_num] # 函数未定义 报错
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

# 计算距离矩阵或直接读取
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

# 计算样本中所有TCR的距离矩阵，将每个距离矩阵保存为一个.tsv文件
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
    start_overall_time = time.time() # 记录时间
    
    # Select the appropriate distance function based on the distance option
    # distance_function = choose_distance_function(distance_option) # 函数未定义
    dis_mat = []
    # 循环遍历每个样本
    for sp in sps:
        seqs = []
        for i in range(len(sp)):
            seqs.append(sp[i][0]) # 取出样本中的每条序列
			
		# 计算每个样本的距离矩阵
        # Split data into chunks
    
        cdr3_chunk = list(chunks(seqs, chunk_size))[chunk_num] # 函数未定义
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

"""
    下面这些是计算距离矩阵用到的函数
"""
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
	aligner = Align.PairwiseAligner() # 创建一个用于进行配对序列比对的 PairwiseAligner 对象。
	aligner.open_gap_score = -10 # 设定了打开一个缺口（即插入一个间隔）时的分数为 -10。这是一个惩罚分数，用于减少因插入过多间隔而得到不切实际比对结果的可能性
	aligner.substitution_matrix = substitution_matrices.load("BLOSUM45") # 指定了用于比对时的替换矩阵。BLOSUM45是一种常用的替换矩阵，特别适用于相似性较低（约45%）的蛋白质序列比对。
	aligner.mode = "global" # 设置比对模式为全局比对（global alignment），即尝试比对输入序列的全部长度。
	score_s12 = aligner.score(s1,s2) # 计算两个序列 s1 和 s2 的全局比对分数
	score11 = aligner.score(s1,s1) # 计算序列 s1 与其自身的比对分数，这通常会提供该序列可达到的最高分数
	score22 = aligner.score(s2,s2) # 计算序列 s2 与其自身的比对分数
	distance = 1- score_s12/max(score11,score22) # 通过这个公式计算归一化距离
	return distance

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
		
def dist_i_list_parallel(cdr3i, cdr3s_list, distance_function_used):
	dist_i = np.array([distance_function(cdr3i, cdr3j) for cdr3j in cdr3s_list])
	return dist_i

distance_function = choose_distance_function("BLOSUM45_score_dist")

# **************************************************************************************************************
# 读取多样性指数的函数
def extract_diversity_features(file_path):
    # 读取整个文件
    df = pd.read_csv(file_path, sep='\t')
    
    # 提取所有区域面积和斜率特征
    area_features = {row['Metric']: row['Value'] for _, row in df.iterrows() if 'area' in row['Metric']}
    slope_features = {row['Metric']: row['Value'] for _, row in df.iterrows() if 'slope' in row['Metric']}
    
    # 读取最后一行的 average_delta_lambda
    avg_delta_lambda = df.iloc[-1]['Value']

    # 合并所有特征
    features = {**area_features, **slope_features, 'average_delta_lambda': avg_delta_lambda}
    return features

def process_all_samples(divP_fpath):
    diversity_features_list = []

    if type(divP_fpath) != list:
        divP_fpath = os.listdir(divP_fpath)

    for d in divP_fpath:
        # print(f"d:{d}")
        # d = os.path.join(divP_fpath, d)
        sample_files = [f for f in os.listdir(d) if f.endswith('.tsv')]

        for file in sample_files:
            # print(f"file:{file}")
            file_path = os.path.join(d, file)
            features = extract_diversity_features(file_path)
            diversity_features_list.append(features)
        
        diversity_features_df = pd.DataFrame(diversity_features_list)
        # print(f"len_div:{len(diversity_features_df)}")
    return diversity_features_df

# 创建一个V基因到整数标签的映射字典
def create_v_gene_mapping(v_gene_list):
    v_gene_dict = {v_gene: idx for idx, v_gene in enumerate(set(v_gene_list))}
    v_gene_dict['<UNK>'] = len(v_gene_dict)  # Add <UNK> token
    return v_gene_dict

# ****************************************************************************************************************

def test_model(net, sps_dir, aa_f, model_f, tcr_num, device, filter_seq, record_r, mask_ratio):
    # Get data.
    test_sps, test_labels, test_spnames, test_mask = read_samples(sps_dir, tcr_num, filter_seq,
                                                                                  mask_ratio)
    # print(net)
    # Make predictions.
    if net == "DeepLION":
        probs = DeepLION.prediction(test_sps, model_f, aa_f, tcr_num, device)
    elif net == "TransMIL":
        probs = TransMIL.prediction(test_sps, model_f, aa_f, tcr_num, device, test_mask)
    elif net == "BiFormer":
        probs = BiFormer.prediction(test_sps, model_f, aa_f, tcr_num, device)
    elif net == "DeepLION2":
        probs = DeepLION2.prediction(test_sps, model_f, aa_f, tcr_num, device)
    elif net == "DeepLION2_bert":
        probs = DeepLION2_bert.prediction(test_sps, model_f, aa_f, tcr_num, device)
    elif net == "TCR_gat":

        # 计算距离矩阵
        test_dist_mat_file = 'test_dist_mat.npy'
        dist_mat_test = get_dist_mat(test_sps, test_dist_mat_file)

        # 读取多样性特征
        divP_fpath_test = ['/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/GICA/profiles_test/0']
        diversity_features_df_test = process_all_samples(divP_fpath_test)

        # 预测
        probs = DeepLION2_GAT_div.prediction(test_sps, dist_mat_test, model_f, tcr_num, device, diversity_features_df_test)
    elif net == "MINN_SA":
        probs = MINN_SA.prediction(test_sps, model_f, aa_f, tcr_num, device)
    elif net == "TCR_mulgat_div_fre":

        # 计算距离矩阵
        test_dist_mat_file = 'test_dist_mat.npy'
        dist_mat_test = get_dist_mat(test_sps, test_dist_mat_file)

        # 读取多样性特征
        divP_fpath_test = ['/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/LUCA/profiles_test/0']
        diversity_features_df_test = process_all_samples(divP_fpath_test)

        # 预测
        probs = DeepLION2_mulgat_div_fre.prediction(test_sps, dist_mat_test, model_f, tcr_num, device, diversity_features_df_test)
    elif net == "TCR_mulgat_div_fre_vgene":

        # 计算距离矩阵
        test_dist_mat_file = 'test_dist_mat.npy'
        dist_mat_test = get_dist_mat(test_sps, test_dist_mat_file)

        # 读取多样性特征
        divP_fpath_test = ['/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/THCA/profiles_test/0']
        # divP_fpath_test = ['/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/GICA/profiles_test_test/5']
        diversity_features_df_test = process_all_samples(divP_fpath_test)

        with open('/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/THCA/v_gene_dict.json', 'r') as f:
            v_gene_dict = json.load(f)

        # 预测
        probs = DeepLION2_mulgat_div_fre_vgene.prediction(test_sps, dist_mat_test, model_f, tcr_num, device, diversity_features_df_test, v_gene_dict)
    elif net == "TCR_mulgat_fre":
        # 计算距离矩阵
        test_dist_mat_file = '/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Covid19_unresolved/test_dist_mat_300.npy'
        dist_mat_test = get_dist_mat(test_sps, test_dist_mat_file)
        # 预测
        probs = DeepLION2_mulgat_fre.prediction(test_sps, dist_mat_test, model_f, tcr_num, device)
        
    elif net == "TCR_mulgat_fre_vgene":

        # 计算距离矩阵
        test_dist_mat_file = '/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/CMV/test_dist_mat_300.npy'
        dist_mat_test = get_dist_mat(test_sps, test_dist_mat_file)

        with open('/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/THCA/v_gene_dict.json', 'r') as f:
            v_gene_dict = json.load(f)

        # 预测
        probs = DeepLION2_mulgat_fre_vgene.prediction(test_sps, dist_mat_test, model_f, tcr_num, device, v_gene_dict)
    elif net == "Mulgat_vgene_fusion_freq":
        test_dist_mat_file = '/mnt/sdb/juhengwei/PDAC_286/Processed_PBMC/raw_data_PBMC/test_dist_mat_300.npy'
        dist_mat_test = get_dist_mat(test_sps, test_dist_mat_file)

        with open('/mnt/sdb/juhengwei/PDAC_286/Processed_PBMC/raw_data_PBMC/v_gene_dict.json', 'r') as f:
            v_gene_dict = json.load(f)
        # 预测
        # probs = Mulgat_vgene_fusion_freq.prediction(test_sps, dist_mat_test, model_f, tcr_num, device, v_gene_dict, test_labels)
        probs = Mulgat_vgene_fusion_freq.prediction_with_umap(test_sps, dist_mat_test, model_f, tcr_num, device, v_gene_dict, test_labels)
        print(probs)
    elif net == "Mulgat_vgene_fusion_freq_meanpooling":
        test_dist_mat_file = '/home/juhengwei/GAT_all/tcr-bert/DeepLION2-main/Data/Covid19_unresolved/few_shot_unseen/test_few_shot_unseen.npy'
        dist_mat_test = get_dist_mat(test_sps, test_dist_mat_file)

        with open('/home/juhengwei/GAT_all/tcr-bert/DeepLION2-main/Data/Covid19_unresolved/few_shot_unseen/few_shot_v_gene_dict_unseen.json', 'r') as f:
            v_gene_dict = json.load(f)
        # 预测
        # probs = Mulgat_vgene_fusion_freq.prediction(test_sps, dist_mat_test, model_f, tcr_num, device, v_gene_dict, test_labels)
        probs = Mulgat_vgene_fusion_freq_meanpooling.prediction_with_umap(test_sps, dist_mat_test, model_f, tcr_num, device, v_gene_dict, test_labels)
        print(f"probs:{probs}")
    elif net == "Mulgat_fre_vgene_fusion":
        # 计算距离矩阵
        test_dist_mat_file = '/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/CMV/test_dist_mat_300.npy'
        dist_mat_test = get_dist_mat(test_sps, test_dist_mat_file)

        with open('/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/CMV/v_gene_dict.json', 'r') as f:
            v_gene_dict = json.load(f)


        # 预测
        probs = Mulgat_fre_vgene_fusion.prediction(test_sps, dist_mat_test, model_f, tcr_num, device, v_gene_dict, test_labels)
        print(probs)
    elif net == "Multi_task":
        # 计算距离矩阵
        test_dist_mat_file = '/mnt/sdb/juhengwei/cancer_multitask_data/test_dist_mat_300.npy'
        dist_mat_test = get_dist_mat(test_sps, test_dist_mat_file)

        with open('/mnt/sdb/juhengwei/cancer_multitask_data/v_gene_dict.json', 'r') as f:
            v_gene_dict = json.load(f)
        
        # 预测
        probs = Mulgat_vgene_fusion_freq_multi_task.prediction(test_sps, dist_mat_test, model_f, tcr_num, device, v_gene_dict, test_labels, num_classes=7)
        print(probs)

    else:
        print("Wrong parameter 'network' set!")
        return -1
    # Evaluation.
    utils.evaluation(probs, test_labels)
    # Record prediction results.
    if record_r:
        record_prediction(test_spnames, probs, record_r)


def training_model(net, sps_dir, valid_sps_dir, tcr_num, lr, ep, dropout, log_inr, aa_f, model_f, device, loss, alpha,
                   beta, gce_q, pretraining, data_balance, filter_seq, mask_ratio):
    # Get data.
    training_sps, training_labels, training_spnames, training_mask = read_samples(sps_dir, tcr_num, filter_seq,
                                                                                  mask_ratio)
    # 多分类
    # training_sps, training_labels, training_spnames, training_mask = read_samples_multitask(sps_dir, tcr_num, filter_seq,
    #                                                                               mask_ratio)
    # print(training_sps[0]) # 这是一个列表，列表中的每个元素代表一个样本，每个样本信息也是一个列表，共有100个元素，每个元素内容大致如下['CASSVSTGVDEQYF', 'TRBV2*01', '0.001081', '0.5489162544882538']
    # print("*******************************************")
    # print(training_sps[1])
    # print(type(training_sps)) # list
    # print(len(training_sps)) # 344，样本总数
    # print(len(training_sps[0])) # 100，过滤后每个样本的TCR总数为100条
    # print(f"training_sps[0][0]:{training_sps[0][1][0]}")
    # print(f"len_labels:{len(training_labels)}")
    if data_balance:
        training_seqs, training_labels = utils.data_balance(training_sps, training_labels) # 平衡正负样本数量
    valid_sps, valid_labels, valid_spnames, valid_mask = read_samples(valid_sps_dir, tcr_num, filter_seq, mask_ratio)
    # print("valid_sps:", valid_sps)
    # Training model.
    if net == "DeepLION":
        if loss[0] == "W":
            # Pretraining.
            pre_ep = pretraining
            pre_model_f = "models/temp.pth"
            pre_loss = "CE"
            DeepLION.training(training_sps, training_labels, tcr_num, lr, pre_ep, dropout, log_inr, pre_model_f, aa_f, device,
                              pre_loss, alpha, beta, gce_q, shuffle=False)
            probs = DeepLION.prediction(training_sps, pre_model_f, aa_f, device)
            os.remove(pre_model_f)
            # Calculate weights.
            weights = []
            for ind, prob in enumerate(probs):
                weights.append(1 - abs(training_labels[ind] - prob))
            # Training.
            DeepLION.training(training_sps, training_labels, tcr_num, lr, ep, dropout, log_inr,
                              model_f, aa_f, device, loss[1:], alpha, beta, gce_q, loss_w=weights, shuffle=False)
        else:
            DeepLION.training(training_sps, training_labels, tcr_num, lr, ep, dropout, log_inr,
                              model_f, aa_f, device, loss, alpha, beta, gce_q)
    elif net == "TransMIL":
        if loss[0] == "W":
            # Pretraining.
            pre_ep = pretraining
            pre_model_f = "models/temp.pth"
            pre_loss = "CE"
            TransMIL.training(training_sps, training_labels, tcr_num, lr, pre_ep, dropout, log_inr, pre_model_f, aa_f,
                              device, pre_loss, alpha, beta, gce_q, shuffle=False)
            probs = TransMIL.prediction(training_sps, pre_model_f, aa_f, device)
            os.remove(pre_model_f)
            # Calculate weights.
            weights = []
            for ind, prob in enumerate(probs):
                weights.append(1 - abs(training_labels[ind] - prob))
            # Training.
            TransMIL.training(training_sps, training_labels, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f,
                                  device, loss[1:], alpha, beta, gce_q, loss_w=weights, shuffle=False,
                                  valid_sps=valid_sps, valid_lbs=valid_labels, attn_mask=training_mask,
                                  valid_attn_mask=valid_mask)
        else:
            TransMIL.training(training_sps, training_labels, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f,
                                  device, loss, alpha, beta, gce_q, valid_sps=valid_sps, valid_lbs=valid_labels,
                                  attn_mask=training_mask, valid_attn_mask=valid_mask)
    elif net == "BiFormer":
        BiFormer.training(training_sps, training_labels, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device,
                              valid_sps=valid_sps, valid_lbs=valid_labels)
    elif net == "DeepLION2":
        DeepLION2.training(training_sps, training_labels, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device,
                              valid_sps=valid_sps, valid_lbs=valid_labels)
    elif net == "DeepLION2_bert":
        # 这是自定义的bert模型
        DeepLION2_bert.training(training_sps, training_labels, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device,
                              valid_sps=valid_sps, valid_lbs=valid_labels)
    # gat模型
    elif net == "TCR_gat":

        # 读取多样性指数
        divP_fpath = ['/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/GICA/profiles_train/1', 
                      '/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/GICA/profiles_train/2', 
                      '/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/GICA/profiles_train/3', 
                      '/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/GICA/profiles_train/4'
                      ]
        # divP_fpath = ['/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/LUCA/profiles_test_test/5']
        diversity_features_df = process_all_samples(divP_fpath) # 356 rows × 185 columns
        # print(f"len_div:{diversity_features_df}")
        # 这里可以先计算距离矩阵
        # 距离矩阵可以用一个变量保存；也可以保存成文件，后续读入文件
        # 如果只是一个100*100的矩阵，直接存入变量应该没有问题，先试一下这种做法
        # # 得到所有样本的距离矩阵
        # dist_mat = cal_distmat(training_sps, 300, 0, distance_function, cores=4) # 一个列表，有329个元素，每个元素是一个100*100的距离矩阵
        training_dist_mat_file = 'training_dist_mat.npy'
        valid_dist_mat_file = 'valid_dist_mat.npy'
        dist_mat = get_dist_mat(training_sps, training_dist_mat_file)
        # print("valid_sps_dir:", valid_sps_dir) # None
        print(len(dist_mat[0]))

        if valid_sps_dir:
            dist_mat_val = get_dist_mat(valid_sps, valid_dist_mat_file)
        else:
             dist_mat_val = None
        # print(f"len(trainingsps):{len(training_sps)}") # 329
        # print(f"len(dist_mat):{len(dist_mat)}") # 329
        print(f"device: {device}")
        # DeepLION2_GAT.training(sps=training_sps, lbs=training_labels, adjs=dist_mat, tcr_num=tcr_num, lr=lr, ep=ep, dropout=dropout, log_inr=log_inr, model_f=model_f, aa_f=aa_f, device=device, 
        #                        valid_sps=valid_sps, valid_lbs=valid_labels, valid_mat = dist_mat_val)

        # 记录损失
        loss = DeepLION2_GAT_div.training(sps=training_sps, lbs=training_labels, adjs=dist_mat, tcr_num=tcr_num, lr=lr, ep=ep, dropout=dropout, log_inr=log_inr, model_f=model_f, aa_f=aa_f, device=device, 
                           diversity_features_df=diversity_features_df, valid_sps=valid_sps, valid_lbs=valid_labels, valid_mat=dist_mat_val)
        loss_path = model_f.replace('.pth', '')
        loss.to_csv(loss_path + "_loss.tsv", sep='\t', index=False)
        print(f"Epoch loss has been saved to {loss_path}_loss.tsv!")
    elif net == "TCR_gcn":
        # Assuming that the distance matrix calculation is also needed for the GraphNet version
        training_dist_mat_file = 'training_dist_mat.npy'
        valid_dist_mat_file = 'valid_dist_mat.npy'
        dist_mat = get_dist_mat(training_sps, training_dist_mat_file)
        # print("valid_sps_dir:", valid_sps_dir) # None
        print(len(dist_mat[0]))

        if valid_sps_dir:
            dist_mat_val = get_dist_mat(valid_sps, valid_dist_mat_file)
        else:
             dist_mat_val = None

        print(f"device: {device}")
        DeepLION2_GCN.training(sps=training_sps, lbs=training_labels, adjs=dist_mat, tcr_num=tcr_num, lr=lr, ep=ep, dropout=dropout, log_inr=log_inr, model_f=model_f, aa_f=aa_f, device=device, 
                                valid_sps=valid_sps, valid_lbs=valid_labels, valid_mat=dist_mat_val)
    elif net == "MINN_SA":
        MINN_SA.training(training_sps, training_labels, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device,
                         valid_sps=valid_sps, valid_lbs=valid_labels)
    elif net == "TCR_mulgat_div_fre":
        # 读取多样性指数
        divP_fpath = ['/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/LUCA/profiles_train/1', 
                      '/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/LUCA/profiles_train/2', 
                      '/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/LUCA/profiles_train/3', 
                      '/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/LUCA/profiles_train/4'
                      ]
        # divP_fpath = ['/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/LUCA/profiles_test_test/5']
        diversity_features_df = process_all_samples(divP_fpath) # 356 rows × 185 columns
        # print(f"len_div:{diversity_features_df}")
        # 这里可以先计算距离矩阵
        # 距离矩阵可以用一个变量保存；也可以保存成文件，后续读入文件
        training_dist_mat_file = 'training_dist_mat.npy'
        valid_dist_mat_file = 'valid_dist_mat.npy'
        dist_mat = get_dist_mat(training_sps, training_dist_mat_file)
        # print("valid_sps_dir:", valid_sps_dir) # None
        print(len(dist_mat[0]))

        if valid_sps_dir:
            dist_mat_val = get_dist_mat(valid_sps, valid_dist_mat_file)
        else:
             dist_mat_val = None
        # print(f"len(trainingsps):{len(training_sps)}") # 329
        # print(f"len(dist_mat):{len(dist_mat)}") # 329
        print(f"device: {device}")
        # DeepLION2_GAT.training(sps=training_sps, lbs=training_labels, adjs=dist_mat, tcr_num=tcr_num, lr=lr, ep=ep, dropout=dropout, log_inr=log_inr, model_f=model_f, aa_f=aa_f, device=device, 
        #                        valid_sps=valid_sps, valid_lbs=valid_labels, valid_mat = dist_mat_val)

        # 记录损失
        loss = DeepLION2_mulgat_div_fre.training(sps=training_sps, lbs=training_labels, adjs=dist_mat, tcr_num=tcr_num, lr=lr, ep=ep, dropout=dropout, log_inr=log_inr, model_f=model_f, aa_f=aa_f, device=device, 
                           diversity_features_df=diversity_features_df, valid_sps=valid_sps, valid_lbs=valid_labels, valid_mat=dist_mat_val)
        loss_path = model_f.replace('.pth', '')
        loss.to_csv(loss_path + "_loss.tsv", sep='\t', index=False)
        print(f"Epoch loss has been saved to {loss_path}_loss.tsv!")
    
    elif net == "TCR_mulgat_div_fre_vgene":
        # 读取多样性指数
        divP_fpath = ['/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/THCA/profiles_train/1', 
                      '/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/THCA/profiles_train/2', 
                      '/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/THCA/profiles_train/3', 
                      '/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/THCA/profiles_train/4'
                      ]
        # divP_fpath = ['/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/GICA/profiles_test_test/5']
        diversity_features_df = process_all_samples(divP_fpath) # 356 rows × 185 columns
        # print(f"len_div:{diversity_features_df}")
        # 这里可以先计算距离矩阵
        # 距离矩阵可以用一个变量保存；也可以保存成文件，后续读入文件
        training_dist_mat_file = 'training_dist_mat.npy'
        valid_dist_mat_file = 'valid_dist_mat.npy'
        dist_mat = get_dist_mat(training_sps, training_dist_mat_file)
        # print("valid_sps_dir:", valid_sps_dir) # None
        print(len(dist_mat[0]))

        if valid_sps_dir:
            dist_mat_val = get_dist_mat(valid_sps, valid_dist_mat_file)
        else:
             dist_mat_val = None

        print(f"device: {device}")
        # DeepLION2_GAT.training(sps=training_sps, lbs=training_labels, adjs=dist_mat, tcr_num=tcr_num, lr=lr, ep=ep, dropout=dropout, log_inr=log_inr, model_f=model_f, aa_f=aa_f, device=device, 
        #                        valid_sps=valid_sps, valid_lbs=valid_labels, valid_mat = dist_mat_val)

        v_gene_list = [tcr[1] for sample in training_sps for tcr in sample]  # Extract all V gene strings
        v_gene_dict = create_v_gene_mapping(v_gene_list)

        # 保存v_gene_dict，以便在测试时使用
        with open('/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/THCA/v_gene_dict.json', 'w') as f:
            json.dump(v_gene_dict, f)

        # 记录损失
        loss = DeepLION2_mulgat_div_fre_vgene.training(sps=training_sps, lbs=training_labels, adjs=dist_mat, tcr_num=tcr_num, lr=lr, ep=ep, dropout=dropout, log_inr=log_inr, model_f=model_f, aa_f=aa_f, device=device, 
                           diversity_features_df=diversity_features_df, v_gene_dict=v_gene_dict, valid_sps=valid_sps, valid_lbs=valid_labels, valid_mat=dist_mat_val)
        
        loss_path = model_f.replace('.pth', '')
        loss.to_csv(loss_path + "_loss.tsv", sep='\t', index=False)
        print(f"Epoch loss has been saved to {loss_path}_loss.tsv!")
    elif net == "TCR_mulgat_fre":
        # 获取距离矩阵
        training_dist_mat_file = '/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Covid19_unresolved/training_dist_mat_300.npy'
        valid_dist_mat_file = 'valid_dist_mat.npy'
        dist_mat = get_dist_mat(training_sps, training_dist_mat_file)
        # print("valid_sps_dir:", valid_sps_dir) # None
        print(len(dist_mat[0]))
        if valid_sps_dir:
            dist_mat_val = get_dist_mat(valid_sps, valid_dist_mat_file)
        else:
             dist_mat_val = None

        print(f"device: {device}")

        # 记录损失
        loss = DeepLION2_mulgat_fre.training(sps=training_sps, lbs=training_labels, adjs=dist_mat, tcr_num=tcr_num, lr=lr, ep=ep, dropout=dropout, log_inr=log_inr, model_f=model_f, aa_f=aa_f, device=device, 
                                             valid_sps=valid_sps, valid_lbs=valid_labels, valid_mat=dist_mat_val)
        
        loss_path = model_f.replace('.pth', '')
        loss.to_csv(loss_path + "_loss.tsv", sep='\t', index=False)
        print(f"Epoch loss has been saved to {loss_path}_loss.tsv!")
    elif net == "TCR_mulgat_fre_vgene":
        # print(f"len_div:{diversity_features_df}")
        # 这里可以先计算距离矩阵
        # 距离矩阵可以用一个变量保存；也可以保存成文件，后续读入文件
        training_dist_mat_file = '/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/THCA/training_dist_mat_300.npy'
        valid_dist_mat_file = '/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/CMV//valid_dist_mat.npy'
        dist_mat = get_dist_mat(training_sps, training_dist_mat_file)
        # print("valid_sps_dir:", valid_sps_dir) # None
        print(len(dist_mat[0]))
        if valid_sps_dir:
            dist_mat_val = get_dist_mat(valid_sps, valid_dist_mat_file)
        else:
             dist_mat_val = None

        print(f"device: {device}")
        # DeepLION2_GAT.training(sps=training_sps, lbs=training_labels, adjs=dist_mat, tcr_num=tcr_num, lr=lr, ep=ep, dropout=dropout, log_inr=log_inr, model_f=model_f, aa_f=aa_f, device=device, 
        #                        valid_sps=valid_sps, valid_lbs=valid_labels, valid_mat = dist_mat_val)

        v_gene_list = [tcr[1] for sample in training_sps for tcr in sample]  # Extract all V gene strings
        v_gene_dict = create_v_gene_mapping(v_gene_list)

        # 保存v_gene_dict，以便在测试时使用
        # 需要根据数据集不同来修改
        with open('/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/Geneplus/THCA/v_gene_dict.json', 'w') as f:
            json.dump(v_gene_dict, f)

        # 记录损失
        loss = DeepLION2_mulgat_fre_vgene.training(sps=training_sps, lbs=training_labels, adjs=dist_mat, tcr_num=tcr_num, lr=lr, ep=ep, dropout=dropout, log_inr=log_inr, model_f=model_f, aa_f=aa_f, device=device, 
                           v_gene_dict=v_gene_dict, valid_sps=valid_sps, valid_lbs=valid_labels, valid_mat=dist_mat_val)
        
        loss_path = model_f.replace('.pth', '')
        loss.to_csv(loss_path + "_loss.tsv", sep='\t', index=False)
        print(f"Epoch loss has been saved to {loss_path}_loss.tsv!")
    elif net == "Mulgat_vgene_fusion_freq":
        # 计算距离矩阵
        training_dist_mat_file = '/mnt/sdb/juhengwei/PDAC_286/Processed_PBMC/raw_data_PBMC/training_dist_mat_300.npy'
        valid_dist_mat_file = '/home/juhengwei/GAT_all/tcr-bert/DeepLION2-main/Data/processed_lung_cancer/v_gene_dict1.json'
        dist_mat = get_dist_mat(training_sps, training_dist_mat_file)
        print(len(dist_mat[0]))
        
        if valid_sps_dir:
            dist_mat_val = get_dist_mat(valid_sps, valid_dist_mat_file)
        else:
            dist_mat_val = None

        print(f"device: {device}")

        # 创建V基因映射字典
        v_gene_list = [tcr[1] for sample in training_sps for tcr in sample]  # 提取所有的V基因字符串
        v_gene_dict = create_v_gene_mapping(v_gene_list)

        # 保存v_gene_dict，以便在测试时使用
        with open('/mnt/sdb/juhengwei/PDAC_286/Processed_PBMC/raw_data_PBMC/v_gene_dict.json', 'w') as f:
            json.dump(v_gene_dict, f)

        # 记录损失
        loss = Mulgat_vgene_fusion_freq.training(
            sps=training_sps, 
            lbs=training_labels, 
            adjs=dist_mat, 
            tcr_num=tcr_num, 
            lr=lr, 
            ep=ep, 
            dropout=dropout, 
            log_inr=log_inr, 
            model_f=model_f, 
            aa_f=aa_f, 
            device=device, 
            v_gene_dict=v_gene_dict, 
            valid_sps=valid_sps, 
            valid_lbs=valid_labels, 
            valid_mat=dist_mat_val
        )

        loss_path = model_f.replace('.pth', '')
        loss.to_csv(loss_path + "_loss.tsv", sep='\t', index=False)
        print(f"Epoch loss has been saved to {loss_path}_loss.tsv!")        

    elif net == "Mulgat_vgene_fusion_freq_Hierarchical":
        # 计算距离矩阵
        training_dist_mat_file = '/home/juhengwei/GAT_all/tcr-bert/DeepLION2-main/Data/Geneplus/LUCA/training_test_dist_mat_300.npy'
        valid_dist_mat_file = '/home/juhengwei/GAT_all/tcr-bert/DeepLION2-main/Data/processed_lung_cancer/v_gene_dict1.json'
        dist_mat = get_dist_mat(training_sps, training_dist_mat_file)
        print(len(dist_mat[0]))
        
        if valid_sps_dir:
            dist_mat_val = get_dist_mat(valid_sps, valid_dist_mat_file)
        else:
            dist_mat_val = None

        print(f"device: {device}")

        # 创建V基因映射字典
        v_gene_list = [tcr[1] for sample in training_sps for tcr in sample]  # 提取所有的V基因字符串
        v_gene_dict = create_v_gene_mapping(v_gene_list)

        # 保存v_gene_dict，以便在测试时使用
        with open('/home/juhengwei/GAT_all/tcr-bert/DeepLION2-main/Data/Geneplus/LUCA/test_v_gene_dict.json', 'w') as f:
            json.dump(v_gene_dict, f)

        # 记录损失
        loss = Mulgat_vgene_fusion_freq_Hierarchical.training(
            sps=training_sps, 
            lbs=training_labels, 
            adjs=dist_mat, 
            tcr_num=tcr_num, 
            lr=lr, 
            ep=ep, 
            dropout=dropout, 
            log_inr=log_inr, 
            model_f=model_f, 
            aa_f=aa_f, 
            device=device, 
            v_gene_dict=v_gene_dict, 
            valid_sps=valid_sps, 
            valid_lbs=valid_labels, 
            valid_mat=dist_mat_val
        )

        loss_path = model_f.replace('.pth', '')
        loss.to_csv(loss_path + "_loss.tsv", sep='\t', index=False)
        print(f"Epoch loss has been saved to {loss_path}_loss.tsv!")     

    elif net == "Mulgat_vgene_fusion_freq_meanpooling":
        # 计算距离矩阵
        training_dist_mat_file = '/home/juhengwei/GAT_all/tcr-bert/DeepLION2-main/Data/Covid19_unresolved/few_shot_unseen/train_few_shot_unseen.npy'
        valid_dist_mat_file = '/home/juhengwei/GAT_all/tcr-bert/DeepLION2-main/Data/processed_lung_cancer/v_gene_dict1.json'
        dist_mat = get_dist_mat(training_sps, training_dist_mat_file)
        print(len(dist_mat[0]))
        
        if valid_sps_dir:
            dist_mat_val = get_dist_mat(valid_sps, valid_dist_mat_file)
        else:
            dist_mat_val = None

        print(f"device: {device}")

        # 创建V基因映射字典
        v_gene_list = [tcr[1] for sample in training_sps for tcr in sample]  # 提取所有的V基因字符串
        v_gene_dict = create_v_gene_mapping(v_gene_list)

        # 保存v_gene_dict，以便在测试时使用
        with open('/home/juhengwei/GAT_all/tcr-bert/DeepLION2-main/Data/Covid19_unresolved/few_shot_unseen/few_shot_v_gene_dict_unseen.json', 'w') as f:
            json.dump(v_gene_dict, f)

        # 记录损失
        loss = Mulgat_vgene_fusion_freq_meanpooling.training(
            sps=training_sps, 
            lbs=training_labels, 
            adjs=dist_mat, 
            tcr_num=tcr_num,
            lr=lr, 
            ep=ep, 
            dropout=dropout, 
            log_inr=log_inr, 
            model_f=model_f, 
            aa_f=aa_f, 
            device=device, 
            v_gene_dict=v_gene_dict, 
            valid_sps=valid_sps, 
            valid_lbs=valid_labels, 
            valid_mat=dist_mat_val,
            pretrained_model_path="/home/juhengwei/GAT_all/tcr-bert/DeepLION2-main/Results/art_covid_big_fix3.pth"
        )

        loss_path = model_f.replace('.pth', '')
        loss.to_csv(loss_path + "_loss.tsv", sep='\t', index=False)
        print(f"Epoch loss has been saved to {loss_path}_loss.tsv!")   

    elif net == "Mulgat_fre_vgene_fusion":
        # 三模态融合的有问题
        # 计算距离矩阵
        training_dist_mat_file = '/home/juhengwei/GAT_all/tcr-bert/DeepLION2-main/Data/CMV/training_dist_mat_300.npy'
        valid_dist_mat_file = '/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/CMV/test_v_gene_dict.json'
        dist_mat = get_dist_mat(training_sps, training_dist_mat_file)
        print(len(dist_mat[0]))
        
        if valid_sps_dir:
            dist_mat_val = get_dist_mat(valid_sps, valid_dist_mat_file)
        else:
            dist_mat_val = None

        print(f"device: {device}")

        # 创建V基因映射字典
        v_gene_list = [tcr[1] for sample in training_sps for tcr in sample]  # 提取所有的V基因字符串
        v_gene_dict = create_v_gene_mapping(v_gene_list)

        # 保存v_gene_dict，以便在测试时使用
        with open('/home/juhengwei/GAT_all/tcr-bert/DeepLION2-main/Data/CMV/v_gene_dict.json', 'w') as f:
            json.dump(v_gene_dict, f)

        # 记录损失
        loss = Mulgat_fre_vgene_fusion.training(
            sps=training_sps, 
            lbs=training_labels, 
            adjs=dist_mat, 
            tcr_num=tcr_num, 
            lr=lr, 
            ep=ep, 
            dropout=dropout, 
            log_inr=log_inr, 
            model_f=model_f, 
            aa_f=aa_f, 
            device=device, 
            v_gene_dict=v_gene_dict, 
            valid_sps=valid_sps, 
            valid_lbs=valid_labels, 
            valid_mat=dist_mat_val
        )

        loss_path = model_f.replace('.pth', '')
        loss.to_csv(loss_path + "_loss.tsv", sep='\t', index=False)
        print(f"Epoch loss has been saved to {loss_path}_loss.tsv!")
    elif net == "Multi_task":
        # 计算距离矩阵
        training_dist_mat_file = '/mnt/sdb/juhengwei/cancer_multitask_data/training_dist_mat_300.npy'
        valid_dist_mat_file = '/home/cc/cc/TCR/tcr-bert/DeepLION2-main/Data/CMV/v_gene_dict.json'
        dist_mat = get_dist_mat(training_sps, training_dist_mat_file)

        print(len(dist_mat[0]))
        if valid_sps_dir:
            dist_mat_val = get_dist_mat(valid_sps, valid_dist_mat_file)
        else:
            dist_mat_val = None

        print(f"device: {device}")

        # 创建V基因映射字典
        v_gene_list = [tcr[1] for sample in training_sps for tcr in sample]  # 提取所有的V基因字符串
        v_gene_dict = create_v_gene_mapping(v_gene_list)

        # 保存v_gene_dict，以便在测试时使用
        with open('/mnt/sdb/juhengwei/cancer_multitask_data/v_gene_dict.json', 'w') as f:
            json.dump(v_gene_dict, f)

        # 记录损失
        loss = Mulgat_vgene_fusion_freq_multi_task.training(
            sps=training_sps, 
            lbs=training_labels, 
            adjs=dist_mat, 
            tcr_num=tcr_num, 
            lr=lr, 
            ep=ep, 
            dropout=dropout, 
            log_inr=log_inr, 
            model_f=model_f, 
            aa_f=aa_f, 
            device=device, 
            v_gene_dict=v_gene_dict, 
            valid_sps=valid_sps, 
            valid_lbs=valid_labels, 
            valid_mat=dist_mat_val,
            num_classes=7
        )

        loss_path = model_f.replace('.pth', '')
        loss.to_csv(loss_path + "_loss.tsv", sep='\t', index=False)
        print(f"Epoch loss has been saved to {loss_path}_loss.tsv!")

    else:
        print("Wrong parameter 'network' set!")
        return -1


def identify_motifs(net, sps_dir, aa_f, model_f, tcr_num, device, filter_seq, record_r, mask_ratio, score_thres):
    test_sps, test_labels, test_spnames, test_mask = read_samples(sps_dir, tcr_num, filter_seq, mask_ratio)
    if net == "DeepLION2":
        results = DeepLION2.motif_identification(test_sps, test_spnames, model_f, aa_f, tcr_num, device)
        # Process sequences.
        threshold = 0.5
        positive_seqs = []
        for s in results:
            if s[0].find("positive") != -1 and float(s[2]) > threshold:
                positive_seqs.append(s)
        positive_seqs = sorted(positive_seqs, key=lambda x: x[2], reverse=True)
        # Motif visualization.
        seq_scale = 10
        basic_score = 1
        motif_thres = 0.5
        selected_seq = []
        for s in positive_seqs:
            if s[2] > score_thres:
                selected_seq.append(s)
        max_len = 0
        seq_weight = []
        for seq in selected_seq:
            temp_score = [basic_score] * len(seq[1])
            for motif in seq[3]:
                if motif[1] < motif_thres:
                    continue
                for idx in range(seq[1].find(motif[0]), seq[1].find(motif[0]) + len(motif[0])):
                    temp_score[idx] += motif[1]
            seq_weight.append(temp_score)
            if len(seq[1]) > max_len:
                max_len = len(seq[1])
        # Group the sequences.
        cluster_num = round(len(selected_seq) / seq_scale)
        seq_num_in_cluster = int(len(selected_seq) / cluster_num)
        clustered_selected_seq = []
        clustered_weight = []
        temp_index = 0
        for i in range(cluster_num):
            if i == cluster_num - 1:
                clustered_selected_seq.append(selected_seq[temp_index: ])
                clustered_weight.append(seq_weight[temp_index: ])
            else:
                clustered_selected_seq.append(selected_seq[temp_index: temp_index + seq_num_in_cluster])
                clustered_weight.append(seq_weight[temp_index: temp_index + seq_num_in_cluster])
                temp_index += seq_num_in_cluster
        # Visualization.
        seq_num = 10
        utils.create_directory(record_r)
        for c in range(0, cluster_num):
            seq_num = len(clustered_selected_seq[c])
            selected_seq = clustered_selected_seq[c][:seq_num]
            seq_weight = clustered_weight[c][:seq_num]
            fig, ax = plt.subplots()
            y_position = 0
            max_font = 24
            min_font = 12
            for i in range(len(selected_seq)):
                sequence = selected_seq[i][1]
                weight = seq_weight[i]
                x_position = 0
                for j in range(len(sequence)):
                    ax.text(x_position, y_position, sequence[j],
                            fontsize=(weight[j] - min(weight)) / (max(weight) - min(weight)) * (max_font - min_font) + min_font,
                            color=((weight[j] - min(weight)) / (max(weight) - min(weight)), 0, 1 - (weight[j] - min(weight)) / (max(weight) - min(weight))))
                    x_position += 1
                y_position -= 1
            ax.set_xlim(0, max_len)
            ax.set_ylim(y_position, 0)
            ax.axis('off')
            plt.savefig("{0}{1}.png".format(record_r, c))


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
        # Model test. 测试模型
        test_model(args.network, args.sample_dir, args.aa_file, args.model_file, args.tcr_num,
                   args.device, args.filter_sequence, args.record_file, args.mask_ratio)
    elif args.mode == 1:
        # Model training. 训练模型
        print(args.device)
        training_model(args.network, args.sample_dir, args.valid_sample_dir, args.tcr_num, args.learning_rate,
                       args.epoch, args.dropout, args.log_interval, args.aa_file, args.model_file, args.device,
                       args.loss, args.alpha, args.beta, args.gce_q, args.pretraining, args.data_balance,
                       args.filter_sequence, args.mask_ratio)
    elif args.mode == 2:
        # Key motif detection. 关键基序识别
        identify_motifs(args.network, args.sample_dir, args.aa_file, args.model_file, args.tcr_num,
                        args.device, args.filter_sequence, args.record_file, args.mask_ratio, args.score_thres)
    else:
        print("Wrong parameter 'mode' set!")


if __name__ == "__main__":
    main()
