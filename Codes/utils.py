# -------------------------------------------------------------------------
# Name: utils.py
# Coding: utf8
# Author: Xinyang Qian
# Intro: Containing frequently used functions.
# -------------------------------------------------------------------------

import os
import shutil
import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, roc_curve, \
                            matthews_corrcoef, f1_score, precision_recall_curve, auc


# 从指定的TSV（Tab-Separated Values，制表符分隔的值）文件中读取数据，并根据指定的列索引（inf_ind）提取信息
def read_tsv(filename, inf_ind, skip_1st=False, file_encoding="utf8"):
    # Return n * m matrix "final_inf" (n is the num of lines, m is the length of list "inf_ind").
    # inf_ind: 一个整数列表，表示需要从每行中提取的列的索引
    extract_inf = [] # 一个空列表，用于存储提取的信息
    with open(filename, "r", encoding=file_encoding) as tsv_f:
        if skip_1st:
            tsv_f.readline() # 使用tsv_f.readline()读取并丢弃第一行
        line = tsv_f.readline() 
        while line:
            line = line[: -1]  # Remove '\n'. 对于每一行，去除末尾的换行符
            line_list = line.split("\t") # 使用split("\t")根据制表符分割行，得到一个列值的列表line_list
            if len(line_list) <= max(inf_ind):  # Wrong parameter "inf_ind" set!
                line = tsv_f.readline()
                continue
            temp_inf = []
            for ind in inf_ind:
                temp_inf.append(line_list[ind])
            extract_inf.append(temp_inf)
            line = tsv_f.readline()
    return extract_inf # 函数最终返回extract_inf，即一个n x m的矩阵，其中n是文件中符合条件行的数量，m是inf_ind列表的长度。


def read_csv(filename, inf_ind, skip_1st=False, file_encoding="utf8"):
    # Return n * m matrix "final_inf" (n is the num of lines, m is the length of list "inf_ind").
    extract_inf = []
    with open(filename, "r", encoding=file_encoding) as csv_f:
        if skip_1st:
            csv_f.readline()
        line = csv_f.readline()
        while line:
            # Process the ',' inside quotation marks.
            temp = line.strip().split('"')
            if len(temp) % 2 == 0:
                line = csv_f.readline()
                continue
            for ind, elem in enumerate(temp):
                if ind % 2 == 1:
                    temp[ind] = temp[ind].replace(",", "\n")
            line = '"'.join(temp)
            line = line.replace(",", "\t")
            line = line.replace("\n", ",")
            line_list = line.split("\t")
            if len(line_list) <= max(inf_ind):  # Wrong parameter "inf_ind" set!
                line = csv_f.readline()
                continue
            temp_inf = []
            for ind in inf_ind:
                temp_inf.append(line_list[ind])
            extract_inf.append(temp_inf)
            line = csv_f.readline()
    return extract_inf


def correct_path(path):
    # Correct the path format to "/xxx/xxx/" or "/xxx/xxx.file".
    path = path.replace("\\", "/")
    if os.path.isdir(path) and path[-1] != "/":
        path += "/"
    return path


def create_directory(dir):
    # Create the directory if not exist.
    if not os.path.exists(dir):
        os.mkdir(dir)
    return 0


def get_last_path(path):
    # Get the filename of a file path or directory name of a directory path.
    if path[-1] == "/":
        last_path = path[path[: -1].rfind("/") + 1:]
    else:
        last_path = path[path.rfind("/") + 1:]
    return last_path


def check_bool(s):
    if type(s) == bool:
        return s
    if s.upper() == "TRUE":
        return True
    elif s.upper() == "FALSE":
        return False
    else:
        return -1


def check_tcr(tcr):
    # Check that a TCR sequence is valid.
    aa_list = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
               "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    if len(tcr) > 24 or len(tcr) < 10:
        return False
    for aa in tcr:
        if aa not in aa_list:
            return False
    if tcr[0].upper() != "C" or tcr[-1].upper() != "F":
        return False
    return True

# 返回f_dict，一个字典，其中键是氨基酸标识，值是对应的标准化后的特征向量
def get_features(filename, f_num=15):
    # Read amino acid feature file and get amino acid vectors.
    f_list = read_tsv(filename, list(range(16)), True)
    f_dict = {}
    left_num = 0
    right_num = 0
    # 如果f_num大于15，计算需要在原有特征向量前后分别填充的0的数量（left_num和right_num），以达到f_num指定的维度。
    if f_num > 15:
        left_num = (f_num - 15) // 2
        right_num = f_num - 15 - left_num
    for f in f_list:
        f_dict[f[0]] = [0] * left_num
        f_dict[f[0]] += [float(x) for x in f[1:]]
        f_dict[f[0]] += [0] * right_num
    # Z-score # 使用Z-score方法对特征进行标准化处理
    xs_map = np.zeros((len(f_dict), len(f_dict['A'])), dtype=np.float64)
    for i, aa in enumerate(sorted(f_dict.keys())):
        xs_map[i, :] = np.array(f_dict[aa], dtype=np.float64)
    for aa in sorted(f_dict.keys()):
        f_dict[aa] = list(np.array(f_dict[aa], dtype=np.float64) / np.std(xs_map, axis=0))
    # Softmax #
    # Significant deterioration in model performance after adding Softmax.
    # xs_map = np.zeros((len(f_dict), len(f_dict['A'])), dtype=np.float64)
    # for i, aa in enumerate(sorted(f_dict.keys())):
    #     xs_map[i, :] = np.array(f_dict[aa], dtype=np.float64)
    # for i in range(len(xs_map[0])):
    #     xs_map[:, i] = np.array(F.softmax(torch.Tensor(xs_map[:, i]), dim=0))
    # for i, aa in enumerate(sorted(f_dict.keys())):
    #     f_dict[aa] = list(np.array(xs_map[i, :], dtype=np.float64))
    f_dict["X"] = [0] * f_num
    return f_dict


# 平衡正负样本的数量，确保在训练模型时，每个类别的样本数相等
def data_balance(sps, lbs):
    # The data of the weak classes were resampled to balance the sample size.
    # The format of sps is [sp_0, sp_1, ...].
    # The format of lbs is [lb_0, lb_1, ...].
    pos_sps, neg_sps = [], []
    for ind, sp in enumerate(sps):
        if lbs[ind] == 1:
            pos_sps.append(sp)
        else:
            neg_sps.append(sp)
    random.shuffle(pos_sps)
    random.shuffle(neg_sps)
    pos_len, neg_len = len(pos_sps), len(neg_sps)
    if pos_len > neg_len:
        neg_sps += neg_sps[: pos_len - neg_len]
    else:
        pos_sps += pos_sps[: neg_len - pos_len]
    samples, labels = pos_sps + neg_sps, [1] * len(pos_sps) + [0] * len(neg_sps)
    return samples, labels


# 根据提供的样本序列（sps）和相应的标签（sp_lbs），以及一个特征字典（feature_dict），生成用于模型训练的输入矩阵
def generate_input_for_training(sps, sp_lbs, feature_dict, ins_num=100, feature_num=15, max_len=24):
    # Generate input matrix for training samples.
    # The format of sps is [sp_0, sp_1, ...]. 输入的样本序列列表，
    # The format of sp_k is [[tcr_0, vgene_0, frequency_0, ...], [tcr_1, vgene_1, frequency_1, ...], ...]. 每个样本包含多个TCR序列
    # The format of sp_lbs is [lb_0, lb_1, ...].
    # feature_dict: 每个氨基酸对应的特征向量的字典。
    # ins_num: 每个样本要保留的TCR序列数量的上限，默认为100。
    # feature_num: 特征向量的维度，默认为15。
    # max_len: TCR序列的最大长度，默认为24
    xs, ys = [], []
    i = 0
    # 对于每个样本
    for sp in sps:
        xs.append([[[0] * feature_num] * max_len] * ins_num) # 在xs中为当前样本添加一个空矩阵，其形状为ins_num * max_len * feature_num，初始化为0。
        ys.append(sp_lbs[i]) # 将当前样本的标签从sp_lbs添加到ys
        j = 0
        # 对于当前样本中的每个TCR序列（tcr）
        for tcr in sp:
            if tcr[0] == "amino_acid": # 去除标题行
                continue
            print(f"应该是一个字符串表示的序列，tcr[0]: {tcr[0]}")
            tcr_seq = tcr[0].replace('*', '') # 得到一个TCR序列
            if len(tcr_seq) > max_len:
                tcr_seq = tcr_seq[:max_len] # 太长，截断
            # Alignment. 填充到24
            right_num = max_len - len(tcr_seq)
            tcr_seq += "X" * right_num
            # Generate matrix.
            tcr_matrix = []
            # # 对于序列中的每个氨基酸
            for aa in tcr_seq:
                tcr_matrix.append(feature_dict[aa.upper()]) # 从feature_dict中获取每个氨基酸的特征向量，并根据TCR序列的氨基酸组成生成一个特征矩阵tcr_matrix，最终矩阵的形状为24*15
            xs[i][j] = tcr_matrix # 最终的形状为100* 24 * 15，其中i是当前样本的索引，j是当前样本中TCR序列的索引
            j += 1
        i += 1
    # for x in xs:
    #     # print(len(x), len(x[0]), len(x[0][0]))
    #     for i in range(len(x)):
    #         if len(x[i]) != 24:
    #             print("i: ", len(x[i]))
    xs = np.array(xs) # 将xs转换为NumPy数组，
    xs = xs.swapaxes(2, 3) # 然后交换第2和第3维度，以适应模型输入的期望形状，变成344*100*15*24
    ys = np.array(ys)
    return xs, ys


def generate_input_for_prediction(sp, feature_dict, ins_num=10000, feature_num=15, max_len=24):
    # Generate input matrix for each sample for prediction.
    # The format of sp is [[tcr_0, vgene_0, frequency_0, ...], [tcr_1, vgene_1, frequency_1, ...], ...].
    xs = [[[[0] * feature_num] * max_len] * ins_num]
    i = 0
    for tcr in sp:
        if tcr[0] == "amino_acid":
            continue
        tcr_seq = tcr[0].replace('*', '')
        if len(tcr_seq) > max_len:
            tcr_seq = tcr_seq[:max_len] # 太长，截断
        # Alignment.
        right_num = max_len - len(tcr_seq)
        tcr_seq += "X" * right_num
        # Generate matrix.
        tcr_matrix = []
        for aa in tcr_seq:
            tcr_matrix.append(feature_dict[aa.upper()])
        xs[0][i] = tcr_matrix
        i += 1
    xs = np.array(xs)
    xs = xs.swapaxes(2, 3)
    return xs


def compute_youden(probs, labels):
    # Compute the optimal threshold using Youden index.
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youden_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[youden_idx]
    return optimal_threshold


def evaluation(probs, labels, thres=0.5):
    # Evaluate the model performance using some metrics.
    preds = [1 if pred > thres else 0 for pred in probs]
    # Evaluation.
    # Debug #
    tp, fp, tn, fn = 0, 0, 0, 0
    for idx, p in enumerate(preds):
        if p == labels[idx]:
            if p == 1:
                tp += 1
            else:
                tn += 1
        else:
            if p == 1:
                fp += 1
            else:
                fn += 1
    # Debug #
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    y_labels_ = [1 - y for y in labels]
    y_pred_ = [1 - y for y in preds]
    specificity = recall_score(y_labels_, y_pred_)
    try:
        roc_auc = roc_auc_score(labels, probs) # Debug
    except Exception:
        roc_auc = -1
    # finally:
    #     print("Accuracy = ", acc)
    #     print("Sensitivity = ", recall)
    #     print("Specificity = ", specificity)
    #     print("MCC = ", mcc)
    #     print("AUC = ", auc) # Debug
    #     print("\n")

    # 转换为numpy数组（确保兼容性）
    labels_np = np.array(labels)
    probs_np = np.array(probs)
    # 计算AUPR
    precision_pr, recall_pr, _ = precision_recall_curve(labels_np, probs_np)
    aupr = auc(recall_pr, precision_pr)

    # print("Accuracy = ", acc)
    # print("Sensitivity = ", recall)
    # print("Specificity = ", specificity)
    # print("MCC = ", mcc)
    # print("AUC = ", roc_auc)  # Debug
    # print("AUPR = ", aupr)
    # print("\n")
    return acc, recall, specificity, mcc, roc_auc, aupr


def select_model(valid_set, thres=0.5):
    final_result = (0, 0, None)
    for group in valid_set:
        temp_probs = group[0]
        temp_lbs = group[1]
        temp_model = group[2]
        temp_preds = [1 if pred > thres else 0 for pred in temp_probs]
        temp_acc = accuracy_score(temp_lbs, temp_preds)
        temp_auc = roc_auc_score(temp_lbs, temp_probs)
        if temp_acc >= final_result[0]:
            if temp_auc > final_result[1]:
                final_result = (temp_acc, temp_auc, temp_model)
    return final_result[2]


def split_data(path, ratio, save_path=None, mode=0):
    # The samples are divided according to the ratio list.
    # mode: 0 for moving; 1 for copying.
    if sum(ratio) != 1:
        print("Wrong parameter 'ratio' set!")
        return -1
    path = correct_path(path)
    if save_path is None:
        save_path = path
    else:
        save_path = correct_path(save_path)
    files = os.listdir(path)
    random.shuffle(files)
    fold_num = len(ratio)
    temp_ind = 0
    for fold in range(fold_num):
        create_directory(save_path + str(fold))
        if (fold + 1) == fold_num:
            for ind, file in enumerate(files[temp_ind:]):
                if mode == 0:
                    os.rename(path + file, save_path + str(fold) + "/" + file)
                elif mode == 1:
                    shutil.copyfile(path + file, save_path + str(fold) + "/" + file)
                else:
                    print("Wrong parameter 'mode' set!")
        else:
            for ind, file in enumerate(files[temp_ind: temp_ind + int(len(files) * ratio[fold])]):
                if mode == 0:
                    os.rename(path + file, save_path + str(fold) + "/" + file)
                elif mode == 1:
                    shutil.copyfile(path + file, save_path + str(fold) + "/" + file)
                else:
                    print("Wrong parameter 'mode' set!")
            temp_ind = temp_ind + int(len(files) * ratio[fold])


def mark_samples(spdir, flag):
    files = os.listdir(spdir)
    for f in files:
        if flag:
            os.rename(spdir + f, spdir + "positive_" + f)
        else:
            os.rename(spdir + f, spdir + "negative_" + f)


def seed_torch(seed=2023):
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True
