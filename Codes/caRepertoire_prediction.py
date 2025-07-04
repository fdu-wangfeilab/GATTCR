# -------------------------------------------------------------------------
# Name: caRepertoire_prediction.py
# Coding: utf8
# Author: Xinyang Qian
# Intro: Train and test the models for predicting caRepertoires.
# -------------------------------------------------------------------------

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils
from network import DeepLION, MINN_SA, TransMIL, BiFormer, DeepLION2


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
        default=100,
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
        default=0.001,
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
        for sp in os.listdir(d):
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

def read_samples_HIV(sp_dir, tcr_num, filter_seq, mask_ratio):
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
        for sp in os.listdir(d):
            if sp.find("negative") != -1:
                labels.append(0) # 根据文件名确定标签为0或1
            elif sp.find("positive") != -1:
                labels.append(1)
            else:
                jump_sum += 1
                continue
            sp_names.append(sp)
            sp = d + sp # 每个样本文件的完整路径
            # sp = utils.read_tsv(sp, [3, 1, 2, 4], True)
            # sp = utils.read_tsv(sp, [0, 1, 2, 3], True) # n x m的矩阵，其中n是文件中符合条件行的数量，m是inf_ind列表的长度
            sp = utils.read_csv(sp, [0, 2, 1, 3]) # cmv数据
            sp = sp[1:]  # 移除标题行，CMV数据
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

def read_samples_tyh(sp_dir, lables_dir):
    sps = []
    save_labels_index = []
    mask_mat = None
    sp_names = None
    labels = np.load(lables_dir)

    #print(sp_dir)

    # print(sp_dir)
    # print(sp_dir[0])
    dirs=np.load(sp_dir[0])
    print(f"dirs:{dirs}")
    # dirs = os.listdir(sp_dir[0])

    for index,i in enumerate(dirs):
        tmp=[]
        count=0
        i=os.path.join(sp_dir[0],i)
        # data=np.load(i)
        if i.endswith('.npy'):
            data = np.load(i, allow_pickle=True)
        elif i.endswith('.tsv'):
            data = np.genfromtxt(i, delimiter='\t', dtype=None, encoding=None)
        elif i.endswith('.csv'):
            data = np.genfromtxt(i, delimiter=',', dtype=None, encoding=None, skip_header=1)
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

def test_model(net, sps_dir, aa_f, model_f, tcr_num, device, filter_seq, record_r, mask_ratio):
    # Get data.
    test_sps, test_labels, test_spnames, test_mask = read_samples(sps_dir, tcr_num, filter_seq, mask_ratio)
    # test_sps, test_labels, test_spnames, test_mask = read_samples_tyh(sps_dir, lables_dir="/mnt/sdc/tyh/TCRFormer/MIL/FHIV/3/files/test_labels.npy")
    # Make predictions.
    if net == "DeepLION":
        probs = DeepLION.prediction(test_sps, model_f, aa_f, tcr_num, device)
        # with open("./Results/results_DeepLION_FHIV2-3_FFF_0530.tsv", 'w') as f:
        #     for prob in probs:
        #         f.write(f"{prob}\n")
        #     print(f"Result has been recorded: ./Results/results_DeepLION_FHIV2-3_FFF_0530.tsv")
    elif net == "TransMIL":
        probs = TransMIL.prediction(test_sps, model_f, aa_f, tcr_num, device, test_mask)
        with open("./Results/results_TransMIL_FHIV2-3_0530.tsv", 'w') as f:
            for prob in probs:
                f.write(f"{prob}\n")
            print(f"Result has been recorded: ./Results/results_TransMIL_FHIV2-3_0530.tsv")
    elif net == "BiFormer":
        probs = BiFormer.prediction(test_sps, model_f, aa_f, tcr_num, device)
        with open("./Results/results_BiFormer_FHIV2-3_0530.tsv", 'w') as f:
            for prob in probs:
                f.write(f"{prob}\n")
            print(f"Result has been recorded: ./Results/results_BiFormer_FHIV2-3_0530.tsv")
    elif net == "DeepLION2":
        probs = DeepLION2.prediction(test_sps, model_f, aa_f, tcr_num, device)
        # with open("./Results/results_DeepLION2_FHIV2-3_0530.tsv", 'w') as f:
        #     for prob in probs:
        #         f.write(f"{prob}\n")
        #     print(f"Result has been recorded: ./Results/results_DeepLION2_FHIV2-3_0530.tsv")
    elif net == "MINN_SA":
        probs = MINN_SA.prediction(test_sps, model_f, aa_f, tcr_num, device)
        with open("./Results/results_MINN_SA_FHIV2-3_0530.tsv", 'w') as f:
            for prob in probs:
                f.write(f"{prob}\n")
            print(f"Result has been recorded: ./Results/results_MINN_SA_FHIV2-3_0530.tsv")
    else:
        print("Wrong parameter 'network' set!")
        return -1
    # Evaluation.
    acc, recall, specificity, mcc, roc_auc, aupr = utils.evaluation(probs, test_labels)

    print("Accuracy = ", acc)
    print("Sensitivity = ", recall)
    print("Specificity = ", specificity)
    print("MCC = ", mcc)
    print("AUC = ", roc_auc)
    print("AUPR = ", aupr)
    print()

    # Record prediction results.
    if record_r:
        record_prediction(test_spnames, probs, record_r)


def training_model(net, sps_dir, valid_sps_dir, tcr_num, lr, ep, dropout, log_inr, aa_f, model_f, device, loss, alpha,
                   beta, gce_q, pretraining, data_balance, filter_seq, mask_ratio):
    # Get data.
    # training_sps, training_labels, training_spnames, training_mask = read_samples_tyh(sps_dir, lables_dir="/mnt/sdc/tyh/TCRFormer/MIL/FHIV/3/files/train_labels.npy")
    # training_sps, training_labels, training_spnames, training_mask = read_samples_HIV(sps_dir, tcr_num, filter_seq,
    #                                                                               mask_ratio)
    training_sps, training_labels, training_spnames, training_mask = read_samples(sps_dir, tcr_num, filter_seq,
                                                                                  mask_ratio)
    # print(training_sps) # 这是一个列表，列表中的每个元素代表一个样本，每个样本信息也是一个列表，共有100个元素，每个元素内容大致如下['CASSVSTGVDEQYF', 'TRBV2*01', '0.001081', '0.5489162544882538']
    # print(type(training_sps)) # 确实是list类型
    # print(len(training_sps)) # 344，样本总数
    # print(len(training_labels)) # 100，过滤后每个样本的TCR总数为100条

    # print(training_sps[0][0])
    # print(len(training_sps[0][0]))
    if data_balance:
        training_seqs, training_labels = utils.data_balance(training_sps, training_labels) # 平衡正负样本数量
    # valid_sps, valid_labels, valid_spnames, valid_mask = read_samples(valid_sps_dir, tcr_num, filter_seq, mask_ratio)
    valid_sps, valid_labels, valid_spnames, valid_mask = read_samples(["/mnt/sdb/juhengwei/covid19/Covid19_big_processed/ImmuneCODE-Review-002/DeepLION_train/5"], tcr_num, filter_seq, mask_ratio)
    # valid_sps, valid_labels, valid_spnames, valid_mask = read_samples_tyh(["/mnt/sdc/tyh/TCRFormer/MIL/FHIV/3/files/val_file_name.npy"], lables_dir="/mnt/sdc/tyh/TCRFormer/MIL/FHIV/3/files/val_labels.npy")
    # Training model.
    if net == "DeepLION":
        if loss[0] == "W":
            # Pretraining.
            pre_ep = pretraining
            print(f"pre_ep:{pre_ep}")
            pre_model_f = "models/temp.pth"
            pre_loss = "CE"
            # DeepLION.training(training_sps, training_labels, tcr_num, lr, pre_ep, dropout, log_inr, pre_model_f, aa_f, device,
            #                   pre_loss, alpha, beta, gce_q, shuffle=False, pretrained_model_path="./Results/DeepLION_big_Covid.pth")
            DeepLION.training(training_sps, training_labels, tcr_num, lr, pre_ep, dropout, log_inr, pre_model_f, aa_f, device,
                              pre_loss, alpha, beta, gce_q, shuffle=False)
            probs = DeepLION.prediction(training_sps, pre_model_f, aa_f, device)
            os.remove(pre_model_f)
            # Calculate weights.
            weights = []
            for ind, prob in enumerate(probs):
                weights.append(1 - abs(training_labels[ind] - prob))
            # Training.
            # DeepLION.training(training_sps, training_labels, tcr_num, lr, ep, dropout, log_inr,
            #                   model_f, aa_f, device, loss[1:], alpha, beta, gce_q, loss_w=weights, shuffle=False, pretrained_model_path="./Results/DeepLION_big_Covid.pth")
            DeepLION.training(training_sps, training_labels, tcr_num, lr, pre_ep, dropout, log_inr, pre_model_f, aa_f, device,
                              pre_loss, alpha, beta, gce_q, shuffle=False)
        else:
            # DeepLION.training(training_sps, training_labels, tcr_num, lr, ep, dropout, log_inr,
            #                   model_f, aa_f, device, loss, alpha, beta, gce_q, pretrained_model_path="./Results/DeepLION_big_Covid.pth")
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
        # 传入的training_sps是列表
        # DeepLION2.training(training_sps, training_labels, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device,
        #                       valid_sps=valid_sps, valid_lbs=valid_labels, pretrained_model_path="./Results/DeepLION2_big_Covid.pth")
        DeepLION2.training(training_sps, training_labels, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device,
                              valid_sps=valid_sps, valid_lbs=valid_labels)
    elif net == "MINN_SA":
        MINN_SA.training(training_sps, training_labels, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device,
                         valid_sps=valid_sps, valid_lbs=valid_labels)
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
