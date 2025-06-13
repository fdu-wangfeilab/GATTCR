# -------------------------------------------------------------------------
# Name: preprocess.py
# Coding: utf8
# Author: Xinyang Qian
# Intro: Extracting information (TCRb CDR3 sequence, v_gene, frequency)
#        from raw TCR sequencing files, and predicting each TCR.
#        The processed file's format is that:
#        ----- TCR_seq.tsv_processed.tsv -----
#        amino_acid v_gene  frequency   target_seq   [caTCR_score]
#        CASRGRGWDTEAFF TRBV19*01 0.010528  CASRGRGWDTEAFF  [0.2238745]
#        ......
# -------------------------------------------------------------------------

import argparse
import os

import utils
from network import TCRD
from tqdm import tqdm


def create_parser():
    parser = argparse.ArgumentParser(
        description='Script to preprocess raw TCR sequencing files.'
    )
    parser.add_argument(
        "--sample_dir",
        dest="sample_dir",
        type=str,
        help="The directory of samples for preprocessing.",
        required=True
    )
    parser.add_argument(
        "--info_index",
        dest="info_index",
        type=str,
        help="The index of information (aaSeqCDR3, allVHitsWithScore, cloneFraction) in the raw files.",
        required=True
    )
    parser.add_argument(
        "--tcr_num",
        dest="tcr_num",
        type=int,
        help="The number of TCRs extracted from the raw files.",
        default=-1,
    )
    parser.add_argument(
        "--crop_num",
        dest="crop_num",
        type=int,
        help="The number of amino acids discarded at the beginning/end of the TCR sequences."
             "E.g., if crop_num = 2, 'CASSFIRLGDSGYTF' => 'SSFIRLGDSGY'.",
        default=0,
    )
    parser.add_argument(
        "--filters_num",
        dest="filters_num",
        type=int,
        help="The number of the filter set in DeepLION.",
        default=1,
    )
    parser.add_argument(
        "--aa_file",
        dest="aa_file",
        type=str,
        help="The file recording animo acid vectors.",
        required=True,
    )
    parser.add_argument(
        "--model_file",
        dest="model_file",
        type=str,
        help="The pre-trained TCRD model file of for TCR prediction in .pth format. "
             "The default value 'None' means no prediction for TCRs.",
        default=None
    )
    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        type=str,
        help="The directory to save preprocessed files.",
        required=True
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        help="The device used to make prediction.",
        default="cpu",
    )
    parser.add_argument(
        "--ratio",
        dest="ratio",
        type=str,
        help="The ratio to split data.",
        default="[1]"
    )
    args = parser.parse_args()
    return args


def preprocess_files(sp_dir, info_i, tcr_n, crop_n, filters_n, aa_f, model_f, device, sv_dir, ratio):
    sv_dir = sv_dir + utils.get_last_path(sp_dir)
    utils.create_directory(sv_dir)
    
    # 获取文件列表
    files = os.listdir(sp_dir)
    
    # 初始化变量以统计TCR数量
    total_tcrs = 0
    file_count = 0
    
    # 使用 tqdm 包裹文件列表，显示整体处理进度
    for e in tqdm(files, desc="Processing files", unit="file"):
        e_path = utils.correct_path(sp_dir + e)
        if os.path.isdir(e_path):
            # 递归处理文件夹
            file_tcrs, count = preprocess_files(e_path, info_i, tcr_n, crop_n, filters_n, aa_f, model_f, device, sv_dir, ratio)
            total_tcrs += file_tcrs
            file_count += count
        else:
            # 处理单个文件
            file_tcrs = preprocess_file(e_path, info_i, tcr_n, crop_n, filters_n, aa_f, model_f, device, sv_dir)
            total_tcrs += file_tcrs
            file_count += 1
    
    utils.split_data(sv_dir, ratio)
    
    # # 打印 TCR 总数和平均数
    # print(f"Total TCRs processed (before filtering): {total_tcrs}")
    # print(f"Average TCRs per file (before filtering): {total_tcrs / file_count if file_count > 0 else 0:.2f}")
    
    return total_tcrs, file_count


def preprocess_file(fname, info_i, tcr_n, crop_n, filters_n, aa_f, model_f, device, sv_dir):
    # 提取信息（过滤前的所有TCR）
    extract_info = utils.read_tsv(fname, info_i, True)
    
    # 统计过滤前的 TCR 数量
    tcr_count_before_filtering = len(extract_info)
    
    # 过滤无效的 TCR 序列
    filtered_info = []
    for tcr in extract_info:
        if not utils.check_tcr(tcr[0]):
            continue
        if not tcr[1]:
            continue
        else:
            tcr[1] = process_vgene(tcr[1])
        if not tcr[2]:
            continue
        if crop_n != 0:
            filtered_info.append([tcr[0], tcr[1], tcr[2], tcr[0][crop_n: -crop_n]])
        else:
            filtered_info.append([tcr[0], tcr[1], tcr[2], tcr[0]])
    
    # 按频率排序 TCRs
    filtered_info = sorted(filtered_info, key=lambda x: float(x[2]), reverse=True)
    if 0 < tcr_n < len(filtered_info):
        filtered_info = filtered_info[: tcr_n]
    
    # TCR 预测
    tcr_scores = []
    if model_f:
        input_tcrs = [tcr[3] for tcr in filtered_info]
        tcr_scores = TCRD.prediction(input_tcrs, filters_n, model_f, aa_f, device)
    
    # 保存结果
    base_fname = utils.get_last_path(fname).split('.')[0]
    # sv_fname = sv_dir + 'positive_' + base_fname
    sv_fname = sv_dir + 'negative_' + base_fname
    if model_f:
        sv_fname = sv_dir + 'negative_' + base_fname + "_processed_TCRD.tsv"
    else:
        sv_fname += "_processed.tsv"
    
    with open(sv_fname, "w", encoding="utf8") as wf:
        if not model_f:
            wf.write("amino_acid\tv_gene\tfrequency\ttarget_seq\n")
        else:
            wf.write("amino_acid\tv_gene\tfrequency\ttarget_seq\tcaTCR_score\n")
        
        for ind, tcr in enumerate(filtered_info):
            wf.write("\t".join(tcr))
            if tcr_scores:
                wf.write("\t{0}".format(tcr_scores[ind]))
            wf.write("\n")
    
    # 返回过滤前的 TCR 数量
    return tcr_count_before_filtering


def process_vgene(vgene):
    # 处理 V 基因
    vgene_list = vgene.split(",")
    if len(vgene_list) > 1:
        final_gene, max_score = "", 0
        for vg in vgene_list:
            vg = vg.strip()[:-1]
            v, s = vg.split("(")
            if float(s) > max_score:
                max_score = float(s)
                final_gene = v.strip()
    else:
        final_gene = vgene_list[0]
    return final_gene


def main():
    # 解析参数
    args = create_parser()
    args.sample_dir = utils.correct_path(args.sample_dir)
    args.aa_file = utils.correct_path(args.aa_file)
    if args.model_file:
        args.model_file = utils.correct_path(args.model_file)
    args.save_dir = utils.correct_path(args.save_dir)
    while type(args.info_index) == str:
        args.info_index = eval(args.info_index)
    if type(args.info_index) != list:
        print("Wrong parameter 'info_index' set!")
        return -1
    while type(args.ratio) == str:
        args.ratio = eval(args.ratio)
    if type(args.ratio) != list:
        print("Wrong parameter 'ratio' set!")
        return -1

    # 预处理原始文件
    utils.create_directory(args.save_dir)
    total_tcrs, file_count = preprocess_files(args.sample_dir, args.info_index, args.tcr_num, args.crop_num, args.filters_num, args.aa_file,
                     args.model_file, args.device, args.save_dir, args.ratio)
    
    # 打印总的 TCR 数量和平均每个文件的 TCR 数量
    print(f"Total TCRs processed (before filtering): {total_tcrs}")
    print(f"Average TCRs per file (before filtering): {total_tcrs / file_count if file_count > 0 else 0:.2f}")


if __name__ == "__main__":
    main()