# 处理新收集的两个癌症数据集
import argparse
import os

import utils
from network import TCRD


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
        help="The index of information (e.g., aaSeqCDR3, allVHitsWithScore, cloneFraction) in the raw files.",
        required=False
    )
    parser.add_argument(
        "--tcr_num",
        dest="tcr_num",
        type=int,
        help="The number of TCRs extracted from the raw files.",
        default=5000,
    )
    parser.add_argument(
        "--crop_num",
        dest="crop_num",
        type=int,
        help="The number of amino acids discarded at the beginning/end of the TCR sequences."
             " E.g., if crop_num = 2, 'CASSFIRLGDSGYTF' => 'SSFIRLGDSGY'.",
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
        help="The file recording amino acid vectors.",
        required=False,
    )
    parser.add_argument(
        "--model_file",
        dest="model_file",
        type=str,
        help="The pre-trained TCRD model file for TCR prediction in .pth format. "
             "The default value 'None' means no prediction for TCRs.",
        default=None
    )
    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        type=str,
        help="Directory to save processed files.",
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
        default="[0.7, 0.3]"
    )
    args = parser.parse_args()
    return args


def preprocess_files(sp_dir, info_index, tcr_n, crop_n, filters_n, aa_f, model_f, device, sv_dir, ratio):
    sv_dir = sv_dir + utils.get_last_path(sp_dir)
    utils.create_directory(sv_dir)
    for e in os.listdir(sp_dir):
        e_path = utils.correct_path(sp_dir + e)
        if os.path.isdir(e_path):
            preprocess_files(e_path, info_index, tcr_n, crop_n, filters_n, aa_f, model_f, device, sv_dir, ratio)
        else:
            preprocess_file(e_path, sv_dir, info_index, crop_n, tcr_n, model_f)
    utils.split_data(sv_dir, ratio)
    return 0


def preprocess_file(file_path, save_dir, info_index, crop_num, tcr_num, model_file=None):
    # 读取文件
    print(file_path)
    df = utils.read_tsv(file_path, info_index)
    
    # 提取新数据中的相关信息列
    amino_acid_column = 'aminoAcid'  # 新数据中氨基酸序列列
    v_gene_column = 'vGeneName'  # 新数据中V基因列
    frequency_column = 'frequencyCount (%)'  # 新数据中频率列

    # 如果info_index存在，并且是有效值
    if info_index:
        # 根据info_index来选择处理的列（你可以根据需要自定义处理逻辑）
        if info_index == "aaSeqCDR3":
            amino_acid_column = 'aaSeqCDR3'
        elif info_index == "allVHitsWithScore":
            v_gene_column = 'allVHitsWithScore'
        elif info_index == "cloneFraction":
            frequency_column = 'cloneFraction'
        else:
            print(f"Invalid info_index: {info_index}. Using default columns.")
    
    # 提取amino_acid, v_gene, frequency列
    df['amino_acid'] = df[amino_acid_column]
    df['v_gene'] = df[v_gene_column]
    df['frequency'] = df[frequency_column]

    # 如果有crop_num参数，则进行裁剪操作
    if crop_num > 0:
        df['amino_acid'] = df['amino_acid'].apply(lambda x: x[crop_num:] if len(x) > crop_num else x)

    # 筛选频率列，并按频率排序
    df = df[df['frequency'] > 0]  # 过滤掉频率为0的数据
    df = df.sort_values(by='frequency', ascending=False)

    # 根据tcr_num限制数量
    if tcr_num:
        df = df.head(tcr_num)

    # 仅保存需要的列
    result_df = df[['amino_acid', 'v_gene', 'frequency']]

    # 如果提供了model_file，进行TCR预测
    if model_file:
        # TCR预测部分
        prediction_df = TCRD.prediction(result_df, model_file)
        result_df['caTCR_score'] = prediction_df['caTCR_score']
    
    # 保存处理后的文件
    result_file = os.path.join(save_dir, f"processed_{os.path.basename(file_path)}")
    result_df.to_csv(result_file, index=False)

    print(f"Processed file saved to {result_file}")


def main():
    # Parse arguments.
    args = create_parser()
    args.sample_dir = utils.correct_path(args.sample_dir)
    # args.aa_file = utils.correct_path(args.aa_file)
    if args.model_file:
        args.model_file = utils.correct_path(args.model_file)
    args.save_dir = utils.correct_path(args.save_dir)

    # Parse and validate ratio parameter
    while type(args.ratio) == str:
        args.ratio = eval(args.ratio)
    if type(args.ratio) != list:
        print("Wrong parameter 'ratio' set!")
        return -1

    # Preprocess raw files.
    utils.create_directory(args.save_dir)
    preprocess_files(args.sample_dir, args.info_index, args.tcr_num, args.crop_num, args.filters_num, args.aa_file,
                     args.model_file, args.device, args.save_dir, args.ratio)


if __name__ == "__main__":
    main()
