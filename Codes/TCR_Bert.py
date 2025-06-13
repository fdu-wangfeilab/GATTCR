import os, sys
import importlib
import glob
from typing import *
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from transformers import BertModel, BertForMaskedLM, BertTokenizer, FeatureExtractionPipeline
from matplotlib import pyplot as plt
from scipy import stats
import anndata as ad
import tqdm
from itertools import zip_longest
import scanpy as sc

# 添加TCRBert项目路径，便于将TCRBert的包导入
SRC_DIR = os.path.join("/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/tcr/")
assert os.path.isdir(SRC_DIR), f"Cannot find src dir: {SRC_DIR}"
sys.path.append(SRC_DIR)

import data_loader as dl
import featurization as ft
import canonical_models as models
# import utils
import importlib.util
spec = importlib.util.spec_from_file_location("utils", "/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/tcr/utils.py")
utils = importlib.util.module_from_spec(spec)
sys.modules["utils"] = utils
spec.loader.exec_module(utils)

FILT_EDIT_DIST = False
N_COMPONENTS = 0.9

if FILT_EDIT_DIST:
    PLOT_DIR = os.path.join(os.path.dirname(SRC_DIR), "plots/pird_antigen_cv_edit_dist_filt")
else:
    PLOT_DIR = os.path.join(os.path.dirname(SRC_DIR), "plots/pird_antigen_cv")
if not os.path.isdir(PLOT_DIR):
    os.makedirs(PLOT_DIR)

class TCR_Bert:
    def __init__(self, model_path: str, src_dir: str, device: int = 1):
        """
        初始化TCRBert类。

        参数:
        - model_path: 预训练模型的路径。
        - src_dir: TCRBert项目的源代码目录。
        - device: 要使用的设备编号。
        """
        assert os.path.isdir(src_dir), f"Cannot find src dir: {src_dir}"
        sys.path.append(src_dir)
        self.model_path = model_path
        self.src_dir = src_dir
        self.device = device
        self.model = BertModel.from_pretrained(model_path)
        
    def get_transformer_embeddings(
        self,
        seqs: Iterable[str],
        seq_pair: Optional[Iterable[str]] = None,
        *,
        layers: List[int] = [-1],
        method: str = "mean",
        batch_size: int = 256
    ) -> np.ndarray:
        """
        获取给定序列从指定层的嵌入表示。
        """
        # 方法体保持不变，仅更改为使用类属性如self.model_path和self.device
        device = utils.get_device(self.device)  # 使用类属性self.device来确定模型运行的设备
        seqs = [s if ft.is_whitespaced(s) else ft.insert_whitespace(s) for s in seqs]
        try:
            tok = ft.get_pretrained_bert_tokenizer(self.model_path)  # 使用类属性self.model_path
        except OSError:
            logging.warning("Could not load saved tokenizer, loading fresh instance")
            tok = ft.get_aa_bert_tokenizer(64)

        model = BertModel.from_pretrained(self.model_path, add_pooling_layer=method == "pool").to(device)  # 使用类属性self.model_path和device

        chunks = dl.chunkify(seqs, batch_size)
        chunks_pair = [None]
        if seq_pair is not None:
            assert len(seq_pair) == len(seqs)
            chunks_pair = dl.chunkify(
                [s if ft.is_whitespaced(s) else ft.insert_whitespace(s) for s in seq_pair],
                batch_size,
            )
        chunks_zipped = list(zip_longest(chunks, chunks_pair))

        embeddings = []
        with torch.no_grad():
            for seq_chunk in chunks_zipped:
                encoded = tok(
                    *seq_chunk, padding="max_length", max_length=64, return_tensors="pt"
                )
                input_mask = encoded["attention_mask"].numpy()
                encoded = {k: v.to(device) for k, v in encoded.items()}

                x = model.forward(
                    **encoded, output_hidden_states=True, output_attentions=True
                )
                if method == "pool":
                    embeddings.append(x.pooler_output.cpu().numpy().astype(np.float64))
                    continue

                for i in range(len(seq_chunk[0])):
                    e = []
                    for l in layers:
                        h = (
                            x.hidden_states[l][i].cpu().numpy().astype(np.float64)
                        )
                        if method == "cls":
                            e.append(h[0])
                            continue
                        if seq_chunk[1] is None:
                            seq_len = len(seq_chunk[0][i].split())
                        else:
                            seq_len = (
                                len(seq_chunk[0][i].split())
                                + len(seq_chunk[1][i].split())
                                + 1
                            )
                        seq_hidden = h[1 : 1 + seq_len]
                        assert len(seq_hidden.shape) == 2

                        if method == "mean":
                            e.append(seq_hidden.mean(axis=0))
                        elif method == "max":
                            e.append(seq_hidden.max(axis=0))
                        elif method == "attn_mean":
                            attn = x.attentions[l][i, :, :, : seq_len + 2]
                            print(attn.sum(axis=-1))
                            raise NotImplementedError
                        else:
                            raise ValueError(f"Unrecognized method: {method}")
                    e = np.hstack(e)
                    assert len(e.shape) == 1
                    embeddings.append(e)
        if len(embeddings[0].shape) == 1:
            embeddings = np.stack(embeddings)
        else:
            embeddings = np.vstack(embeddings)

        del x
        del model
        torch.cuda.empty_cache()
        return embeddings

        
    def process_directory(self, input_dir: str, output_dir: str, batch_size: int = 256, layers: List[int] = [6], method: str = "mean"):
        """
        处理指定目录下的所有文件，并保存处理后的结果到输出目录。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        rfs = os.listdir(input_dir)
        for i in range(len(rfs)):
            d = pd.read_csv(os.path.join(input_dir, rfs[i]), sep='\t')
            aa_tokens = d['amino_acid'].astype('str').apply(lambda x: list(x))
            embeddings = self.get_transformer_embeddings(
                seqs=aa_tokens,
                layers=layers,
                method=method,
                batch_size=batch_size
            )
            np.savez(os.path.join(output_dir, os.path.splitext(rfs[i])[0] + ".npz"), seq=aa_tokens, emb=embeddings)
    
    def process_sps(self, sps: List[List[List[str]]], batch_size: int = 100, layers: List[int] = [6], method: str = "mean"):
        """
        处理sps中的所有样本及其TCR序列，返回嵌入表示的三维数组。

        参数:
        - sps: 一个包含所有样本信息的列表，每个样本包含多条TCR序列信息，每条序列信息是包含氨基酸序列的列表。
        - layers: 指定从哪些层获取嵌入表示，默认为最后一层。
        - method: 提取嵌入的方法，包括mean、max、attn_mean、cls和pool。
        - batch_size: 处理序列时的批处理大小。

        返回:
        - 一个形状为[样本总数, TCR数量, 特征数]的三维数组。
        """
        sample_embeddings = []  # 用于存储所有样本的嵌入表示    
        count = 0
        total_samples = len(sps)
        from tqdm import tqdm
        with tqdm(total=total_samples, desc="处理样本", unit="个") as pbar:
            for sample in sps:
                seqs = [seq_info[0] for seq_info in sample]  # 提取每个样本中所有序列的氨基酸信息
                # print(f"seqs:{seqs}\n")
                # print(f"embedding: {seqs}")
                embeddings = self.get_transformer_embeddings(seqs=seqs, layers=layers, method=method, batch_size=batch_size)
                # print(embeddings)
                # print(f"len of embeddings: {len(embeddings)}")
                # print(f"len of embeddings[0]: {len(embeddings[0])}")
                # print("###########################################################")
                sample_embeddings.append(embeddings)
                # print(sample_embeddings)
                # count = count + 1
                # print(count)
                pbar.update(1)

        # # print(sample_embeddings)
        # print(f"len of sample_embeddings: {len(sample_embeddings)}")
        # print(f"len of sample_embeddings[0]:{len(sample_embeddings[0])}")
        # print(f"len of sample_embeddings[0][0]: {len(sample_embeddings[0][0])}") 
        
        # 将所有样本的嵌入表示堆叠成一个三维数组
        sample_embeddings = np.array(sample_embeddings)
        return sample_embeddings
    
    def process_sp(self, sp, batch_size: int = 100, layers: List[int] = [6], method: str = "mean"):
        """
            处理单个样本sp，返回其嵌入表示。
            返回: 一个形状为[1, TCR数量, 特征数]的数组
        """
        seqs = [seq_info[0] for seq_info in sp]  # 提取样本中所有序列的氨基酸信息
        embeddings = self.get_transformer_embeddings(seqs=seqs, layers=layers, method=method, batch_size=batch_size)
        embeddings = np.array(embeddings).reshape(1, *embeddings.shape) # 调整结果的形状
        
        return embeddings
