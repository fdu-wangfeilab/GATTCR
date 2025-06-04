# -------------------------------------------------------------------------
# Name: network.py
# Coding: utf8
# Author: Xinyang Qian
# Intro: Containing deep learning network classes.
# -------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as Data
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import math
import os
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight # 为每个类别分配不同的权重
from sparsemax import Sparsemax
import numpy as np
import pandas as pd

import utils
from loss import SCELoss, GCELoss, CELoss
from TCR_Bert import TCR_Bert

from torch_geometric.nn import MessagePassing, SAGEConv, SAGPooling, global_max_pool as gmp, global_mean_pool as gap
from torch_geometric.utils import add_self_loops, degree, dense_to_sparse, softmax

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import umap
import matplotlib.pyplot as plt

class TCRD(nn.Module):
    # TCR detector (TCRD) can predict a TCR sequence's class (e.g. cancer-associated TCR) (binary classification).
    # It can extract the antigen-specific biochemical features of TCRs based on the convolutional neural network.
    def __init__(self, aa_num=24, feature_num=15, kernel_size=None, filter_num=None, filters_num=1, drop_out=0.4):
        super(TCRD, self).__init__()
        self.aa_num = aa_num  # The number of amino acids that one TCR contains.
        self.feature_num = feature_num  # The dimension of the feature vector of one amino acid.
        if kernel_size is None:
            kernel_size = [2, 3, 4, 5, 6, 7]
        self.kernel_size = kernel_size  # The specification of the convolution kernel in the convolution layer.
        if filter_num is None:
            filter_num = [3, 3, 3, 2, 2, 1]
        assert len(filter_num) == len(kernel_size), \
            "The parameters 'kernel_size' and 'filter_num' set do not match!"
        self.filter_num = []  # The number of the corresponding convolution kernels.
        for ftr in filter_num:
            self.filter_num.append(ftr * filters_num)
        self.drop_out = drop_out
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(self.kernel_size)
        ])
        self.fc = nn.Linear(sum(self.filter_num), 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, sum(self.filter_num))
        out = self.dropout(self.fc(out))
        return out

    @staticmethod
    def training(tcrs, lbs, filters_n, lr, ep, dropout, log_inr, model_f,
                 aa_f, device, loss, alpha, beta, gce_q, loss_w=None, shuffle=True):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # The format of lbs is [lb_0, lb_1, ..., lb_n].
        # Generate input.
        training_sps = []
        for tcr in tcrs:
            training_sps.append([[tcr]])
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        input_batch, label_batch = utils.generate_input_for_training(training_sps, lbs, aa_v, ins_num=1)
        input_batch, label_batch = torch.Tensor(input_batch), torch.LongTensor(label_batch)
        dataset = Data.TensorDataset(input_batch, label_batch)
        loader = Data.DataLoader(dataset, len(input_batch), shuffle)
        # Set model.
        model = TCRD(filters_num=filters_n, drop_out=dropout).to(torch.device(device))
        criterion = nn.CrossEntropyLoss().to(device)  # The default loss function is CE.
        weights = []
        if loss_w is not None:
            for w in loss_w:
                weights.append([w] * 2)
            weights = torch.FloatTensor(weights)
        else:
            weights = None
        if loss == "SCE":
            criterion = SCELoss(alpha=alpha, beta=beta, num_classes=2).to(device)
        elif loss == "GCE":
            criterion = GCELoss(q=gce_q, num_classes=2).to(device)
        elif loss == "CE":
            criterion = CELoss(w=weights, num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Training model.
        for epoch in range(ep):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                if (epoch + 1) % log_inr == 0:
                    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Save model.
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)
        return 0

    @staticmethod
    def prediction(tcrs, filters_n, model_f, aa_f, device):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        model = TCRD(filters_num=filters_n).to(device)
        model.load_state_dict(torch.load(model_f))
        model = model.eval()
        # Predict each TCR.
        tcr_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for tcr in tcrs:
            # Generate input.
            input_x = utils.generate_input_for_prediction([[tcr]], aa_v, ins_num=1)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict = model(input_x)
            prob = float(1 / (1 + math.exp(-predict[0][1] + predict[0][0])))
            tcr_scores.append(prob)
        return tcr_scores


class MINN_SA(nn.Module):
    # MINN_SA can predict a TCR repertoire's class (e.g. cancer-associated TCR repertoire) (binary classification).
    def __init__(self, tcr_num=100, aa_num=24, feature_num=15, kernel_size=None, filter_num=None, attention_head_num=1,
                 attention_hidden_size=10, drop_out=0.4, topk=0.05):
        super(MINN_SA, self).__init__()
        self.tcr_num = tcr_num  # The number of TCR sequences that one input individual sample contains.
        self.aa_num = aa_num  # The number of amino acids that one TCR contains.
        self.feature_num = feature_num  # The dimension of the feature vector of one amino acid.
        if kernel_size is None:
            kernel_size = [2, 3, 4, 5, 6, 7]
        self.kernel_size = kernel_size  # The specification of the convolution kernel in the convolution layer.
        if filter_num is None:
            filter_num = [3, 3, 3, 2, 2, 1]
        assert len(filter_num) == len(kernel_size), \
            "The parameters 'kernel_size' and 'filter_num' set do not match!"
        self.filter_num = filter_num  # The number of the corresponding convolution kernels.
        self.all_filter_num = sum(self.filter_num)
        self.attention_head_num = attention_head_num
        self.attention_hidden_size = attention_hidden_size
        self.attention_head_size = int(self.attention_hidden_size / self.attention_head_num)
        self.drop_out = drop_out
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(self.kernel_size)
        ])
        # Attention
        ninp = self.all_filter_num
        self.waint = nn.Linear(ninp, ninp, bias=False)
        self.waout = nn.Linear(ninp, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.sparsemax = Sparsemax(dim=1)
        self.fclayer1 = nn.Linear(ninp, ninp, bias=True)
        self.fclayer2 = nn.Linear(ninp, ninp, bias=True)
        self.bn1 = nn.BatchNorm1d(ninp)
        self.decoder_f = nn.Linear(ninp, ninp)
        self.decoder_s = nn.Linear(ninp, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # Attention
        out = out.transpose(0, 1)
        output = torch.relu(self.dropout(self.fclayer1(out)))
        attw = torch.squeeze(self.waout(output), dim=-1)
        attw = torch.transpose(attw, 0, 1)
        attw = self.sparsemax(attw)
        output = torch.transpose(output, 0, 1)
        output = torch.bmm(torch.unsqueeze(attw, 1), output)
        output = torch.squeeze(output, dim=1)
        output = self.decoder_f(output)
        output = torch.relu(self.bn1(output))
        output = self.decoder_s(output)
        return output, attw

    @staticmethod
    def training(sps, lbs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, shuffle=False, valid_sps=None,
                 valid_lbs=None):
        # The format of sps is [[tcr_0, tcr_1, ..., tcr_n], ...].
        # The format of lbs is [lb_0, lb_1, ..., lb_n].
        # Generate input.
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        input_batch, label_batch = utils.generate_input_for_training(sps, lbs, aa_v, ins_num=tcr_num)
        input_batch, label_batch = torch.Tensor(input_batch), torch.LongTensor(label_batch)
        dataset = Data.TensorDataset(input_batch, label_batch)
        loader = Data.DataLoader(dataset, len(input_batch), shuffle)
        # Set model.
        model = MINN_SA(drop_out=dropout).to(torch.device(device))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Training model.
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        max_acc, valid_model_f = 0, model_f + "_temp.pth"
        criterion = nn.CrossEntropyLoss().to(device)
        for epoch in range(ep):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred, attn = model(batch_x)
                loss = criterion(pred, batch_y)
                if (epoch + 1) % log_inr == 0:
                    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
                    # Valid the model.
                    if valid_sps and valid_lbs:
                        torch.save(model.state_dict(), valid_model_f)
                        valid_probs = MINN_SA.prediction(valid_sps, valid_model_f, aa_f, tcr_num, device)
                        # utils.evaluation(valid_probs, valid_lbs)  # Debug #
                        valid_preds = [1 if pred > 0.5 else 0 for pred in valid_probs]
                        valid_acc = accuracy_score(valid_lbs, valid_preds)
                        if valid_acc > max_acc:
                            max_acc = valid_acc
                            if os.path.exists(model_f):
                                os.remove(model_f)
                            os.rename(valid_model_f, model_f)
                        else:
                            os.remove(valid_model_f)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Save model.
        if not (valid_sps and valid_lbs):
            torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)
        return 0

    @staticmethod
    def prediction(sps, model_f, aa_f, tcr_num, device):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        model = MINN_SA().to(device)
        model.load_state_dict(torch.load(model_f))
        model = model.eval()
        # Predict each TCR.
        repertoire_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict, attn = model(input_x)
            prob = float(1 / (1 + math.exp(-predict[0][1] + predict[0][0])))
            repertoire_scores.append(prob)
        return repertoire_scores


class DeepLION(nn.Module):
    # DeepLION can predict a TCR repertoire's class (e.g. cancer-associated TCR repertoire) (binary classification).
    # It is a deep multi-instance learning model, containing TCRD for caTCRs identification.
    def __init__(self, tcr_num=100, aa_num=24, feature_num=15, kernel_size=None, filter_num=None, drop_out=0.4):
        super(DeepLION, self).__init__()
        self.tcr_num = tcr_num  # The number of TCR sequences that one input individual sample contains.
        self.aa_num = aa_num  # The number of amino acids that one TCR contains.
        self.feature_num = feature_num  # The dimension of the feature vector of one amino acid.
        if kernel_size is None:
            kernel_size = [2, 3, 4, 5, 6, 7]
        self.kernel_size = kernel_size  # The specification of the convolution kernel in the convolution layer.
        if filter_num is None:
            filter_num = [3, 3, 3, 2, 2, 1]
        assert len(filter_num) == len(kernel_size), \
            "The parameters 'kernel_size' and 'filter_num' set do not match!"
        self.filter_num = filter_num  # The number of the corresponding convolution kernels.
        self.drop_out = drop_out
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(self.kernel_size)
        ])
        self.cnn_fc = nn.Linear(sum(self.filter_num), 1)
        self.mil_fc = nn.Linear(self.tcr_num, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, 1, sum(self.filter_num))
        out = self.dropout(self.cnn_fc(out))
        out = out.reshape(-1, self.tcr_num)
        out = self.dropout(self.mil_fc(out))
        return out

    def forward_tcr(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, 1, sum(self.filter_num))
        out = self.dropout(self.cnn_fc(out))
        out = out.reshape(-1, self.tcr_num)
        return out

    def forward_motifs(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        motif_index = [torch.argmax(conv[0](x), dim=-1) for conv in self.convs]
        motif_index = torch.cat(motif_index, dim=1)
        motif_length = []
        for ind, size in enumerate(self.kernel_size):
            for f in range(self.filter_num[ind]):
                motif_length.append(size)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, 1, sum(self.filter_num))
        out = self.dropout(self.cnn_fc(out))
        motif_score = out * self.cnn_fc.weight[0] + self.cnn_fc.bias[0] / out.shape[-1]  # 只能计算每一序列层面的，还需要加上对于序列的权重
        out = out.reshape(-1, self.tcr_num)
        out_0 = out * self.mil_fc.weight[0] + self.mil_fc.bias[0] / out.shape[-1]
        out_1 = out * self.mil_fc.weight[1] + self.mil_fc.bias[1] / out.shape[-1]
        out = torch.ones(out.shape) / (torch.ones(out.shape) +
                                           torch.exp(-out_1 + out_0))
        motif_score = motif_score.reshape(-1, self.tcr_num, sum(self.filter_num))
        motif_score_0 = motif_score * self.mil_fc.weight[0].unsqueeze(-1) + \
                        self.mil_fc.bias[0].unsqueeze(-1) / motif_score.shape[-2]
        motif_score_1 = motif_score * self.mil_fc.weight[1].unsqueeze(-1) + \
                        self.mil_fc.bias[1].unsqueeze(-1) / motif_score.shape[-2]
        motif_score = torch.ones(motif_score.shape) / (torch.ones(motif_score.shape) +
                                       torch.exp(-motif_score_1 + motif_score_0))
        return out, motif_index, motif_length, motif_score

    @staticmethod
    def training(sps, lbs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, loss, alpha, beta, gce_q,
                 loss_w=None, shuffle=True, pretrained_model_path=None):
        # The format of sps is [[tcr_0, tcr_1, ..., tcr_n],...].
        # The format of lbs is [lb_0, lb_1, ..., lb_n].
        # Generate input.
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        input_batch, label_batch = utils.generate_input_for_training(sps, lbs, aa_v, ins_num=tcr_num)
        input_batch, label_batch = torch.Tensor(input_batch), torch.LongTensor(label_batch)
        dataset = Data.TensorDataset(input_batch, label_batch)
        loader = Data.DataLoader(dataset, len(input_batch), shuffle)
        # Set model.
        model = DeepLION(drop_out=dropout).to(torch.device(device))
        
        if pretrained_model_path is not None:
            pretrained_dict = torch.load(pretrained_model_path, map_location=device)
            model.load_state_dict(pretrained_dict, strict=False)
            print(f"Loaded model from {pretrained_model_path}")


        criterion = nn.CrossEntropyLoss().to(device)  # The default loss function is CE.
        weights = []
        if loss_w is not None:
            for w in loss_w:
                weights.append([w] * 2)
            weights = torch.FloatTensor(weights)
        else:
            weights = None
        if loss == "SCE":
            criterion = SCELoss(alpha=alpha, beta=beta, num_classes=2).to(device)
        elif loss == "GCE":
            criterion = GCELoss(q=gce_q, num_classes=2).to(device)
        elif loss == "CE":
            criterion = CELoss(w=weights, num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Training model.
        for epoch in range(ep):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                if (epoch + 1) % log_inr == 0:
                    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Save model.
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)
        return 0

    @staticmethod
    def prediction(sps, model_f, aa_f, tcr_num, device):
        # The format of sps is [[tcr_0, tcr_1, ..., tcr_n], ...].
        # The format of lbs is [lb_0, lb_1, ..., lb_n].
        # Load model.
        model = DeepLION().to(device)
        model.load_state_dict(torch.load(model_f))
        model = model.eval()
        # Predict each TCR repertoire.
        repertoire_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for sp in sps:
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict = model(input_x)
            prob = float(1 / (1 + math.exp(-predict[0][1] + predict[0][0])))
            repertoire_scores.append(prob)
        return repertoire_scores

    @staticmethod
    def predict_tcrs(sps, model_f, aa_f, tcr_num, device):
        # The format of sps is [[tcr_0, tcr_1, ..., tcr_n], ...].
        # The format of lbs is [lb_0, lb_1, ..., lb_n].
        # Load model.
        model = DeepLION().to(device)
        model_paras = torch.load(model_f)
        model.load_state_dict(model_paras)
        w0, w1 = model_paras["mil_fc.weight"][0], model_paras["mil_fc.weight"][1]
        model = model.eval()
        # Predict each TCR repertoire.
        tcr_scores, repertoire_scores = [], []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for sp in sps:
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict = model.forward_tcr(input_x).tolist()
            scores, probs = [], []
            for tcr in range(tcr_num):
                scores.append(predict[0][tcr])
                probs.append(
                    float(math.exp(-predict[0][tcr] * w1[tcr] + predict[0][tcr] * w0[tcr])))
            tcr_scores.append(scores)
            repertoire_scores.append(probs)
        return tcr_scores, repertoire_scores

    @staticmethod
    def motif_identification(sps, sps_name, model_f, aa_f, tcr_num, device):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        model = DeepLION().to(device)
        model.load_state_dict(torch.load(model_f))
        model = model.eval()
        # Predict each TCR.
        sequence_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict, motif_index, motif_length, motif_score = model.forward_motifs(input_x)
            for i, s in enumerate(sp):
                result = [sps_name[ind]]
                seq = s[0]
                result.append(seq)
                result.append(predict[0][i])
                motifs = []
                for j, m in enumerate(motif_length):
                    motifs.append([seq[motif_index[i][j]: motif_index[i][j] + m], float(motif_score[0][i][j])])
                motifs = sorted(motifs, key=lambda x: x[1], reverse=True)
                result.append(motifs)
                sequence_scores.append(result)
        return sequence_scores


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = self.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)
        return self.layer_norm(output + residual), attn
        # return context, attn

    def scaled_dot_product_attention(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class TransMIL(nn.Module):
    # TransMIL can predict a TCR repertoire's class (e.g. cancer-associated TCR repertoire) (binary classification).
    # It contains TCRD and the self-attention mechanism for caTCRs identification.
    def __init__(self, tcr_num=100, aa_num=24, feature_num=15, kernel_size=None, filter_num=None, attention_head_num=1,
                 attention_hidden_size=10, n_layers=1, drop_out=0.4):
        super(TransMIL, self).__init__()
        self.tcr_num = tcr_num  # The number of TCR sequences that one input individual sample contains.
        self.aa_num = aa_num  # The number of amino acids that one TCR contains.
        self.feature_num = feature_num  # The dimension of the feature vector of one amino acid.
        if kernel_size is None:
            kernel_size = [2, 3, 4, 5, 6, 7]
        self.kernel_size = kernel_size  # The specification of the convolution kernel in the convolution layer.
        if filter_num is None:
            filter_num = [3, 3, 3, 2, 2, 1]
        assert len(filter_num) == len(kernel_size), \
            "The parameters 'kernel_size' and 'filter_num' set do not match!"
        self.filter_num = filter_num  # The number of the corresponding convolution kernels.
        self.all_filter_num = sum(self.filter_num)
        self.attention_head_num = attention_head_num
        self.attention_hidden_size = attention_hidden_size
        self.attention_head_size = int(self.attention_hidden_size / self.attention_head_num)
        self.drop_out = drop_out
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(self.kernel_size)
        ])
        # Attention
        self.attention_layers = nn.ModuleList([MultiHeadAttention(self.all_filter_num, self.attention_head_size,
                                                                  self.attention_head_size, self.attention_head_num)
                                               for _ in range(n_layers)])
        # Method I: Average #
        self.mil_fc = nn.Linear(self.all_filter_num, 2)
        # Method II: MLP #
        # self.fc = nn.Linear(self.attention_hidden_size, 1)
        # self.mil_fc = nn.Linear(self.tcr_num, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x, attn_mask=None):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # Attention
        enc_self_attns = []
        if not attn_mask:
            attn_mask = torch.full([out.shape[0], self.tcr_num, self.tcr_num], False, dtype=torch.bool)
        else:
            attn_mask = torch.Tensor(attn_mask)  # Question: Can't transform to the Tensor with data type of torch.bool directly.
            attn_mask = attn_mask.data.eq(1)
        for layer in self.attention_layers:
            out, self_attn = layer(out, out, out, attn_mask)
            enc_self_attns.append(self_attn)
        # Pooling.
        # Method I #
        out = self.mil_fc(out)
        # Avg pooling.
        out = torch.sum(out, dim=1) / self.tcr_num
        # Method II #
        # out = self.fc(hidden_states)
        # out = out.reshape(-1, self.tcr_num)
        # out = self.mil_fc(out)
        out = self.dropout(out)
        return out

    def forward_tcr(self, x, attn_mask=None):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # Attention
        enc_self_attns = []
        if not attn_mask:
            attn_mask = torch.full([out.shape[0], self.tcr_num, self.tcr_num], False, dtype=torch.bool)
        else:
            attn_mask = torch.Tensor(attn_mask)
            attn_mask = attn_mask.data.eq(1)
        for layer in self.attention_layers:
            out, self_attn = layer(out, out, out, attn_mask)
            enc_self_attns.append(self_attn)
        # Pooling.
        # Method I #
        out = self.mil_fc(out)
        return out

    def forward_attention(self, x, attn_mask=None):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # Attention
        enc_self_attns = []
        if not attn_mask:
            attn_mask = torch.full([out.shape[0], self.tcr_num, self.tcr_num], False, dtype=torch.bool)
        else:
            attn_mask = torch.Tensor(attn_mask)
            attn_mask = attn_mask.data.eq(1)
        for layer in self.attention_layers:
            out, self_attn = layer(out, out, out, attn_mask)
            enc_self_attns.append(self_attn)
        return enc_self_attns

    def forward_motifs(self, x, attn_mask=None):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        motif_index = [torch.argmax(conv[0](x), dim=-1) for conv in self.convs]
        motif_index = torch.cat(motif_index, dim=1)
        motif_length = []
        for ind, size in enumerate(self.kernel_size):
            for f in range(self.filter_num[ind]):
                motif_length.append(size)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # Attention
        enc_self_attns = []
        if not attn_mask:
            attn_mask = torch.full([out.shape[0], self.tcr_num, self.tcr_num], False, dtype=torch.bool)
        else:
            attn_mask = torch.Tensor(
                attn_mask)  # Question: Can't transform to the Tensor with data type of torch.bool directly.
            attn_mask = attn_mask.data.eq(1)
        for layer in self.attention_layers:
            out, self_attn = layer(out, out, out, attn_mask)
            enc_self_attns.append(self_attn)
        # Pooling.
        # Method I #
        context_layer = out
        out = self.mil_fc(out)
        motif_score_0 = context_layer * self.mil_fc.weight[0] + self.mil_fc.bias[0] / context_layer.shape[-1]
        motif_score_1 = context_layer * self.mil_fc.weight[1] + self.mil_fc.bias[1] / context_layer.shape[-1]
        motif_score = torch.ones(context_layer.shape) / (torch.ones(context_layer.shape) +
                                                         torch.exp(-motif_score_1 + motif_score_0))
        return out, motif_index, motif_length, enc_self_attns, motif_score

    @staticmethod
    def training(sps, lbs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, loss, alpha, beta, gce_q,
                 loss_w=None, shuffle=False, valid_sps=None, valid_lbs=None, attn_mask=None, valid_attn_mask=None):
        # The format of sps is [[tcr_0, tcr_1, ..., tcr_n], ...].
        # The format of lbs is [lb_0, lb_1, ..., lb_n].
        # Generate input.
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        input_batch, label_batch = utils.generate_input_for_training(sps, lbs, aa_v, ins_num=tcr_num)
        input_batch, label_batch = torch.Tensor(input_batch), torch.LongTensor(label_batch)
        dataset = Data.TensorDataset(input_batch, label_batch)
        loader = Data.DataLoader(dataset, len(input_batch), shuffle)
        # Set model.
        model = TransMIL(drop_out=dropout).to(torch.device(device))
        criterion = nn.CrossEntropyLoss().to(device)  # The default loss function is CE.
        weights = []
        if loss_w is not None:
            for w in loss_w:
                weights.append([w] * 2)
            weights = torch.FloatTensor(weights)
        else:
            weights = None
        if loss == "SCE":
            criterion = SCELoss(alpha=alpha, beta=beta, num_classes=2).to(device)
        elif loss == "GCE":
            criterion = GCELoss(q=gce_q, num_classes=2).to(device)
        elif loss == "CE":
            criterion = CELoss(w=weights, num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Training model.
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        max_acc, valid_model_f = 0, model_f + "temp.pth"
        for epoch in range(ep):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x, attn_mask)
                loss = criterion(pred, batch_y)
                if (epoch + 1) % log_inr == 0:
                    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
                    # Valid the model.
                    if valid_sps and valid_lbs:
                        torch.save(model.state_dict(), valid_model_f)
                        valid_probs = TransMIL.prediction(valid_sps, valid_model_f, aa_f, tcr_num, device,
                                                              valid_attn_mask)
                        # utils.evaluation(valid_probs, valid_lbs)  # Debug #
                        valid_preds = [1 if pred > 0.5 else 0 for pred in valid_probs]
                        valid_acc = accuracy_score(valid_lbs, valid_preds)
                        if valid_acc > max_acc:
                            max_acc = valid_acc
                            if os.path.exists(model_f):
                                os.remove(model_f)
                            os.rename(valid_model_f, model_f)
                        else:
                            os.remove(valid_model_f)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Save model.
        if not (valid_sps and valid_lbs):
            torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)
        return 0

    @staticmethod
    def prediction(sps, model_f, aa_f, tcr_num, device, attn_mask=None):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        model = TransMIL().to(device)
        model.load_state_dict(torch.load(model_f))
        model = model.eval()
        # Predict each TCR.
        repertoire_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            if attn_mask:
                predict = model(input_x, [attn_mask[ind]])
            else:
                predict = model(input_x, attn_mask)
            prob = float(1 / (1 + math.exp(-predict[0][1] + predict[0][0])))
            repertoire_scores.append(prob)
        return repertoire_scores

    @staticmethod
    def predict_tcrs(sps, model_f, aa_f, tcr_num, device, attn_mask=None):
        # The format of sps is [[tcr_0, tcr_1, ..., tcr_n], ...].
        # The format of lbs is [lb_0, lb_1, ..., lb_n].
        # Load model.
        model = TransMIL().to(device)
        model_paras = torch.load(model_f)
        model.load_state_dict(model_paras)
        model = model.eval()
        # Predict each TCR repertoire.
        tcr_scores, repertoire_scores = [], []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            if attn_mask:
                predict = model.forward_tcr(input_x, [attn_mask[ind]]).tolist()
            else:
                predict = model.forward_tcr(input_x, attn_mask).tolist()
            probs = []
            for tcr in range(tcr_num):
                probs.append(
                    float(math.exp((-predict[0][tcr][1] + predict[0][tcr][0]) / tcr_num)))
            repertoire_scores.append(probs)
        return repertoire_scores

    @staticmethod
    def predict_attention(sps, model_f, aa_f, tcr_num, device, layer_no=0, attn_mask=None):
        # The format of sps is [[tcr_0, tcr_1, ..., tcr_n], ...].
        # The format of lbs is [lb_0, lb_1, ..., lb_n].
        # Load model.
        model = TransMIL().to(device)
        model_paras = torch.load(model_f)
        model.load_state_dict(model_paras)
        model = model.eval()
        # Predict each TCR repertoire.
        attention_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            if attn_mask:
                predict = model.forward_attention(input_x, [attn_mask[ind]])[layer_no].tolist()
            else:
                predict = model.forward_attention(input_x, attn_mask).tolist()
            attention_scores.append(predict)
        return attention_scores

    @staticmethod
    def motif_identification(sps, sps_name, model_f, aa_f, tcr_num, device, attention_head_num=1,
                             attention_hidden_size=10, topk=0.05):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        model = TransMIL().to(device)
        model_paras = torch.load(model_f)
        model.load_state_dict(model_paras)
        model = model.eval()
        # Predict each TCR.
        sequence_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict, motif_index, motif_length, attn, motif_score = model.forward_motifs(input_x)
            for i, s in enumerate(sp):
                result = [sps_name[ind]]
                seq = s[0]
                result.append(seq)
                result.append(float(1 / (1 + math.exp(-predict[0][i][1] + predict[0][i][0]))))
                motifs = []
                for j, m in enumerate(motif_length):
                    motifs.append([seq[motif_index[i][j]: motif_index[i][j] + m], float(motif_score[0][i][j])])
                motifs = sorted(motifs, key=lambda x: x[1], reverse=True)
                result.append(motifs)
                sequence_scores.append(result)
        return sequence_scores


class BiFormer(nn.Module):
    # BiFormer can predict a TCR repertoire's class (e.g. cancer-associated TCR repertoire) (binary classification).
    # It contains TCRD and the self-attention mechanism with topk for caTCRs identification.
    def __init__(self, tcr_num=100, aa_num=24, feature_num=15, kernel_size=None, filter_num=None, attention_head_num=1,
                 attention_hidden_size=10, drop_out=0.4, topk=0.05):
        super(BiFormer, self).__init__()
        self.tcr_num = tcr_num  # The number of TCR sequences that one input individual sample contains.
        self.aa_num = aa_num  # The number of amino acids that one TCR contains.
        self.feature_num = feature_num  # The dimension of the feature vector of one amino acid.
        if kernel_size is None:
            kernel_size = [2, 3, 4, 5, 6, 7]
        self.kernel_size = kernel_size  # The specification of the convolution kernel in the convolution layer.
        if filter_num is None:
            filter_num = [3, 3, 3, 2, 2, 1]
        assert len(filter_num) == len(kernel_size), \
            "The parameters 'kernel_size' and 'filter_num' set do not match!"
        self.filter_num = filter_num  # The number of the corresponding convolution kernels.
        self.all_filter_num = sum(self.filter_num)
        self.attention_head_num = attention_head_num
        self.attention_hidden_size = attention_hidden_size
        self.attention_head_size = int(self.attention_hidden_size / self.attention_head_num)
        self.drop_out = drop_out
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(self.kernel_size)
        ])
        # Attention
        self.topk = int(self.tcr_num * topk)
        self.query = nn.Linear(self.all_filter_num, self.attention_hidden_size)
        self.key = nn.Linear(self.all_filter_num, self.attention_hidden_size)
        self.value = nn.Linear(self.all_filter_num, self.attention_hidden_size)
        self.attn_fc = nn.Linear(self.attention_hidden_size, self.all_filter_num)
        # Method I: Average #
        self.mil_fc = nn.Linear(self.all_filter_num, 2)
        # Method II: MLP #
        # self.fc = nn.Linear(self.attention_hidden_size, 1)
        # self.mil_fc = nn.Linear(self.tcr_num, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.attention_head_num, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # Attention
        mixed_query_layer = self.query(out)
        mixed_key_layer = self.key(out)
        mixed_value_layer = self.value(out)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # * Topk self-attention
        a_r = torch.matmul(query_layer.detach(), key_layer.detach().transpose(-1, -2))
        _, idx_r = torch.topk(torch.mean(a_r, 1), k=self.topk, dim=-1)
        # ** Compute topk self-attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        idx_r = idx_r.unsqueeze(1).expand(-1, self.attention_head_num, -1, -1)
        value_layer = torch.gather(value_layer.unsqueeze(3).expand(-1, -1, -1, self.tcr_num, -1),
                                   dim=3, index=idx_r.unsqueeze(-1).expand(-1, -1, -1, -1, self.attention_head_size))
        attention_scores = torch.gather(attention_scores, dim=3, index=idx_r)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        attention_probs = attention_probs.unsqueeze(-2)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.flatten(-2, -1)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.attention_hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.attn_fc(context_layer)
        # Pooling.
        # Method I #
        out = self.mil_fc(context_layer)
        # Avg pooling.
        out = torch.sum(out, dim=1) / self.tcr_num
        # Method II #
        # out = self.fc(hidden_states)
        # out = out.reshape(-1, self.tcr_num)
        # out = self.mil_fc(out)
        out = self.dropout(out)
        return out, attention_probs

    def forward_motifs(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        motif_index = [torch.argmax(conv[0](x), dim=-1) for conv in self.convs]
        motif_index = torch.cat(motif_index, dim=1)
        motif_length = []
        for ind, size in enumerate(self.kernel_size):
            for f in range(self.filter_num[ind]):
                motif_length.append(size)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # Attention
        mixed_query_layer = self.query(out)
        mixed_key_layer = self.key(out)
        mixed_value_layer = self.value(out)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # * Topk self-attention
        a_r = torch.matmul(query_layer.detach(), key_layer.detach().transpose(-1, -2))
        _, idx_r = torch.topk(torch.mean(a_r, 1), k=self.topk, dim=-1)
        # ** Compute topk self-attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        idx_r = idx_r.unsqueeze(1).expand(-1, self.attention_head_num, -1, -1)
        value_layer = torch.gather(value_layer.unsqueeze(3).expand(-1, -1, -1, self.tcr_num, -1),
                                   dim=3, index=idx_r.unsqueeze(-1).expand(-1, -1, -1, -1, self.attention_head_size))
        attention_scores = torch.gather(attention_scores, dim=3, index=idx_r)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        attention_probs = attention_probs.unsqueeze(-2)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.flatten(-2, -1)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.attention_hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.attn_fc(context_layer)
        # Pooling.
        # Method I #
        out = self.mil_fc(context_layer)
        motif_score_0 = context_layer * self.mil_fc.weight[0] + self.mil_fc.bias[0] / context_layer.shape[-1]
        motif_score_1 = context_layer * self.mil_fc.weight[1] + self.mil_fc.bias[1] / context_layer.shape[-1]
        motif_score = torch.ones(context_layer.shape) / (torch.ones(context_layer.shape) +
                                                         torch.exp(-motif_score_1 + motif_score_0))
        return out, motif_index, motif_length, attention_probs, motif_score

    @staticmethod
    def training(sps, lbs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, shuffle=False, valid_sps=None,
                 valid_lbs=None):
        # The format of sps is [[tcr_0, tcr_1, ..., tcr_n], ...].
        # The format of lbs is [lb_0, lb_1, ..., lb_n].
        # Generate input.
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        input_batch, label_batch = utils.generate_input_for_training(sps, lbs, aa_v, ins_num=tcr_num)
        input_batch, label_batch = torch.Tensor(input_batch), torch.LongTensor(label_batch)
        dataset = Data.TensorDataset(input_batch, label_batch)
        loader = Data.DataLoader(dataset, len(input_batch), shuffle)
        # Set model.
        model = BiFormer(drop_out=dropout).to(torch.device(device))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Training model.
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        max_acc, valid_model_f = 0, model_f + "temp.pth"
        criterion = nn.CrossEntropyLoss().to(device)
        for epoch in range(ep):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred, attn = model(batch_x)
                loss = criterion(pred, batch_y)
                if (epoch + 1) % log_inr == 0:
                    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
                    # Valid the model.
                    if valid_sps and valid_lbs:
                        torch.save(model.state_dict(), valid_model_f)
                        valid_probs = BiFormer.prediction(valid_sps, valid_model_f, aa_f, tcr_num, device)
                        # utils.evaluation(valid_probs, valid_lbs)  # Debug #
                        valid_preds = [1 if pred > 0.5 else 0 for pred in valid_probs]
                        valid_acc = accuracy_score(valid_lbs, valid_preds)
                        if valid_acc > max_acc:
                            max_acc = valid_acc
                            if os.path.exists(model_f):
                                os.remove(model_f)
                            os.rename(valid_model_f, model_f)
                        else:
                            os.remove(valid_model_f)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Save model.
        if not (valid_sps and valid_lbs):
            torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)
        return 0

    @staticmethod
    def prediction(sps, model_f, aa_f, tcr_num, device):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        model = BiFormer().to(device)
        model.load_state_dict(torch.load(model_f))
        model = model.eval()
        # Predict each TCR.
        repertoire_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict, attn = model(input_x)
            prob = float(1 / (1 + math.exp(-predict[0][1] + predict[0][0])))
            repertoire_scores.append(prob)
        return repertoire_scores

    @staticmethod
    def motif_identification(sps, sps_name, model_f, aa_f, tcr_num, device, attention_head_num=1,
                             attention_hidden_size=10, topk=0.05):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        model = BiFormer().to(device)
        model_paras = torch.load(model_f)
        model.load_state_dict(model_paras)
        model = model.eval()
        # Predict each TCR.
        sequence_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict, motif_index, motif_length, attn, motif_score = model.forward_motifs(input_x)
            for i, s in enumerate(sp):
                result = [sps_name[ind]]
                seq = s[0]
                result.append(seq)
                result.append(float(1 / (1 + math.exp(-predict[0][i][1] + predict[0][i][0]))))
                motifs = []
                for j, m in enumerate(motif_length):
                    motifs.append([seq[motif_index[i][j]: motif_index[i][j] + m], float(motif_score[0][i][j])])
                motifs = sorted(motifs, key=lambda x: x[1], reverse=True)
                result.append(motifs)
                sequence_scores.append(result)
        return sequence_scores


class DeepLION2(nn.Module):
    # DeepLION2 can predict a TCR repertoire's class (e.g. cancer-associated TCR repertoire) (binary classification). 二分类
    # It contains TCRD and the self-attention mechanism with topk and self-learning for caTCRs identification.
    def __init__(self, tcr_num=100, aa_num=24, feature_num=15, kernel_size=None, filter_num=None, attention_head_num=1,
                 attention_hidden_size=10, drop_out=0.4, topk=0.05):
        super(DeepLION2, self).__init__()
        self.tcr_num = tcr_num  # The number of TCR sequences that one input individual sample contains.
        self.aa_num = aa_num  # The number of amino acids that one TCR contains.
        self.feature_num = feature_num  # The dimension of the feature vector of one amino acid.
        if kernel_size is None:
            kernel_size = [2, 3, 4, 5, 6, 7]
        self.kernel_size = kernel_size  # The specification of the convolution kernel in the convolution layer.
        if filter_num is None:
            filter_num = [3, 3, 3, 2, 2, 1]
        assert len(filter_num) == len(kernel_size), \
            "The parameters 'kernel_size' and 'filter_num' set do not match!"
        self.filter_num = filter_num  # The number of the corresponding convolution kernels. 卷积核的数量为14,这个数字后面会用到,后面计算注意力时需要改动
        self.all_filter_num = sum(self.filter_num)
        self.attention_head_num = attention_head_num
        self.attention_hidden_size = attention_hidden_size
        self.attention_head_size = int(self.attention_hidden_size / self.attention_head_num)
        self.drop_out = drop_out
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(self.kernel_size)
        ])
        # Attention，关注的重点
        self.topk = int(self.tcr_num * topk)
        self.query = nn.Linear(self.all_filter_num, self.attention_hidden_size)
        self.key = nn.Linear(self.all_filter_num, self.attention_hidden_size)
        self.value = nn.Linear(self.all_filter_num, self.attention_hidden_size)
        self.attn_fc = nn.Linear(self.attention_hidden_size, self.all_filter_num)
        # Method I: Average #
        self.mil_fc = nn.Linear(self.all_filter_num, 2)
        # Method II: MLP #
        # self.fc = nn.Linear(self.attention_hidden_size, 1)
        # self.mil_fc = nn.Linear(self.tcr_num, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.attention_head_num, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        # 经验证发现原始x为4维，接着x和out都是三维向量，而tcrbert输出的是二维向量，所以需要了解这里三维向量每一维的具体含义
        # print(f"x_ore:{x.shape}") # x_ore:torch.Size([344, 100, 15, 24]) # 在pridect中这里是x_ore:torch.Size([1, 100, 15, 24])
        x = x.reshape(-1, self.feature_num, self.aa_num)
        # print(x)
        # print(f"x.shape:{x.shape}") # x.shape:torch.Size([34400, 15, 24])
        out = [conv(x) for conv in self.convs] # 卷积，注意输入维度和输出维度，这里需要替换
        # print(out)
        # print(out.shape)
        print("###########################")
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # print(out)
        print(f"out.shape:{out.shape}") # out.shape:torch.Size([344, 100, 14]) # predict时应该变成torch.Size([1, 100, 14])
        # 经检查发现,在计算注意力前,输入的out第一维是样本总数,第二维是每个样本的TCR数,第三维是每个TCR的特征维输14.
        # 而tcrbert输出的是每个样本的特征,即对于每一个样本,输出形状为(TCR数量, 768),因此,在这里,只需要两步改动:
        # 1. 将所有样本放到一起, 2. 将768经过线性层变为14
        # 此外,在预处理中还需要进行的一个操作就是计算caTCR_score得分
        # Attention
        mixed_query_layer = self.query(out)
        mixed_key_layer = self.key(out)
        mixed_value_layer = self.value(out)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # * Complete self-attention
        attention_scores_complete = torch.matmul(query_layer.detach(), key_layer.detach().transpose(-1, -2))
        attention_scores_complete = attention_scores_complete / math.sqrt(self.attention_head_size)
        attention_scores_complete = self.dropout(nn.Softmax(dim=-1)(attention_scores_complete))
        context_layer_complete = torch.matmul(attention_scores_complete, value_layer.detach())
        context_layer_complete = context_layer_complete.permute(0, 2, 1, 3).contiguous()
        new_context_layer_complete_shape = context_layer_complete.size()[:-2] + (self.attention_hidden_size,)
        context_layer_complete = context_layer_complete.view(*new_context_layer_complete_shape)
        context_layer_complete = self.attn_fc(context_layer_complete)
        # * Topk self-attention
        a_r = torch.matmul(query_layer.detach(), key_layer.detach().transpose(-1, -2))
        _, idx_r = torch.topk(torch.mean(a_r, 1), k=self.topk, dim=-1)
        # ** Compute topk self-attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        idx_r = idx_r.unsqueeze(1).expand(-1, self.attention_head_num, -1, -1)
        value_layer = torch.gather(value_layer.unsqueeze(3).expand(-1, -1, -1, self.tcr_num, -1),
                                   dim=3, index=idx_r.unsqueeze(-1).expand(-1, -1, -1, -1, self.attention_head_size))
        attention_scores = torch.gather(attention_scores, dim=3, index=idx_r)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        attention_probs = attention_probs.unsqueeze(-2)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.flatten(-2, -1)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.attention_hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.attn_fc(context_layer)
        # Pooling.
        # Method I #
        out = self.mil_fc(context_layer)
        # Avg pooling.
        out = torch.sum(out, dim=1) / self.tcr_num
        # Method II #
        # out = self.fc(hidden_states)
        # out = out.reshape(-1, self.tcr_num)
        # out = self.mil_fc(out)
        out = self.dropout(out)
        return out, attention_probs, context_layer, context_layer_complete

    def forward_motifs(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        motif_index = [torch.argmax(conv[0](x), dim=-1) for conv in self.convs]
        motif_index = torch.cat(motif_index, dim=1)
        motif_length = []
        for ind, size in enumerate(self.kernel_size):
            for f in range(self.filter_num[ind]):
                motif_length.append(size)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # Attention
        mixed_query_layer = self.query(out)
        mixed_key_layer = self.key(out)
        mixed_value_layer = self.value(out)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # * Topk self-attention
        a_r = torch.matmul(query_layer.detach(), key_layer.detach().transpose(-1, -2))
        _, idx_r = torch.topk(torch.mean(a_r, 1), k=self.topk, dim=-1)
        # ** Compute topk self-attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        idx_r = idx_r.unsqueeze(1).expand(-1, self.attention_head_num, -1, -1)
        value_layer = torch.gather(value_layer.unsqueeze(3).expand(-1, -1, -1, self.tcr_num, -1),
                                   dim=3, index=idx_r.unsqueeze(-1).expand(-1, -1, -1, -1, self.attention_head_size))
        attention_scores = torch.gather(attention_scores, dim=3, index=idx_r)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        attention_probs = attention_probs.unsqueeze(-2)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.flatten(-2, -1)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.attention_hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.attn_fc(context_layer)
        # Pooling.
        # Method I #
        out = self.mil_fc(context_layer)
        motif_score_0 = context_layer * self.mil_fc.weight[0] + self.mil_fc.bias[0] / context_layer.shape[-1]
        motif_score_1 = context_layer * self.mil_fc.weight[1] + self.mil_fc.bias[1] / context_layer.shape[-1]
        motif_score = torch.ones(context_layer.shape) / (torch.ones(context_layer.shape) +
                                                         torch.exp(-motif_score_1 + motif_score_0))
        return out, motif_index, motif_length, attention_probs, motif_score

    @staticmethod
    def training(sps, lbs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, shuffle=False, valid_sps=None,
                 valid_lbs=None, attention_head_num=1, attention_hidden_size=10, topk=0.05, pretrained_model_path=None):
        # The format of sps is [[tcr_0, tcr_1, ..., tcr_n], ...]. 列表,所有样本信息，tcr_0的格式如下：['CASSLSLHNEQFF', 'TRBV11-2*01', '0.000338', '0.4406363161932763']
        # 每个样本有100条TCR，也就是说n = 100
        # The format of lbs is [lb_0, lb_1, ..., lb_n]. 这里n等于样本数
        # Generate input.
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors. 返回一个字典，其中键是氨基酸标识，值是对应的标准化后的特征向量，大小为15
        input_batch, label_batch = utils.generate_input_for_training(sps, lbs, aa_v, ins_num=tcr_num) # 这里搞清楚input_batch到底是什么
        # print(f"type_label_batch:{type(label_batch)}") # type_label_batch:<class 'numpy.ndarray'>
        # print(f"type_input_batch:{type(input_batch)}") # type_input_batch:<class 'numpy.ndarray'>
        # 经检查得知，input_batch的形状为344*100*15*24，344是样本数，100是每个样本TCR的数量，15是每个氨基酸的特征维数，24是每个TCR序列的氨基酸数目
        input_batch, label_batch = torch.Tensor(input_batch), torch.LongTensor(label_batch)
        # dataset是一个TensorDataset对象，它封装了input_batch和label_batch两个张量。
        # TensorDataset将这两个张量打包在一起，使得每次迭代可以同时返回一对样本和标签。
        dataset = Data.TensorDataset(input_batch, label_batch)
        # loader是一个DataLoader对象，用于封装dataset，提供了一个迭代接口来批量获取数据和标签，支持自动批处理、样本打乱和多进程数据加载等功能
        loader = Data.DataLoader(dataset, len(input_batch), shuffle) # 批次大小等于样本总数,每次迭代返回整个数据集，相当于没有采用真正的批处理
        # Set model.
        model = DeepLION2(drop_out=dropout, attention_head_num=1, attention_hidden_size=10, topk=0.05).\
            to(torch.device(device))
        # 加载预训练的模型
        if pretrained_model_path is not None: 
            pretrained_dict = torch.load(pretrained_model_path, map_location=device)
            model.load_state_dict(pretrained_dict, strict=False)
            print(f"Loaded model from {pretrained_model_path}")

        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Training model.
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        max_acc, valid_model_f = 0, model_f + "temp.pth"
        criterion_1 = nn.CrossEntropyLoss().to(device)
        criterion_2 = nn.MSELoss().to(device)
        for epoch in range(ep):
            # 批次大小等于样本总数,相当于加载了所有的样本
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred, attn, attn_out, attn_out_complete = model(batch_x) # 前向传播，主要的改动在这里，batch_x是4维，来自loader,loader来自dataset
                # debug
                print(f"pred:{len(pred)}")
                print(f"batch_y:{len(batch_y)}")
                loss = criterion_1(pred, batch_y) + criterion_2(attn_out, attn_out_complete) # 计算损失
                if (epoch + 1) % log_inr == 0:
                    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
                    # Valid the model.
                    if valid_sps and valid_lbs:
                        torch.save(model.state_dict(), valid_model_f)
                        valid_probs = DeepLION2.prediction(valid_sps, valid_model_f, aa_f, tcr_num, device)
                        # utils.evaluation(valid_probs, valid_lbs)  # Debug #
                        valid_preds = [1 if pred > 0.5 else 0 for pred in valid_probs] # 0.5是阈值
                        valid_acc = accuracy_score(valid_lbs, valid_preds)
                        if valid_acc > max_acc:
                            max_acc = valid_acc
                            if os.path.exists(model_f):
                                os.remove(model_f)
                            os.rename(valid_model_f, model_f)
                        else:
                            os.remove(valid_model_f)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Save model. 保存训练好的模型
        if not (valid_sps and valid_lbs):
            torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)
        return 0

    @staticmethod
    def prediction(sps, model_f, aa_f, tcr_num, device, attention_head_num=1, attention_hidden_size=10, topk=0.05):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        print(f"#############################{len(sps[2])}") # sps是一个列表,列表中有86个样本,每个样本有100条TCR数据
        model = DeepLION2(attention_head_num=1, attention_hidden_size=10, topk=0.05).to(device)
        model.load_state_dict(torch.load(model_f))
        model = model.eval()
        # Predict each TCR.
        repertoire_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        # 对于每一个样本
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            print(f"input_x:{len(input_x[0][0])}") # input_x:15
            input_x = torch.Tensor(input_x).to(torch.device(device))
            print(input_x.shape) # torch.Size([1, 100, 15, 24])
            # Make prediction.
            predict, attn, _, _ = model(input_x)
            prob = float(1 / (1 + math.exp(-predict[0][1] + predict[0][0])))
            repertoire_scores.append(prob)
        return repertoire_scores

    @staticmethod
    def motif_identification(sps, sps_name, model_f, aa_f, tcr_num, device, attention_head_num=1, attention_hidden_size=10, topk=0.05):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        model = DeepLION2(attention_head_num=1, attention_hidden_size=10, topk=0.05).to(device)
        model_paras = torch.load(model_f)
        model.load_state_dict(model_paras)
        model = model.eval()
        # Predict each TCR.
        sequence_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict, motif_index, motif_length, attn, motif_score = model.forward_motifs(input_x)
            for i, s in enumerate(sp):
                result = [sps_name[ind]]
                seq = s[0]
                result.append(seq)
                result.append(float(1 / (1 + math.exp(-predict[0][i][1] + predict[0][i][0]))))
                motifs = []
                for j, m in enumerate(motif_length):
                    motifs.append([seq[motif_index[i][j]: motif_index[i][j] + m], float(motif_score[0][i][j])])
                motifs = sorted(motifs, key=lambda x: x[1], reverse=True)
                result.append(motifs)
                sequence_scores.append(result)
        return sequence_scores

class DeepLION2_bert(nn.Module):
    # DeepLION2_bert can predict a TCR repertoire's class (e.g. cancer-associated TCR repertoire) (binary classification). 二分类,将CNN换成bert提取特征
    # It contains TCRD and the self-attention mechanism with topk and self-learning for caTCRs identification.
    def __init__(self, tcr_num=100, aa_num=24, feature_num=15, kernel_size=None, filter_num=None, attention_head_num=1,
                 attention_hidden_size=10, drop_out=0.4, topk=0.05):
        super(DeepLION2_bert, self).__init__()
        self.tcr_num = tcr_num  # The number of TCR sequences that one input individual sample contains. 一个样本的TCR数目
        self.aa_num = aa_num  # The number of amino acids that one TCR contains. 一个TCR包含的氨基酸数目
        self.feature_num = feature_num  # The dimension of the feature vector of one amino acid. 氨基酸特征矩阵的维数
        if kernel_size is None: # 卷积相关,注释
            kernel_size = [2, 3, 4, 5, 6, 7]
        self.kernel_size = kernel_size  # The specification of the convolution kernel in the convolution layer.
        if filter_num is None:
            filter_num = [3, 3, 3, 2, 2, 1]
        assert len(filter_num) == len(kernel_size), \
            "The parameters 'kernel_size' and 'filter_num' set do not match!"
        self.filter_num = filter_num  # The number of the corresponding convolution kernels. 
        # self.all_filter_num = sum(self.filter_num) # 卷积核的总数为14,这个数字后面会用到,后面计算注意力时需要改动
        self.all_filter_num = 768
        # 注意力相关的,不改
        self.attention_head_num = attention_head_num
        self.attention_hidden_size = attention_hidden_size
        self.attention_head_size = int(self.attention_hidden_size / self.attention_head_num)
        self.drop_out = drop_out
        # 卷积相关,注释
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(self.kernel_size)
        ])
        # Attention，关注的重点
        self.topk = int(self.tcr_num * topk)
        self.query = nn.Linear(self.all_filter_num, self.attention_hidden_size)
        self.key = nn.Linear(self.all_filter_num, self.attention_hidden_size)
        self.value = nn.Linear(self.all_filter_num, self.attention_hidden_size)
        self.attn_fc = nn.Linear(self.attention_hidden_size, self.all_filter_num)
        # Method I: Average #
        self.mil_fc = nn.Linear(self.all_filter_num, 2)
        # Method II: MLP #
        # self.fc = nn.Linear(self.attention_hidden_size, 1)
        # self.mil_fc = nn.Linear(self.tcr_num, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.attention_head_num, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        # 经验证发现原始x为4维，接着x和out都是三维向量，而tcrbert输出的是二维向量，所以需要了解这里三维向量每一维的具体含义
        print(f"x_ore:{x.shape}") # x_ore:torch.Size([344, 100, 15, 24]) 现在是x_ore:torch.Size([343, 100, 768])
        # x = x.reshape(-1, self.feature_num, self.aa_num)
        # print(x)
        # print(f"x.shape:{x.shape}") # x.shape:torch.Size([34400, 15, 24])
        # out = [conv(x) for conv in self.convs] # 卷积，注意输入维度和输出维度，这里需要替换
        # print(out)
        # print(out.shape)
        print("###########################")
        # out = torch.cat(out, dim=1)
        # out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # print(out)
        # print(f"out.shape:{out.shape}") # out.shape:torch.Size([344, 100, 14])
        # 经检查发现,在计算注意力前,输入的out第一维是样本总数,第二维是每个样本的TCR数,第三维是每个TCR的特征维输14.
        # 而tcrbert输出的是每个样本的特征,即对于每一个样本,输出形状为(TCR数量, 768),因此,在这里,只需要两步改动:
        # 1. 将所有样本放到一起, 2. 将768经过线性层变为14
        # 此外,在预处理中还需要进行的一个操作就是计算caTCR_score得分
        # Attention
        mixed_query_layer = self.query(x)
        mixed_key_layer = self.key(x)
        mixed_value_layer = self.value(x)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # * Complete self-attention
        attention_scores_complete = torch.matmul(query_layer.detach(), key_layer.detach().transpose(-1, -2))
        attention_scores_complete = attention_scores_complete / math.sqrt(self.attention_head_size)
        attention_scores_complete = self.dropout(nn.Softmax(dim=-1)(attention_scores_complete))
        context_layer_complete = torch.matmul(attention_scores_complete, value_layer.detach())
        context_layer_complete = context_layer_complete.permute(0, 2, 1, 3).contiguous()
        new_context_layer_complete_shape = context_layer_complete.size()[:-2] + (self.attention_hidden_size,)
        context_layer_complete = context_layer_complete.view(*new_context_layer_complete_shape)
        context_layer_complete = self.attn_fc(context_layer_complete)
        # * Topk self-attention
        a_r = torch.matmul(query_layer.detach(), key_layer.detach().transpose(-1, -2))
        _, idx_r = torch.topk(torch.mean(a_r, 1), k=self.topk, dim=-1)
        # ** Compute topk self-attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        idx_r = idx_r.unsqueeze(1).expand(-1, self.attention_head_num, -1, -1)
        value_layer = torch.gather(value_layer.unsqueeze(3).expand(-1, -1, -1, self.tcr_num, -1),
                                   dim=3, index=idx_r.unsqueeze(-1).expand(-1, -1, -1, -1, self.attention_head_size))
        attention_scores = torch.gather(attention_scores, dim=3, index=idx_r)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        attention_probs = attention_probs.unsqueeze(-2)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.flatten(-2, -1)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.attention_hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.attn_fc(context_layer)
        print(f"context_layer:{context_layer.shape}")
        # Pooling.
        # Method I #
        out = self.mil_fc(context_layer)
        print(f"out:{out.shape}")
        # Avg pooling.
        out = torch.sum(out, dim=1) / self.tcr_num
        # Method II #
        # out = self.fc(hidden_states)
        # out = out.reshape(-1, self.tcr_num)
        # out = self.mil_fc(out)
        out = self.dropout(out)
        return out, attention_probs, context_layer, context_layer_complete

    def forward_motifs(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        motif_index = [torch.argmax(conv[0](x), dim=-1) for conv in self.convs]
        motif_index = torch.cat(motif_index, dim=1)
        motif_length = []
        for ind, size in enumerate(self.kernel_size):
            for f in range(self.filter_num[ind]):
                motif_length.append(size)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # Attention
        mixed_query_layer = self.query(out)
        mixed_key_layer = self.key(out)
        mixed_value_layer = self.value(out)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # * Topk self-attention
        a_r = torch.matmul(query_layer.detach(), key_layer.detach().transpose(-1, -2))
        _, idx_r = torch.topk(torch.mean(a_r, 1), k=self.topk, dim=-1)
        # ** Compute topk self-attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        idx_r = idx_r.unsqueeze(1).expand(-1, self.attention_head_num, -1, -1)
        value_layer = torch.gather(value_layer.unsqueeze(3).expand(-1, -1, -1, self.tcr_num, -1),
                                   dim=3, index=idx_r.unsqueeze(-1).expand(-1, -1, -1, -1, self.attention_head_size))
        attention_scores = torch.gather(attention_scores, dim=3, index=idx_r)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        attention_probs = attention_probs.unsqueeze(-2)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.flatten(-2, -1)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.attention_hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.attn_fc(context_layer)
        # Pooling.
        # Method I #
        out = self.mil_fc(context_layer)
        motif_score_0 = context_layer * self.mil_fc.weight[0] + self.mil_fc.bias[0] / context_layer.shape[-1]
        motif_score_1 = context_layer * self.mil_fc.weight[1] + self.mil_fc.bias[1] / context_layer.shape[-1]
        motif_score = torch.ones(context_layer.shape) / (torch.ones(context_layer.shape) +
                                                         torch.exp(-motif_score_1 + motif_score_0))
        return out, motif_index, motif_length, attention_probs, motif_score

    @staticmethod
    def training(sps, lbs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, shuffle=False, valid_sps=None,
                 valid_lbs=None, attention_head_num=1, attention_hidden_size=10, topk=0.05):
        # The format of sps is [[tcr_0, tcr_1, ..., tcr_n], ...]. 所有样本信息，tcr_0的格式如下：['CASSLSLHNEQFF', 'TRBV11-2*01', '0.000338', '0.4406363161932763']
        # 每个样本有100条TCR，也就是说n = 100
        # The format of lbs is [lb_0, lb_1, ..., lb_n]. 这里n等于样本数
        # Generate input.
        # aa_v = utils.get_features(aa_f)  # Read amino acid vectors. 返回一个字典，其中键是氨基酸标识，值是对应的标准化后的特征向量，大小为15
        # input_batch, label_batch = utils.generate_input_for_training(sps, lbs, aa_v, ins_num=tcr_num) # 这里搞清楚input_batch到底是什么
        # 经检查得知，input_batch的形状为344*100*15*24，344是样本数，100是每个样本TCR的数量，15是每个氨基酸的特征维数，24是每个TCR序列的氨基酸数目
        # 准备在这里提取特征
        # input_batch = np.array(sps)
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        print("sps:", sps) # 为空
        input_batch = tcrbert.process_sps(sps)
        label_batch = np.array(lbs)
        print(f"type_label_batch:{type(label_batch)}") # type_label_batch:<class 'numpy.ndarray'>
        print(f"type_input_batch:{type(input_batch)}") # type_input_batch:<class 'numpy.ndarray'>
        input_batch, label_batch = torch.Tensor(input_batch), torch.LongTensor(label_batch)
        # dataset是一个TensorDataset对象，它封装了input_batch和label_batch两个张量。
        # TensorDataset将这两个张量打包在一起，使得每次迭代可以同时返回一对样本和标签。
        dataset = Data.TensorDataset(input_batch, label_batch)
        # loader是一个DataLoader对象，用于封装dataset，提供了一个迭代接口来批量获取数据和标签，支持自动批处理、样本打乱和多进程数据加载等功能
        loader = Data.DataLoader(dataset, len(input_batch), shuffle) # 批次大小等于样本总数,每次迭代返回整个数据集，相当于没有采用真正的批处理
        # Set model.
        model = DeepLION2_bert(drop_out=dropout, attention_head_num=1, attention_hidden_size=10, topk=0.05).\
            to(torch.device(device))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Training model.
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        max_acc, valid_model_f = 0, model_f + "temp.pth"
        criterion_1 = nn.CrossEntropyLoss().to(device)
        criterion_2 = nn.MSELoss().to(device)
        for epoch in range(ep):
            # 批次大小等于样本总数,相当于加载了所有的样本
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred, attn, attn_out, attn_out_complete = model(batch_x) # 前向传播，主要的改动在这里，batch_x是4维，来自loader,loader来自dataset
                print(f"pred:{pred}")
                print(f"batch_y:{batch_y}")
                loss = criterion_1(pred, batch_y) + criterion_2(attn_out, attn_out_complete) # 计算损失
                if (epoch + 1) % log_inr == 0:
                    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
                    # Valid the model.
                    if valid_sps and valid_lbs:
                        torch.save(model.state_dict(), valid_model_f)
                        valid_probs = DeepLION2.prediction(valid_sps, valid_model_f, aa_f, tcr_num, device)
                        # utils.evaluation(valid_probs, valid_lbs)  # Debug #
                        valid_preds = [1 if pred > 0.5 else 0 for pred in valid_probs] # 0.5是阈值
                        valid_acc = accuracy_score(valid_lbs, valid_preds)
                        if valid_acc > max_acc:
                            max_acc = valid_acc
                            if os.path.exists(model_f):
                                os.remove(model_f)
                            os.rename(valid_model_f, model_f)
                        else:
                            os.remove(valid_model_f)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Save model. 保存训练好的模型
        if not (valid_sps and valid_lbs):
            torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)
        return 0

    @staticmethod
    def prediction(sps, model_f, aa_f, tcr_num, device, attention_head_num=1, attention_hidden_size=10, topk=0.05):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        print(sps) # 共86个样本,每个样本还是100条TCR,每条TCR还是包含四个数据
        model = DeepLION2_bert(attention_head_num=1, attention_hidden_size=10, topk=0.05).to(device)
        model.load_state_dict(torch.load(model_f))
        model = model.eval()
        # Predict each TCR.
        repertoire_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
            input_x = tcrbert.process_sp(sp)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            print(f"input_xx:{input_x.shape}") # 这里应该是[1, 100, 768]
            # Make prediction.
            predict, attn, _, _ = model(input_x)
            prob = float(1 / (1 + math.exp(-predict[0][1] + predict[0][0])))
            repertoire_scores.append(prob)
        return repertoire_scores

    @staticmethod
    def motif_identification(sps, sps_name, model_f, aa_f, tcr_num, device, attention_head_num=1, attention_hidden_size=10, topk=0.05):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        model = DeepLION2(attention_head_num=1, attention_hidden_size=10, topk=0.05).to(device)
        model_paras = torch.load(model_f)
        model.load_state_dict(model_paras)
        model = model.eval()
        # Predict each TCR.
        sequence_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict, motif_index, motif_length, attn, motif_score = model.forward_motifs(input_x)
            for i, s in enumerate(sp):
                result = [sps_name[ind]]
                seq = s[0]
                result.append(seq)
                result.append(float(1 / (1 + math.exp(-predict[0][i][1] + predict[0][i][0]))))
                motifs = []
                for j, m in enumerate(motif_length):
                    motifs.append([seq[motif_index[i][j]: motif_index[i][j] + m], float(motif_score[0][i][j])])
                motifs = sorted(motifs, key=lambda x: x[1], reverse=True)
                result.append(motifs)
                sequence_scores.append(result)
        return sequence_scores

class DeepLION2_GAT(nn.Module):
    def __init__(self, tcr_num=300, feature_dim=768, attention_head_num=4, attention_hidden_size=200, num_gat_layers=2, drop_out=0.4):
        super(DeepLION2_GAT, self).__init__()
        self.tcr_num = tcr_num  # The number of TCR sequences per sample
        self.feature_dim = feature_dim  # Dimension of each TCR sequence feature (from TCRbert)
        self.num_gat_layers = num_gat_layers
        
        # Attention-related parameters
        self.attention_head_num = attention_head_num  # Number of attention heads
        self.attention_hidden_size = attention_hidden_size  # Hidden size for attention
        self.attention_head_size = int(attention_hidden_size / attention_head_num)  # 100 Size of each attention head
        
        # Graph Attention Layers (we will define these layers below)
        # GAT 可以运行
        # self.attentions = nn.ModuleList([
        #     GraphAttentionLayer(feature_dim, self.attention_head_size, dropout=drop_out)
        #     for _ in range(attention_head_num)
        # ])
        
        # # GATv2 试一下
        # self.attentions = nn.ModuleList([
        #     GraphAttentionLayerV2(feature_dim, self.attention_head_size, dropout=drop_out)
        #     for _ in range(attention_head_num)
        # ])
        # 换成两层
        self.gat_layers = nn.ModuleList([
            nn.ModuleList([
                GraphAttentionLayerV2(feature_dim if i == 0 else self.attention_hidden_size, self.attention_head_size, dropout=drop_out)
                for _ in range(attention_head_num)
            ]) for i in range(num_gat_layers)
        ])

        # Output fully connected layer
        # self.out_fc = nn.Linear(tcr_num, 2)
        self.fc1 = nn.Linear(self.attention_head_size * attention_head_num, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x, adj):
        # x: Node features, assumed to be [batch_size, tcr_num, feature_dim]
        # adj: Adjacency matrix, assumed to be [batch_size, tcr_num, tcr_num]
        # The values in adj are not binary but represent the weighted distance between nodes.

        # Node features are initially of size [batch_size, tcr_num, feature_dim]
        # print(f"Initial x shape: {x.shape}")  # Debug: check the shape of the input features # torch.Size([8, 100, 768])
        # print(f"Adjacency matrix shape: {adj.shape}")  # Debug: check the shape of the adjacency matrix # torch.Size([8, 100, 100])
 
        # Apply dropout to input features
        x = self.dropout(x)
        
        # Apply the graph attention layers
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        # 换成两层
        for gat_layer in self.gat_layers:
            x = torch.cat([att(x, adj) for att in gat_layer], dim=-1)
        # After concatenating the output of each attention head, x should have shape:
        # [batch_size, tcr_num, feature_dim * attention_head_num]
        # print(f"Shape after attention: {x.shape}")  # Debug: check the shape after attention

        # Mean pooling across the node dimension to aggregate node features into a single vector per sample
        x = torch.mean(x, dim=1)  # [batch_size, feature_dim * attention_head_num]
        # print(f"x:{x.shape}")
        # Apply a final linear transformation
        # x = self.out_fc(x)  # [batch_size, 2] x:torch.Size([8, 2])
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # Assuming want a single probability per sample, apply softmax and select one class's probability,
        # typically, this would be the probability of class 1 for binary classification
        return x
    @staticmethod
    def validate_model(model, valid_sps, valid_lbs, valid_mat, device):
        # Validation logic to compute the accuracy
        model.eval()
        # 准备数据
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(valid_sps) # 计算节点特征
        adjacencies = np.array(valid_mat)
        labels = np.array(valid_lbs)

        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        labels_tensor = torch.LongTensor(labels).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(features_tensor, adjacencies_tensor)

        # Get probabilities and predicted classes
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probabilities for class 1
        preds = (probs > 0.5).astype(int)  # Convert probabilities to 0 or 1 based on threshold

        # Calculate accuracy, AUC, and F1 score
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, preds)

        # print(f"Validation Accuracy: {acc:.4f}")
        # print(f"Validation AUC: {auc:.4f}")
        # print(f"Validation F1 Score: {f1:.4f}")

        return acc, auc, f1

    @staticmethod
    def training(sps, lbs, adjs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, shuffle=False, valid_sps=None, valid_lbs=None, valid_mat=None):
        # Assume sps includes both features and adjacency matrices for each sample
        # Assume lbs is a list of labels corresponding to each sample in sps

        # Convert features and adjacency matrices to tensors and pack them
        # sps已经包含了节点特征和邻接矩阵
        # features = [sp['features'] for sp in sps]
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps) # 节点特征
        # adjacencies = [sp['adjacency'] for sp in sps]
        adjacencies = np.array(adjs) # 所有样本的邻接矩阵
        lbs = np.array(lbs) # 样本标签
        features_tensor = torch.Tensor(features)
        adjacencies_tensor = torch.Tensor(adjacencies)
        labels_tensor = torch.LongTensor(lbs)

        # Create a dataset and loader
        dataset = TensorDataset(features_tensor, adjacencies_tensor, labels_tensor)
        # loader = DataLoader(dataset, batch_size=len(features_tensor), shuffle=shuffle)
        loader = DataLoader(dataset, batch_size=64, shuffle=shuffle)
        # print(f"batch_size:{len(features_tensor)}")

        # Initialize the model
        model = DeepLION2_GAT(tcr_num=tcr_num, feature_dim=768, drop_out=dropout, num_gat_layers=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss().to(device)

        # Check if a model file already exists and adjust the filename
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        valid_model_f = model_f + "temp.pth"

        epoch_losses = []
        # Training loop
        for epoch in range(ep):
            total_loss = 0
            for batch_x, batch_adj, batch_y in loader:
                batch_x, batch_adj, batch_y = batch_x.to(device), batch_adj.to(device), batch_y.to(device)

                # print(f"batch_x:{batch_x.shape}") # torch.Size([8, 100, 768])
                # print(f"batch_adj:{batch_adj.shape}") # torch.Size([8, 100, 100])
                # print(f"batch_y:{batch_y.shape}")
                # Forward pass
                outputs = model(batch_x, batch_adj)
                # print(f"outputs:{outputs.shape}")
                # print(outputs)
                # print(batch_y)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                # print(f"epoch{epoch}, loss = {loss}")

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_losses.append(total_loss)
            print(f"Epoch {epoch + 1}, Total Loss = {total_loss}")

            # Logging
            if (epoch + 1) % log_inr == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss.item()))

            # print("valid_sps:", valid_sps)
            # Optional validation step
            if valid_sps is not None and valid_lbs is not None and valid_mat is not None:
                valid_acc, valid_auc, valid_f1 = DeepLION2_GAT.validate_model(model, valid_sps, valid_lbs, valid_mat, device)
                print('Validation Accuracy:', valid_acc)
                print('Validation AUC:', valid_auc)
                print('Validation f1 score:', valid_f1)
                # Save the best model
                if valid_acc > max_acc:
                    max_acc = valid_acc
                    torch.save(model.state_dict(), valid_model_f)
                    if os.path.exists(model_f):
                        os.remove(model_f)
                    os.rename(valid_model_f, model_f)
                else:
                    os.remove(valid_model_f)

        # Save the final model
        torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)
        epoch_losses_df = pd.DataFrame(epoch_losses, columns=["Loss"])

        return epoch_losses_df

    @staticmethod
    def prediction(sps, dismat, model_f, tcr_num, device):
        # 加载模型
        model = DeepLION2_GAT(tcr_num=tcr_num, feature_dim=768, attention_head_num=4, attention_hidden_size=200, drop_out=0.4).to(device)
        model.load_state_dict(torch.load(model_f, map_location=device))
        model.eval()

        # 准备测试数据
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 计算节点特征
        adjacencies = np.array(dismat)
        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)

        # forward
        with torch.no_grad():
            outputs = model(features_tensor, adjacencies_tensor)

        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 获取类别1的概率

        return probs

class DeepLION2_GAT_div(nn.Module):
    def __init__(self, tcr_num=300, feature_dim=768, attention_head_num=4, attention_hidden_size=200, num_gat_layers=2, drop_out=0.4, diversity_features_dim=10):
        super(DeepLION2_GAT_div, self).__init__()
        self.tcr_num = tcr_num  # The number of TCR sequences per sample
        self.feature_dim = feature_dim  # Dimension of each TCR sequence feature (from TCRbert)
        self.diversity_features_dim = diversity_features_dim
        self.num_gat_layers = num_gat_layers
        
        # Attention-related parameters
        self.attention_head_num = attention_head_num  # Number of attention heads
        self.attention_hidden_size = attention_hidden_size  # Hidden size for attention
        self.attention_head_size = int(attention_hidden_size / attention_head_num)  # 100 Size of each attention head
        
        # Graph Attention Layers (we will define these layers below)
        # GAT 可以运行
        # self.attentions = nn.ModuleList([
        #     GraphAttentionLayer(feature_dim, self.attention_head_size, dropout=drop_out)
        #     for _ in range(attention_head_num)
        # ])
        
        # GATv2 试一下
        # self.attentions = nn.ModuleList([
        #     GraphAttentionLayerV2(feature_dim, self.attention_head_size, dropout=drop_out)
        #     for _ in range(attention_head_num)
        # ])
        
        # 换成两层
        self.gat_layers = nn.ModuleList([
            nn.ModuleList([
                GraphAttentionLayerV2(feature_dim if i == 0 else self.attention_hidden_size, self.attention_head_size, dropout=drop_out)
                for _ in range(attention_head_num)
            ]) for i in range(num_gat_layers)
        ])
        # Output fully connected layer
        # self.out_fc = nn.Linear(tcr_num, 2)
        # 换成两层
        self.fc1 = nn.Linear(self.attention_head_size * attention_head_num + diversity_features_dim, 128)
        self.fc2 = nn.Linear(128, 2)
        # self.out_fc = nn.Linear(self.attention_head_size * attention_head_num + diversity_features_dim, 2)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x, adj, diversity_features):
        # x: Node features, assumed to be [batch_size, tcr_num, feature_dim]
        # adj: Adjacency matrix, assumed to be [batch_size, tcr_num, tcr_num]
        # The values in adj are not binary but represent the weighted distance between nodes.

        # Node features are initially of size [batch_size, tcr_num, feature_dim]
        # print(f"Initial x shape: {x.shape}")  # Debug: check the shape of the input features # torch.Size([8, 100, 768])
        # print(f"Adjacency matrix shape: {adj.shape}")  # Debug: check the shape of the adjacency matrix # torch.Size([8, 100, 100])
 
        # Apply dropout to input features
        x = self.dropout(x)
        
        for gat_layer in self.gat_layers:
            x = torch.cat([att(x, adj) for att in gat_layer], dim=-1)

        # Apply the graph attention layers
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        # After concatenating the output of each attention head, x should have shape:
        # [batch_size, tcr_num, feature_dim * attention_head_num]
        # print(f"Shape after attention: {x.shape}")  # Debug: check the shape after attention

        # Mean pooling across the node dimension to aggregate node features into a single vector per sample
        x = torch.mean(x, dim=1)  # [batch_size, feature_dim * attention_head_num]
        
        # # 标准化
        # x = self.x_scaler.fit_transform(x.cpu()).to(x.device)
        # diversity_features = self.diversity_scaler.fit_transform(diversity_features.cpu()).to(diversity_features.device)

        x = torch.cat([x, diversity_features], dim=-1) 
        # print(f"x:{x.shape}")
        # Apply a final linear transformation
        # x = self.out_fc(x)  # [batch_size, 2] x:torch.Size([8, 2])
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Assuming want a single probability per sample, apply softmax and select one class's probability,
        # typically, this would be the probability of class 1 for binary classification
        return x
    
    @staticmethod
    def validate_model(model, valid_sps, valid_lbs, valid_mat, device):
        # Validation logic to compute the accuracy
        model.eval()
        # 准备数据
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(valid_sps) # 计算节点特征
        adjacencies = np.array(valid_mat)
        labels = np.array(valid_lbs)

        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        labels_tensor = torch.LongTensor(labels).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(features_tensor, adjacencies_tensor)

        # Get probabilities and predicted classes
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probabilities for class 1
        preds = (probs > 0.5).astype(int)  # Convert probabilities to 0 or 1 based on threshold

        # Calculate accuracy, AUC, and F1 score
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, preds)

        # print(f"Validation Accuracy: {acc:.4f}")
        # print(f"Validation AUC: {auc:.4f}")
        # print(f"Validation F1 Score: {f1:.4f}")

        return acc, auc, f1

    @staticmethod
    def training(sps, lbs, adjs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, diversity_features_df, shuffle=False, valid_sps=None, valid_lbs=None, valid_mat=None):
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 节点特征

        adjacencies = np.array(adjs)  # 所有样本的邻接矩阵
        lbs = np.array(lbs)  # 样本标签

        features_tensor = torch.Tensor(features)
        adjacencies_tensor = torch.Tensor(adjacencies)
        labels_tensor = torch.LongTensor(lbs)
        diversity_features_tensor = torch.Tensor(diversity_features_df.values)

        # # debug
        # print(f"features_tensor size: {features_tensor.size()}")
        # print(f"adjacencies_tensor size: {adjacencies_tensor.size()}")
        # print(f"labels_tensor size: {labels_tensor.size()}")
        # print(f"diversity_features_tensor size: {diversity_features_tensor.size()}")

        # Initialize the model
        diversity_features_dim = diversity_features_df.shape[1]
        model = DeepLION2_GAT_div(tcr_num=tcr_num, feature_dim=768, drop_out=dropout, diversity_features_dim=diversity_features_dim, num_gat_layers=2).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss().to(device)

        # Check if a model file already exists and adjust the filename
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        valid_model_f = model_f + "temp.pth"

        # Create a dataset and loader
        dataset = TensorDataset(features_tensor, adjacencies_tensor, labels_tensor, diversity_features_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=shuffle) # 这里一开始是100

        epoch_losses = []
        # Training loop
        for epoch in range(ep):
            total_loss = 0
            for batch_x, batch_adj, batch_y, batch_diversity in loader:
                batch_x, batch_adj, batch_y, batch_diversity = batch_x.to(device), batch_adj.to(device), batch_y.to(device), batch_diversity.to(device)

                outputs = model(batch_x, batch_adj, batch_diversity)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                # print(f"epoch {epoch}, loss = {loss}")

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_losses.append(total_loss)
            print(f"Epoch {epoch + 1}, Total Loss = {total_loss}")
            # Logging
            if (epoch + 1) % log_inr == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss.item()))

            # Optional validation step
            if valid_sps is not None and valid_lbs is not None and valid_mat is not None:
                valid_acc, valid_auc, valid_f1 = DeepLION2_GAT.validate_model(model, valid_sps, valid_lbs, valid_mat, device, diversity_features_df)
                print('Validation Accuracy:', valid_acc)
                print('Validation AUC:', valid_auc)
                print('Validation f1 score:', valid_f1)
                # Save the best model
                if valid_acc > max_acc:
                    max_acc = valid_acc
                    torch.save(model.state_dict(), valid_model_f)
                    if os.path.exists(model_f):
                        os.remove(model_f)
                    os.rename(valid_model_f, model_f)
                else:
                    os.remove(valid_model_f)

        # Save the final model
        torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)

        epoch_losses_df = pd.DataFrame(epoch_losses, columns=["Loss"])
        # epoch_losses_df.to_csv("epoch_losses.tsv", sep='\t', index=False)
        # print("Epoch losses have been saved to epoch_losses.tsv")

        return epoch_losses_df

    @staticmethod
    def prediction(sps, dismat, model_f, tcr_num, device, diversity_features_df):
        # 加载模型
        model = DeepLION2_GAT_div(tcr_num=tcr_num, feature_dim=768, attention_head_num=4, 
                              attention_hidden_size=200, drop_out=0.4, 
                              diversity_features_dim=diversity_features_df.shape[1], 
                              num_gat_layers=2
                              ).to(device)
        model.load_state_dict(torch.load(model_f, map_location=device))
        model.eval()

        # 准备测试数据
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 计算节点特征
        adjacencies = np.array(dismat)
        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        diversity_features_tensor = torch.Tensor(diversity_features_df.values).to(device)

        # forward
        with torch.no_grad():
            outputs = model(features_tensor, adjacencies_tensor, diversity_features_tensor)

        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 获取类别1的概率

        return probs

class DeepLION2_mulgat_div_fre(nn.Module):
    def __init__(self, tcr_num=300, feature_dim=768, attention_head_num=4, attention_hidden_size=200, num_gat_layers=2, drop_out=0.4, diversity_features_dim=10, frequency_dim=1):
        super(DeepLION2_mulgat_div_fre, self).__init__()
        self.tcr_num = tcr_num  # The number of TCR sequences per sample
        self.feature_dim = feature_dim  # Dimension of each TCR sequence feature (from TCRbert)
        self.diversity_features_dim = diversity_features_dim
        self.num_gat_layers = num_gat_layers
        self.frequency_dim = frequency_dim
        
        # Attention-related parameters
        self.attention_head_num = attention_head_num  # Number of attention heads
        self.attention_hidden_size = attention_hidden_size  # Hidden size for attention
        self.attention_head_size = int(attention_hidden_size / attention_head_num)  # 100 Size of each attention head
        
        # Graph Attention Layers (we will define these layers below)
        # GAT 可以运行
        # self.attentions = nn.ModuleList([
        #     GraphAttentionLayer(feature_dim, self.attention_head_size, dropout=drop_out)
        #     for _ in range(attention_head_num)
        # ])
        
        # GATv2 试一下
        # self.attentions = nn.ModuleList([
        #     GraphAttentionLayerV2(feature_dim, self.attention_head_size, dropout=drop_out)
        #     for _ in range(attention_head_num)
        # ])
        
        # 换成两层
        self.gat_layers = nn.ModuleList([
            nn.ModuleList([
                GraphAttentionLayerV2(feature_dim if i == 0 else self.attention_hidden_size, self.attention_head_size, dropout=drop_out)
                for _ in range(attention_head_num)
            ]) for i in range(num_gat_layers)
        ])

        # FCN for frequency features
        self.fc_freq1 = nn.Linear(frequency_dim, 128)
        self.fc_freq2 = nn.Linear(128, 128)

        # Output fully connected layer
        # self.out_fc = nn.Linear(tcr_num, 2)
        # 换成两层
        self.fc1 = nn.Linear(self.attention_head_size * attention_head_num + diversity_features_dim + 128, 128)
        self.fc2 = nn.Linear(128, 2)
        # self.out_fc = nn.Linear(self.attention_head_size * attention_head_num + diversity_features_dim, 2)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x, adj, diversity_features, frequency_features):
        # x: Node features, assumed to be [batch_size, tcr_num, feature_dim]
        # adj: Adjacency matrix, assumed to be [batch_size, tcr_num, tcr_num]
        # The values in adj are not binary but represent the weighted distance between nodes.

        # Node features are initially of size [batch_size, tcr_num, feature_dim]
        # print(f"Initial x shape: {x.shape}")  # Debug: check the shape of the input features # torch.Size([8, 100, 768])
        # print(f"Adjacency matrix shape: {adj.shape}")  # Debug: check the shape of the adjacency matrix # torch.Size([8, 100, 100])
 
        # Apply dropout to input features
        x = self.dropout(x)
        
        for gat_layer in self.gat_layers:
            x = torch.cat([att(x, adj) for att in gat_layer], dim=-1)

        # Apply the graph attention layers
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        # After concatenating the output of each attention head, x should have shape:
        # [batch_size, tcr_num, feature_dim * attention_head_num]
        # print(f"Shape after attention: {x.shape}")  # Debug: check the shape after attention

        # Mean pooling across the node dimension to aggregate node features into a single vector per sample
        x = torch.mean(x, dim=1)  # [batch_size, feature_dim * attention_head_num]
        
        # freq_features = F.relu(self.fc_freq1(frequency_features))
        # freq_features = F.relu(self.fc_freq2(freq_features))
        batch_size, tcr_num = frequency_features.shape[:2]

        # # Apply log transform to frequency features, adding a small constant to avoid log(0)
        log_frequency_features = torch.log(frequency_features + 1e-10)
        log_frequency_features = log_frequency_features.view(batch_size * tcr_num, -1)  # Reshape to [batch_size * tcr_num, frequency_dim]
        log_frequency_features = F.relu(self.fc_freq1(log_frequency_features))
        log_frequency_features = F.relu(self.fc_freq2(log_frequency_features))
        log_frequency_features = log_frequency_features.view(batch_size, tcr_num, -1)  # Reshape back to [batch_size, tcr_num, 128]
        log_frequency_features = torch.mean(log_frequency_features, dim=1)  # [batch_size, 128]

        x = torch.cat([x, diversity_features, log_frequency_features], dim=-1) 
        # print(f"x:{x.shape}")
        # Apply a final linear transformation
        # x = self.out_fc(x)  # [batch_size, 2] x:torch.Size([8, 2])
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Assuming want a single probability per sample, apply softmax and select one class's probability,
        # typically, this would be the probability of class 1 for binary classification
        return x
    
    @staticmethod
    def validate_model(model, valid_sps, valid_lbs, valid_mat, device):
        # Validation logic to compute the accuracy
        model.eval()
        # 准备数据
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(valid_sps) # 计算节点特征
        adjacencies = np.array(valid_mat)
        labels = np.array(valid_lbs)

        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        labels_tensor = torch.LongTensor(labels).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(features_tensor, adjacencies_tensor)

        # Get probabilities and predicted classes
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probabilities for class 1
        preds = (probs > 0.5).astype(int)  # Convert probabilities to 0 or 1 based on threshold

        # Calculate accuracy, AUC, and F1 score
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, preds)

        # print(f"Validation Accuracy: {acc:.4f}")
        # print(f"Validation AUC: {auc:.4f}")
        # print(f"Validation F1 Score: {f1:.4f}")

        return acc, auc, f1

    @staticmethod
    def training(sps, lbs, adjs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, diversity_features_df, shuffle=False, valid_sps=None, valid_lbs=None, valid_mat=None):
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 节点特征

        adjacencies = np.array(adjs)  # 所有样本的邻接矩阵
        lbs = np.array(lbs)  # 样本标签
        frequency_features = np.array([[float(tcr[2]) for tcr in sample] for sample in sps])

        features_tensor = torch.Tensor(features)
        adjacencies_tensor = torch.Tensor(adjacencies)
        labels_tensor = torch.LongTensor(lbs)
        diversity_features_tensor = torch.Tensor(diversity_features_df.values)
        frequency_features_tensor = torch.Tensor(frequency_features)

        # # debug
        # print(f"features_tensor size: {features_tensor.size()}")
        # print(f"adjacencies_tensor size: {adjacencies_tensor.size()}")
        # print(f"labels_tensor size: {labels_tensor.size()}")
        # print(f"diversity_features_tensor size: {diversity_features_tensor.size()}")

        # Initialize the model
        diversity_features_dim = diversity_features_df.shape[1]
        model = DeepLION2_mulgat_div_fre(tcr_num=tcr_num, feature_dim=768, drop_out=dropout, diversity_features_dim=diversity_features_dim, frequency_dim=1, num_gat_layers=2).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss().to(device)

        # Check if a model file already exists and adjust the filename
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        valid_model_f = model_f + "temp.pth"

        # Create a dataset and loader
        dataset = TensorDataset(features_tensor, adjacencies_tensor, labels_tensor, 
                                diversity_features_tensor, frequency_features_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=shuffle) # 这里一开始是100

        epoch_losses = []
        # Training loop
        for epoch in range(ep):
            total_loss = 0
            for batch_x, batch_adj, batch_y, batch_diversity, batch_frequency in loader:
                batch_x, batch_adj, batch_y, batch_diversity, batch_frequency = batch_x.to(device), batch_adj.to(device), batch_y.to(device), batch_diversity.to(device), batch_frequency.to(device)

                outputs = model(batch_x, batch_adj, batch_diversity, batch_frequency)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                # print(f"epoch {epoch}, loss = {loss}")

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_losses.append(total_loss)
            print(f"Epoch {epoch + 1}, Total Loss = {total_loss}")
            # Logging
            if (epoch + 1) % log_inr == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss.item()))

            # Optional validation step
            if valid_sps is not None and valid_lbs is not None and valid_mat is not None:
                valid_acc, valid_auc, valid_f1 = DeepLION2_GAT.validate_model(model, valid_sps, valid_lbs, valid_mat, device, diversity_features_df)
                print('Validation Accuracy:', valid_acc)
                print('Validation AUC:', valid_auc)
                print('Validation f1 score:', valid_f1)
                # Save the best model
                if valid_acc > max_acc:
                    max_acc = valid_acc
                    torch.save(model.state_dict(), valid_model_f)
                    if os.path.exists(model_f):
                        os.remove(model_f)
                    os.rename(valid_model_f, model_f)
                else:
                    os.remove(valid_model_f)

        # Save the final model
        torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)

        epoch_losses_df = pd.DataFrame(epoch_losses, columns=["Loss"])
        # epoch_losses_df.to_csv("epoch_losses.tsv", sep='\t', index=False)
        # print("Epoch losses have been saved to epoch_losses.tsv")

        return epoch_losses_df

    @staticmethod
    def prediction(sps, dismat, model_f, tcr_num, device, diversity_features_df):
        # 加载模型
        model = DeepLION2_mulgat_div_fre(tcr_num=tcr_num, feature_dim=768, attention_head_num=4, 
                              attention_hidden_size=200, drop_out=0.4, 
                              diversity_features_dim=diversity_features_df.shape[1], 
                              frequency_dim=1, 
                              num_gat_layers=2
                              ).to(device)
        model.load_state_dict(torch.load(model_f, map_location=device))
        model.eval()

        # 准备测试数据
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 计算节点特征
        adjacencies = np.array(dismat)
        frequency_features = np.array([[float(tcr[2]) for tcr in sample] for sample in sps])

        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        diversity_features_tensor = torch.Tensor(diversity_features_df.values).to(device)
        frequency_features_tensor = torch.Tensor(frequency_features).to(device)
        

        # forward
        with torch.no_grad():
            outputs = model(features_tensor, adjacencies_tensor, diversity_features_tensor, frequency_features_tensor)

        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 获取类别1的概率

        return probs

class DeepLION2_mulgat_div_fre_vgene(nn.Module):
    def __init__(self, tcr_num=300, feature_dim=768, attention_head_num=4, attention_hidden_size=200, num_gat_layers=2, drop_out=0.4, diversity_features_dim=10, frequency_dim=1, v_gene_vocab_size=100):
        super(DeepLION2_mulgat_div_fre_vgene, self).__init__()
        self.tcr_num = tcr_num  # The number of TCR sequences per sample
        self.feature_dim = feature_dim  # Dimension of each TCR sequence feature (from TCRbert)
        self.diversity_features_dim = diversity_features_dim
        self.num_gat_layers = num_gat_layers
        self.frequency_dim = frequency_dim

        # # V gene embedding
        self.v_gene_embedding_dim = 16  # Dimension of V gene embedding
        self.v_gene_vocab_size = v_gene_vocab_size  # Vocabulary size for V gene
        self.v_gene_embedding = nn.Embedding(self.v_gene_vocab_size, self.v_gene_embedding_dim)
        
        # Attention-related parameters
        self.attention_head_num = attention_head_num  # Number of attention heads
        self.attention_hidden_size = attention_hidden_size  # Hidden size for attention
        self.attention_head_size = int(attention_hidden_size / attention_head_num)  # 100 Size of each attention head
        
        # Graph Attention Layers (we will define these layers below)
        # GAT 可以运行
        # self.attentions = nn.ModuleList([
        #     GraphAttentionLayer(feature_dim, self.attention_head_size, dropout=drop_out)
        #     for _ in range(attention_head_num)
        # ])
        
        # GATv2 试一下
        # self.attentions = nn.ModuleList([
        #     GraphAttentionLayerV2(feature_dim, self.attention_head_size, dropout=drop_out)
        #     for _ in range(attention_head_num)
        # ])
        
        # 换成两层
        self.gat_layers = nn.ModuleList([
            nn.ModuleList([
                GraphAttentionLayerV2(feature_dim if i == 0 else self.attention_hidden_size, self.attention_head_size, dropout=drop_out)
                for _ in range(attention_head_num)
            ]) for i in range(num_gat_layers)
        ])

        # FCN for frequency features
        self.fc_freq1 = nn.Linear(frequency_dim, 128)
        self.fc_freq2 = nn.Linear(128, 128)

        # FCN for V gene features
        self.fc_vgene1 = nn.Linear(self.v_gene_embedding_dim, 128)
        self.fc_vgene2 = nn.Linear(128, 128)

        # Output fully connected layer
        # self.out_fc = nn.Linear(tcr_num, 2)
        # 换成两层
        self.fc1 = nn.Linear(self.attention_head_size * attention_head_num + diversity_features_dim + 128 + 128, 128)
        self.fc2 = nn.Linear(128, 2)
        # self.out_fc = nn.Linear(self.attention_head_size * attention_head_num + diversity_features_dim, 2)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x, adj, diversity_features, frequency_features, v_gene_features):
        # x: Node features, assumed to be [batch_size, tcr_num, feature_dim]
        # adj: Adjacency matrix, assumed to be [batch_size, tcr_num, tcr_num]
        # The values in adj are not binary but represent the weighted distance between nodes.

        # Node features are initially of size [batch_size, tcr_num, feature_dim]
        # print(f"Initial x shape: {x.shape}")  # Debug: check the shape of the input features # torch.Size([8, 100, 768])
        # print(f"Adjacency matrix shape: {adj.shape}")  # Debug: check the shape of the adjacency matrix # torch.Size([8, 100, 100])
 
        # Apply dropout to input features
        x = self.dropout(x)
        
        for gat_layer in self.gat_layers:
            x = torch.cat([att(x, adj) for att in gat_layer], dim=-1)

        # Apply the graph attention layers
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        # After concatenating the output of each attention head, x should have shape:
        # [batch_size, tcr_num, feature_dim * attention_head_num]
        # print(f"Shape after attention: {x.shape}")  # Debug: check the shape after attention

        # Mean pooling across the node dimension to aggregate node features into a single vector per sample
        x = torch.mean(x, dim=1)  # [batch_size, feature_dim * attention_head_num]
        
        # freq_features = F.relu(self.fc_freq1(frequency_features))
        # freq_features = F.relu(self.fc_freq2(freq_features))
        batch_size, tcr_num = frequency_features.shape[:2]

        # # Apply log transform to frequency features, adding a small constant to avoid log(0)
        log_frequency_features = torch.log(frequency_features + 1e-10)
        log_frequency_features = log_frequency_features.view(batch_size * tcr_num, -1)  # Reshape to [batch_size * tcr_num, frequency_dim]
        log_frequency_features = F.relu(self.fc_freq1(log_frequency_features))
        log_frequency_features = F.relu(self.fc_freq2(log_frequency_features))
        log_frequency_features = log_frequency_features.view(batch_size, tcr_num, -1)  # Reshape back to [batch_size, tcr_num, 128]
        log_frequency_features = torch.mean(log_frequency_features, dim=1)  # [batch_size, 128]

        # Process V gene features through embedding and FCN
        v_gene_features = self.v_gene_embedding(v_gene_features)  # [batch_size, tcr_num, v_gene_embedding_dim]
        v_gene_features = v_gene_features.view(batch_size * tcr_num, -1)  # Reshape to [batch_size * tcr_num, v_gene_embedding_dim]
        v_gene_features = F.relu(self.fc_vgene1(v_gene_features))
        v_gene_features = F.relu(self.fc_vgene2(v_gene_features))
        v_gene_features = v_gene_features.view(batch_size, tcr_num, -1)  # Reshape back to [batch_size, tcr_num, 128]
        v_gene_features = torch.mean(v_gene_features, dim=1)  # [batch_size, 128]

        x = torch.cat([x, diversity_features, log_frequency_features, v_gene_features], dim=-1) 
        # print(f"x:{x.shape}")
        # Apply a final linear transformation
        # x = self.out_fc(x)  # [batch_size, 2] x:torch.Size([8, 2])
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Assuming want a single probability per sample, apply softmax and select one class's probability,
        # typically, this would be the probability of class 1 for binary classification
        return x
    
    # @staticmethod
    def validate_model(model, valid_sps, valid_lbs, valid_mat, device):
        # Validation logic to compute the accuracy
        model.eval()
        # 准备数据
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(valid_sps) # 计算节点特征
        adjacencies = np.array(valid_mat)
        labels = np.array(valid_lbs)

        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        labels_tensor = torch.LongTensor(labels).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(features_tensor, adjacencies_tensor)

        # Get probabilities and predicted classes
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probabilities for class 1
        preds = (probs > 0.5).astype(int)  # Convert probabilities to 0 or 1 based on threshold

        # Calculate accuracy, AUC, and F1 score
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, preds)

        # print(f"Validation Accuracy: {acc:.4f}")
        # print(f"Validation AUC: {auc:.4f}")
        # print(f"Validation F1 Score: {f1:.4f}")

        return acc, auc, f1

    @staticmethod
    def v_gene_to_numeric(v_gene, v_gene_dict):
        return v_gene_dict.get(v_gene, v_gene_dict['<UNK>'])
    
    def training(sps, lbs, adjs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, diversity_features_df, v_gene_dict, shuffle=False, valid_sps=None, valid_lbs=None, valid_mat=None):
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 节点特征

        adjacencies = np.array(adjs)  # 所有样本的邻接矩阵
        lbs = np.array(lbs)  # 样本标签

        frequency_features = np.array([[float(tcr[2]) for tcr in sample] for sample in sps])
        # v_gene_features = np.array([[v_gene_dict[tcr[1]] for tcr in sample] for sample in sps])
        v_gene_features = np.array([[DeepLION2_mulgat_div_fre_vgene.v_gene_to_numeric(tcr[1], v_gene_dict) for tcr in sample] for sample in sps])

        features_tensor = torch.Tensor(features)
        adjacencies_tensor = torch.Tensor(adjacencies)
        labels_tensor = torch.LongTensor(lbs)
        diversity_features_tensor = torch.Tensor(diversity_features_df.values)
        frequency_features_tensor = torch.Tensor(frequency_features)
        v_gene_features_tensor = torch.LongTensor(v_gene_features)

        # # debug
        # print(f"features_tensor size: {features_tensor.size()}")
        # print(f"adjacencies_tensor size: {adjacencies_tensor.size()}")
        # print(f"labels_tensor size: {labels_tensor.size()}")
        # print(f"diversity_features_tensor size: {diversity_features_tensor.size()}")

        # Initialize the model
        diversity_features_dim = diversity_features_df.shape[1]
        model = DeepLION2_mulgat_div_fre_vgene(tcr_num=tcr_num, feature_dim=768, drop_out=dropout, diversity_features_dim=diversity_features_dim, 
                                               frequency_dim=1, v_gene_vocab_size=len(v_gene_dict), num_gat_layers=2).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss().to(device)

        # Check if a model file already exists and adjust the filename
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        valid_model_f = model_f + "temp.pth"

        # # Print shapes for debugging
        print(f"features_tensor shape: {features_tensor.shape}")
        print(f"adjacencies_tensor shape: {adjacencies_tensor.shape}")
        print(f"labels_tensor shape: {labels_tensor.shape}")
        print(f"diversity_features_tensor shape: {diversity_features_tensor.shape}")
        print(f"frequency_features_tensor shape: {frequency_features_tensor.shape}")
        print(f"v_gene_features_tensor shape: {v_gene_features_tensor.shape}")

        # Create a dataset and loader
        dataset = TensorDataset(features_tensor, adjacencies_tensor, labels_tensor, diversity_features_tensor, frequency_features_tensor, v_gene_features_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=shuffle) # 这里一开始是100

        epoch_losses = []
        # Training loop
        for epoch in range(ep):
            total_loss = 0
            for batch_x, batch_adj, batch_y, batch_diversity, batch_frequency, batch_vgene in loader:
                batch_x, batch_adj, batch_y, batch_diversity, batch_frequency, batch_vgene = batch_x.to(device), batch_adj.to(device), batch_y.to(device), batch_diversity.to(device), batch_frequency.to(device), batch_vgene.to(device)

                outputs = model(batch_x, batch_adj, batch_diversity, batch_frequency, batch_vgene)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                # print(f"epoch {epoch}, loss = {loss}")

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_losses.append(total_loss)
            print(f"Epoch {epoch + 1}, Total Loss = {total_loss}")
            # Logging
            if (epoch + 1) % log_inr == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss.item()))

            # Optional validation step
            if valid_sps is not None and valid_lbs is not None and valid_mat is not None:
                valid_acc, valid_auc, valid_f1 = DeepLION2_GAT.validate_model(model, valid_sps, valid_lbs, valid_mat, device, diversity_features_df)
                print('Validation Accuracy:', valid_acc)
                print('Validation AUC:', valid_auc)
                print('Validation f1 score:', valid_f1)
                # Save the best model
                if valid_acc > max_acc:
                    max_acc = valid_acc
                    torch.save(model.state_dict(), valid_model_f)
                    if os.path.exists(model_f):
                        os.remove(model_f)
                    os.rename(valid_model_f, model_f)
                else:
                    os.remove(valid_model_f)

        # Save the final model
        torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)

        epoch_losses_df = pd.DataFrame(epoch_losses, columns=["Loss"])
        # epoch_losses_df.to_csv("epoch_losses.tsv", sep='\t', index=False)
        # print("Epoch losses have been saved to epoch_losses.tsv")

        return epoch_losses_df

    # @staticmethod
    def prediction(sps, dismat, model_f, tcr_num, device, diversity_features_df, v_gene_dict):
        # 加载模型
        model = DeepLION2_mulgat_div_fre_vgene(tcr_num=tcr_num, feature_dim=768, attention_head_num=4, 
                              attention_hidden_size=200, drop_out=0.4, 
                              diversity_features_dim=diversity_features_df.shape[1], 
                              frequency_dim=1, 
                              v_gene_vocab_size=len(v_gene_dict), 
                              num_gat_layers=2
                              ).to(device)
        model.load_state_dict(torch.load(model_f, map_location=device))
        model.eval()

        # 准备测试数据
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 计算节点特征
        adjacencies = np.array(dismat)

        frequency_features = np.array([[float(tcr[2]) for tcr in sample] for sample in sps])
        # v_gene_features = np.array([[v_gene_dict[tcr[1]] for tcr in sample] for sample in sps])  # Convert V gene to numeric features
        v_gene_features = np.array([[DeepLION2_mulgat_div_fre_vgene.v_gene_to_numeric(tcr[1], v_gene_dict) for tcr in sample] for sample in sps])  # Convert V gene to numeric features

        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        diversity_features_tensor = torch.Tensor(diversity_features_df.values).to(device)
        frequency_features_tensor = torch.Tensor(frequency_features).to(device)
        v_gene_features_tensor = torch.LongTensor(v_gene_features).to(device)  # Use LongTensor for embedding layer

        # forward
        with torch.no_grad():
            outputs = model(features_tensor, adjacencies_tensor, diversity_features_tensor, frequency_features_tensor, v_gene_features_tensor)

        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 获取类别1的概率

        return probs

class DeepLION2_mulgat_fre(nn.Module):
    def __init__(self, tcr_num=300, feature_dim=768, attention_head_num=4, attention_hidden_size=200, num_gat_layers=2, drop_out=0.4, frequency_dim=1):
        super(DeepLION2_mulgat_fre, self).__init__()
        self.tcr_num = tcr_num  # The number of TCR sequences per sample
        self.feature_dim = feature_dim  # Dimension of each TCR sequence feature (from TCRbert)
        self.num_gat_layers = num_gat_layers
        self.frequency_dim = frequency_dim
        
        # Attention-related parameters
        self.attention_head_num = attention_head_num  # Number of attention heads
        self.attention_hidden_size = attention_hidden_size  # Hidden size for attention
        self.attention_head_size = int(attention_hidden_size / attention_head_num)  # 100 Size of each attention head
        
        # 换成两层
        self.gat_layers = nn.ModuleList([
            nn.ModuleList([
                GraphAttentionLayerV2(feature_dim if i == 0 else self.attention_hidden_size, self.attention_head_size, dropout=drop_out)
                for _ in range(attention_head_num)
            ]) for i in range(num_gat_layers)
        ])

        # FCN for frequency features
        self.fc_freq1 = nn.Linear(frequency_dim, 128)
        self.fc_freq2 = nn.Linear(128, 128)

        # Output fully connected layer
        self.fc1 = nn.Linear(self.attention_head_size * attention_head_num + 128, 128)
        self.fc2 = nn.Linear(128, 2)
        # self.out_fc = nn.Linear(self.attention_head_size * attention_head_num + diversity_features_dim, 2)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x, adj, frequency_features):

        x = self.dropout(x)
        
        for gat_layer in self.gat_layers:
            x = torch.cat([att(x, adj) for att in gat_layer], dim=-1)

        x = torch.mean(x, dim=1)  # [batch_size, feature_dim * attention_head_num]
        
        # freq_features = F.relu(self.fc_freq1(frequency_features))
        # freq_features = F.relu(self.fc_freq2(freq_features))
        batch_size, tcr_num = frequency_features.shape[:2]

        # # Apply log transform to frequency features, adding a small constant to avoid log(0)
        log_frequency_features = torch.log(frequency_features + 1e-10)
        log_frequency_features = log_frequency_features.view(batch_size * tcr_num, -1)  # Reshape to [batch_size * tcr_num, frequency_dim]
        log_frequency_features = F.relu(self.fc_freq1(log_frequency_features))
        log_frequency_features = F.relu(self.fc_freq2(log_frequency_features))
        log_frequency_features = log_frequency_features.view(batch_size, tcr_num, -1)  # Reshape back to [batch_size, tcr_num, 128]
        log_frequency_features = torch.mean(log_frequency_features, dim=1)  # [batch_size, 128]

        x = torch.cat([x, log_frequency_features], dim=-1) 
        # print(f"x:{x.shape}")
        # Apply a final linear transformation
        # x = self.out_fc(x)  # [batch_size, 2] x:torch.Size([8, 2])
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Assuming want a single probability per sample, apply softmax and select one class's probability,
        # typically, this would be the probability of class 1 for binary classification
        return x
    
    @staticmethod
    def validate_model(model, valid_sps, valid_lbs, valid_mat, device):
        # 现在用不上
        # Validation logic to compute the accuracy
        model.eval()
        # 准备数据
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(valid_sps) # 计算节点特征
        adjacencies = np.array(valid_mat)
        labels = np.array(valid_lbs)

        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        labels_tensor = torch.LongTensor(labels).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(features_tensor, adjacencies_tensor)

        # Get probabilities and predicted classes
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probabilities for class 1
        preds = (probs > 0.5).astype(int)  # Convert probabilities to 0 or 1 based on threshold

        # Calculate accuracy, AUC, and F1 score
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, preds)

        # print(f"Validation Accuracy: {acc:.4f}")
        # print(f"Validation AUC: {auc:.4f}")
        # print(f"Validation F1 Score: {f1:.4f}")

        return acc, auc, f1

    @staticmethod
    def training(sps, lbs, adjs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, shuffle=False, valid_sps=None, valid_lbs=None, valid_mat=None):
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 节点特征

        adjacencies = np.array(adjs)  # 所有样本的邻接矩阵
        lbs = np.array(lbs)  # 样本标签
        frequency_features = np.array([[float(tcr[2]) for tcr in sample] for sample in sps])

        features_tensor = torch.Tensor(features)
        adjacencies_tensor = torch.Tensor(adjacencies)
        labels_tensor = torch.LongTensor(lbs)
        frequency_features_tensor = torch.Tensor(frequency_features)

        # # debug
        # print(f"features_tensor size: {features_tensor.size()}")
        # print(f"adjacencies_tensor size: {adjacencies_tensor.size()}")
        # print(f"labels_tensor size: {labels_tensor.size()}")
        # print(f"diversity_features_tensor size: {diversity_features_tensor.size()}")

        # Initialize the model
        model = DeepLION2_mulgat_fre(tcr_num=tcr_num, feature_dim=768, drop_out=dropout, frequency_dim=1, num_gat_layers=2).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss().to(device)

        # Check if a model file already exists and adjust the filename
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        valid_model_f = model_f + "temp.pth"

        # Create a dataset and loader
        dataset = TensorDataset(features_tensor, adjacencies_tensor, labels_tensor, 
                                frequency_features_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=shuffle) # 这里一开始是100

        epoch_losses = []
        # Training loop
        for epoch in range(ep):
            total_loss = 0
            for batch_x, batch_adj, batch_y, batch_frequency in loader:
                batch_x, batch_adj, batch_y, batch_frequency = batch_x.to(device), batch_adj.to(device), batch_y.to(device), batch_frequency.to(device)

                outputs = model(batch_x, batch_adj, batch_frequency)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                # print(f"epoch {epoch}, loss = {loss}")

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_losses.append(total_loss)
            print(f"Epoch {epoch + 1}, Total Loss = {total_loss}")
            # Logging
            if (epoch + 1) % log_inr == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss.item()))

            # Optional validation step
            # 先不改
            if valid_sps is not None and valid_lbs is not None and valid_mat is not None:
                valid_acc, valid_auc, valid_f1 = DeepLION2_GAT.validate_model(model, valid_sps, valid_lbs, valid_mat, device, diversity_features_df)
                print('Validation Accuracy:', valid_acc)
                print('Validation AUC:', valid_auc)
                print('Validation f1 score:', valid_f1)
                # Save the best model
                if valid_acc > max_acc:
                    max_acc = valid_acc
                    torch.save(model.state_dict(), valid_model_f)
                    if os.path.exists(model_f):
                        os.remove(model_f)
                    os.rename(valid_model_f, model_f)
                else:
                    os.remove(valid_model_f)

        # Save the final model
        torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)

        epoch_losses_df = pd.DataFrame(epoch_losses, columns=["Loss"])
        # epoch_losses_df.to_csv("epoch_losses.tsv", sep='\t', index=False)
        # print("Epoch losses have been saved to epoch_losses.tsv")

        return epoch_losses_df

    @staticmethod
    def prediction(sps, dismat, model_f, tcr_num, device):
        # 加载模型
        model = DeepLION2_mulgat_fre(tcr_num=tcr_num, feature_dim=768, attention_head_num=4, 
                              attention_hidden_size=200, drop_out=0.4, 
                              frequency_dim=1, 
                              num_gat_layers=2
                              ).to(device)
        model.load_state_dict(torch.load(model_f, map_location=device))
        model.eval()

        # 准备测试数据
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 计算节点特征
        adjacencies = np.array(dismat)
        frequency_features = np.array([[float(tcr[2]) for tcr in sample] for sample in sps])

        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        frequency_features_tensor = torch.Tensor(frequency_features).to(device)
        

        # forward
        with torch.no_grad():
            outputs = model(features_tensor, adjacencies_tensor, frequency_features_tensor)

        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 获取类别1的概率

        return probs

class DeepLION2_mulgat_fre_vgene(nn.Module):
    def __init__(self, tcr_num=300, feature_dim=768, attention_head_num=4, attention_hidden_size=256, num_gat_layers=2, drop_out=0.4, frequency_dim=1, v_gene_vocab_size=100):
        super(DeepLION2_mulgat_fre_vgene, self).__init__()
        self.tcr_num = tcr_num  # The number of TCR sequences per sample
        self.feature_dim = feature_dim  # Dimension of each TCR sequence feature (from TCRbert)
        self.num_gat_layers = num_gat_layers
        self.frequency_dim = frequency_dim

        # # V gene embedding
        self.v_gene_embedding_dim = 16  # Dimension of V gene embedding
        self.v_gene_vocab_size = v_gene_vocab_size  # Vocabulary size for V gene
        self.v_gene_embedding = nn.Embedding(self.v_gene_vocab_size, self.v_gene_embedding_dim)
        
        # Attention-related parameters
        self.attention_head_num = attention_head_num  # Number of attention heads
        self.attention_hidden_size = attention_hidden_size  # Hidden size for attention
        self.attention_head_size = int(attention_hidden_size / attention_head_num)  # 100 Size of each attention head
        
        self.gat_layers = nn.ModuleList([
            nn.ModuleList([
                GraphAttentionLayerV2(feature_dim if i == 0 else self.attention_hidden_size, self.attention_head_size, dropout=drop_out)
                for _ in range(attention_head_num)
            ]) for i in range(num_gat_layers)
        ])

        # FCN for frequency features
        self.fc_freq1 = nn.Linear(frequency_dim, 128)
        self.fc_freq2 = nn.Linear(128, 128)

        # FCN for V gene features
        self.fc_vgene1 = nn.Linear(self.v_gene_embedding_dim, 128)
        self.fc_vgene2 = nn.Linear(128, 128)

        # Output fully connected layer
        # 换成两层
        self.fc1 = nn.Linear(self.attention_head_size * attention_head_num + 128 + 128, 128)
        self.fc2 = nn.Linear(128, 2)

        self.dropout = nn.Dropout(drop_out)

    def forward(self, x, adj, frequency_features, v_gene_features):
        # x: Node features, assumed to be [batch_size, tcr_num, feature_dim]
        # adj: Adjacency matrix, assumed to be [batch_size, tcr_num, tcr_num]
        # The values in adj are not binary but represent the weighted distance between nodes.

        # Node features are initially of size [batch_size, tcr_num, feature_dim]
        # print(f"Initial x shape: {x.shape}")  # Debug: check the shape of the input features # torch.Size([8, 100, 768])
        # print(f"Adjacency matrix shape: {adj.shape}")  # Debug: check the shape of the adjacency matrix # torch.Size([8, 100, 100])
 
        # Apply dropout to input features
        x = self.dropout(x)
        
        for gat_layer in self.gat_layers:
            x = torch.cat([att(x, adj) for att in gat_layer], dim=-1)

        # Mean pooling across the node dimension to aggregate node features into a single vector per sample
        # x = torch.mean(x, dim=1)  # [batch_size, feature_dim * attention_head_num]
        x, _ = torch.max(x, dim=1)
        
        # freq_features = F.relu(self.fc_freq1(frequency_features))
        # freq_features = F.relu(self.fc_freq2(freq_features))
        batch_size, tcr_num = frequency_features.shape[:2]

        # # Apply log transform to frequency features, adding a small constant to avoid log(0)
        log_frequency_features = torch.log(frequency_features + 1e-10)
        log_frequency_features = log_frequency_features.view(batch_size * tcr_num, -1)  # Reshape to [batch_size * tcr_num, frequency_dim]
        log_frequency_features = F.relu(self.fc_freq1(log_frequency_features))
        log_frequency_features = F.relu(self.fc_freq2(log_frequency_features))
        log_frequency_features = log_frequency_features.view(batch_size, tcr_num, -1)  # Reshape back to [batch_size, tcr_num, 128]
        # log_frequency_features = torch.mean(log_frequency_features, dim=1)  # [batch_size, 128]
        log_frequency_features, _ = torch.max(log_frequency_features, dim=1)

        # Process V gene features through embedding and FCN
        v_gene_features = self.v_gene_embedding(v_gene_features)  # [batch_size, tcr_num, v_gene_embedding_dim]
        v_gene_features = v_gene_features.view(batch_size * tcr_num, -1)  # Reshape to [batch_size * tcr_num, v_gene_embedding_dim]
        v_gene_features = F.relu(self.fc_vgene1(v_gene_features))
        v_gene_features = F.relu(self.fc_vgene2(v_gene_features))
        v_gene_features = v_gene_features.view(batch_size, tcr_num, -1)  # Reshape back to [batch_size, tcr_num, 128]
        # v_gene_features = torch.mean(v_gene_features, dim=1)  # [batch_size, 128]
        v_gene_features, _ = torch.max(v_gene_features, dim=1)

        x = torch.cat([x, log_frequency_features, v_gene_features], dim=-1) 
        # print(f"x:{x.shape}")
        # Apply a final linear transformation
        # x = self.out_fc(x)  # [batch_size, 2] x:torch.Size([8, 2])
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Assuming want a single probability per sample, apply softmax and select one class's probability,
        # typically, this would be the probability of class 1 for binary classification
        return x
    
    # @staticmethod
    def validate_model(model, valid_sps, valid_lbs, valid_mat, device):
        # Validation logic to compute the accuracy
        model.eval()
        # 准备数据
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(valid_sps) # 计算节点特征
        adjacencies = np.array(valid_mat)
        labels = np.array(valid_lbs)

        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        labels_tensor = torch.LongTensor(labels).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(features_tensor, adjacencies_tensor)

        # Get probabilities and predicted classes
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probabilities for class 1
        preds = (probs > 0.5).astype(int)  # Convert probabilities to 0 or 1 based on threshold

        # Calculate accuracy, AUC, and F1 score
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, preds)

        # print(f"Validation Accuracy: {acc:.4f}")
        # print(f"Validation AUC: {auc:.4f}")
        # print(f"Validation F1 Score: {f1:.4f}")

        return acc, auc, f1

    @staticmethod
    def v_gene_to_numeric(v_gene, v_gene_dict):
        return v_gene_dict.get(v_gene, v_gene_dict['<UNK>'])
    
    def training(sps, lbs, adjs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, v_gene_dict, shuffle=False, valid_sps=None, valid_lbs=None, valid_mat=None):
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 节点特征

        adjacencies = np.array(adjs)  # 所有样本的邻接矩阵
        lbs = np.array(lbs)  # 样本标签

        frequency_features = np.array([[float(tcr[2]) for tcr in sample] for sample in sps])
        # v_gene_features = np.array([[v_gene_dict[tcr[1]] for tcr in sample] for sample in sps])
        v_gene_features = np.array([[DeepLION2_mulgat_div_fre_vgene.v_gene_to_numeric(tcr[1], v_gene_dict) for tcr in sample] for sample in sps])

        features_tensor = torch.Tensor(features)
        adjacencies_tensor = torch.Tensor(adjacencies)
        labels_tensor = torch.LongTensor(lbs)
        frequency_features_tensor = torch.Tensor(frequency_features)
        v_gene_features_tensor = torch.LongTensor(v_gene_features)

        # # debug
        # print(f"features_tensor size: {features_tensor.size()}")
        # print(f"adjacencies_tensor size: {adjacencies_tensor.size()}")
        # print(f"labels_tensor size: {labels_tensor.size()}")
        # print(f"diversity_features_tensor size: {diversity_features_tensor.size()}")

        # Initialize the model
        model = DeepLION2_mulgat_fre_vgene(tcr_num=tcr_num, feature_dim=768, drop_out=dropout,  
                                               frequency_dim=1, v_gene_vocab_size=len(v_gene_dict), num_gat_layers=2).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss().to(device)

        # Check if a model file already exists and adjust the filename
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        valid_model_f = model_f + "temp.pth"

        # Print shapes for debugging
        print(f"features_tensor shape: {features_tensor.shape}")
        print(f"adjacencies_tensor shape: {adjacencies_tensor.shape}")
        print(f"labels_tensor shape: {labels_tensor.shape}")
        print(f"frequency_features_tensor shape: {frequency_features_tensor.shape}")
        print(f"v_gene_features_tensor shape: {v_gene_features_tensor.shape}")

        # Create a dataset and loader
        dataset = TensorDataset(features_tensor, adjacencies_tensor, labels_tensor, frequency_features_tensor, v_gene_features_tensor)
        loader = DataLoader(dataset, batch_size=48, shuffle=shuffle) # 这里一开始是100

        epoch_losses = []
        # Training loop
        for epoch in range(ep):
            total_loss = 0
            for batch_x, batch_adj, batch_y, batch_frequency, batch_vgene in loader:
                batch_x, batch_adj, batch_y, batch_frequency, batch_vgene = batch_x.to(device), batch_adj.to(device), batch_y.to(device), batch_frequency.to(device), batch_vgene.to(device)

                outputs = model(batch_x, batch_adj, batch_frequency, batch_vgene)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                # print(f"epoch {epoch}, loss = {loss}")

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_losses.append(total_loss)
            print(f"Epoch {epoch + 1}, Total Loss = {total_loss}")
            # Logging
            if (epoch + 1) % log_inr == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss.item()))

            # Optional validation step
            if valid_sps is not None and valid_lbs is not None and valid_mat is not None:
                valid_acc, valid_auc, valid_f1 = DeepLION2_GAT.validate_model(model, valid_sps, valid_lbs, valid_mat, device, diversity_features_df)
                print('Validation Accuracy:', valid_acc)
                print('Validation AUC:', valid_auc)
                print('Validation f1 score:', valid_f1)
                # Save the best model
                if valid_acc > max_acc:
                    max_acc = valid_acc
                    torch.save(model.state_dict(), valid_model_f)
                    if os.path.exists(model_f):
                        os.remove(model_f)
                    os.rename(valid_model_f, model_f)
                else:
                    os.remove(valid_model_f)

        # Save the final model
        torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)

        epoch_losses_df = pd.DataFrame(epoch_losses, columns=["Loss"])
        # epoch_losses_df.to_csv("epoch_losses.tsv", sep='\t', index=False)
        # print("Epoch losses have been saved to epoch_losses.tsv")

        return epoch_losses_df

    # @staticmethod
    def prediction(sps, dismat, model_f, tcr_num, device, v_gene_dict):
        # 加载模型
        model = DeepLION2_mulgat_fre_vgene(tcr_num=tcr_num, feature_dim=768, attention_head_num=4, 
                              attention_hidden_size=256, drop_out=0.4, 
                              frequency_dim=1, 
                              v_gene_vocab_size=len(v_gene_dict), 
                              num_gat_layers=2
                              ).to(device)
        model.load_state_dict(torch.load(model_f, map_location=device))
        model.eval()

        # 准备测试数据
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 计算节点特征
        adjacencies = np.array(dismat)

        frequency_features = np.array([[float(tcr[2]) for tcr in sample] for sample in sps])
        # v_gene_features = np.array([[v_gene_dict[tcr[1]] for tcr in sample] for sample in sps])  # Convert V gene to numeric features
        v_gene_features = np.array([[DeepLION2_mulgat_div_fre_vgene.v_gene_to_numeric(tcr[1], v_gene_dict) for tcr in sample] for sample in sps])  # Convert V gene to numeric features

        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        frequency_features_tensor = torch.Tensor(frequency_features).to(device)
        v_gene_features_tensor = torch.LongTensor(v_gene_features).to(device)  # Use LongTensor for embedding layer

        # forward
        with torch.no_grad():
            outputs = model(features_tensor, adjacencies_tensor, frequency_features_tensor, v_gene_features_tensor)

        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 获取类别1的概率

        return probs
    
from umap import UMAP
# 两个模态融合
class Mulgat_vgene_fusion_freq(nn.Module): 
    def __init__(self, tcr_num=300, feature_dim=768, attention_head_num=4, attention_hidden_size=128, num_gat_layers=2, drop_out=0.4, frequency_dim=1, v_gene_vocab_size=100):
        super(Mulgat_vgene_fusion_freq, self).__init__()
        self.tcr_num = tcr_num
        self.feature_dim = feature_dim
        self.num_gat_layers = num_gat_layers
        self.frequency_dim = frequency_dim
        
        # V gene embedding
        self.v_gene_embedding_dim = 16
        self.v_gene_vocab_size = v_gene_vocab_size
        self.v_gene_embedding = nn.Embedding(self.v_gene_vocab_size, self.v_gene_embedding_dim)
        
        # Attention-related parameters
        self.attention_head_num = attention_head_num
        self.attention_hidden_size = attention_hidden_size
        self.attention_head_size = int(attention_hidden_size / attention_head_num)
        
        self.gat_layers = nn.ModuleList([
            nn.ModuleList([
                GraphAttentionLayerV2(feature_dim if i == 0 else self.attention_hidden_size, self.attention_head_size, dropout=drop_out)
                for _ in range(attention_head_num)
            ]) for i in range(num_gat_layers)
        ])

        # FCN for frequency features
        # 1→256→128
        self.fc_freq1 = nn.Linear(frequency_dim, attention_hidden_size * 2)
        self.fc_freq2 = nn.Linear(attention_hidden_size * 2, 128)

        # FCN for V gene features
        self.fc_vgene1 = nn.Linear(self.v_gene_embedding_dim, attention_hidden_size * 2)
        self.fc_vgene2 = nn.Linear(attention_hidden_size * 2, 128)
        self.vgene_dropout = nn.AlphaDropout(p=0.2)

        # Gating mechanism for feature fusion (TCR and V gene only)
        self.linear_h1 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU())
        self.linear_z1 = nn.Bilinear(attention_hidden_size, attention_hidden_size, attention_hidden_size)
        self.linear_o1 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU(), nn.Dropout(p=drop_out))

        self.linear_h3 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU())
        self.linear_z3 = nn.Bilinear(attention_hidden_size, attention_hidden_size, attention_hidden_size)
        self.linear_o3 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU(), nn.Dropout(p=drop_out))

        self.post_fusion_dropout = nn.Dropout(p=drop_out)
        fused_dim = (attention_hidden_size + 1) ** 2  # 根据融合逻辑计算的维度
        final_dim = fused_dim + 128  # 融合后的维度加上频率特征的维度
        self.pre_encoder = nn.Linear(final_dim, 768)  # 确保输入维度匹配
        self.encoder1 = nn.Sequential(nn.Linear(768, attention_hidden_size), nn.LeakyReLU(), nn.Dropout(p=drop_out))
        # self.encoder2 = nn.Sequential(nn.Linear(attention_hidden_size, 2), nn.LeakyReLU(), nn.Dropout(p=drop_out))
        self.encoder2 = nn.Sequential(nn.Linear(attention_hidden_size, 2))

        # self.encoder1 = nn.Sequential(nn.Linear(2, attention_hidden_size), nn.LeakyReLU(), nn.Dropout(p=drop_out))
        # self.encoder2 = nn.Sequential(nn.Linear(attention_hidden_size, 2))

        self.dropout = nn.Dropout(p=drop_out)
        self.softmax = nn.Softmax(dim=1)  # 添加 softmax 激活函数

        # # 添加UMAP降维器
        # self.umap = UMAP(n_components=2, random_state=42)

    # 原始代码
    # def forward(self, x, adj, frequency_features, v_gene_features, caTCR_tag=False):

    #     # x: Node features, assumed to be [batch_size, tcr_num, feature_dim]
    #     # adj: Adjacency matrix, assumed to be [batch_size, tcr_num, tcr_num]
    #     # print(f"x_raw: {x.shape}") # torch.Size([64, 300, 768])
    #     x = self.dropout(x) 
        
    #     for gat_layer in self.gat_layers:
    #         x = torch.cat([att(x, adj) for att in gat_layer], dim=-1)
    #     # print(f"x_before_max: {x.shape}") # torch.Size([64, 300, 128])
    #     if not caTCR_tag:
    #         x_max, _ = torch.max(x, dim=1)
    #         # print(f"x_after_max: {x.shape}") # torch.Size([64, 128])
    #     else:
    #         # 最大值和对应的节点索引
    #         x_max, max_indices = torch.max(x, dim=1)
    #         # 初始化节点贡献计数
    #         batch_size, feature_dim = max_indices.shape
    #         node_counts = torch.zeros(batch_size, x.size(1))  # [batch_size, tcr_num]
    #         # 统计每个节点的贡献
    #         for i in range(batch_size):
    #             for j in range(feature_dim):
    #                 node_counts[i, max_indices[i, j]] += 1

    #         # 找出前 5 个贡献最大的节点
    #         top_k = 5
    #         top_values, top_indices = torch.topk(node_counts, k=top_k, dim=1)
    #         # print("贡献最大的节点索引:", top_indices)
    #         # print("贡献值:", top_values)
    #         # 下面可以根据这些索引找到对应节点的特征

    #     # Frequency features
    #     batch_size, tcr_num = frequency_features.shape[:2]
    #     log_frequency_features = torch.log(frequency_features + 1e-10)
    #     log_frequency_features = log_frequency_features.view(batch_size * tcr_num, -1)
    #     log_frequency_features = F.leaky_relu(self.fc_freq1(log_frequency_features))
    #     log_frequency_features = F.leaky_relu(self.fc_freq2(log_frequency_features)) 
    #     log_frequency_features = log_frequency_features.view(batch_size, tcr_num, -1)
    #     log_frequency_features, _ = torch.max(log_frequency_features, dim=1)

    #     # V gene features
    #     v_gene_features = self.v_gene_embedding(v_gene_features)
    #     v_gene_features = v_gene_features.view(batch_size * tcr_num, -1)
    #     v_gene_features = F.elu(self.fc_vgene1(v_gene_features))
    #     v_gene_features = self.vgene_dropout(v_gene_features)
    #     v_gene_features = F.elu(self.fc_vgene2(v_gene_features))
    #     v_gene_features = self.vgene_dropout(v_gene_features)
    #     v_gene_features = v_gene_features.view(batch_size, tcr_num, -1)
    #     v_gene_features, _ = torch.max(v_gene_features, dim=1)

    #     # Gating and fusion for TCR and V gene only
    #     h1 = self.linear_h1(x_max)
    #     z1 = F.leaky_relu(self.linear_z1(x_max, v_gene_features))
    #     o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)

    #     h3 = self.linear_h3(v_gene_features)
    #     z3 = F.leaky_relu(self.linear_z3(v_gene_features, x_max))
    #     o3 = self.linear_o3(nn.Sigmoid()(z3) * h3)

    #     # Fusion of TCR and V gene features
    #     device = o1.device
    #     o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=device)), 1)
    #     o3 = torch.cat((o3, torch.ones(o3.shape[0], 1, device=device)), 1)

    #     o13 = torch.bmm(o1.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
    #     out = self.post_fusion_dropout(o13)

    #     # Combine fused features with frequency features
    #     combined_out = torch.cat((out, log_frequency_features), dim=-1)
    #     # print(f"combined_out: {combined_out.shape}") # torch.Size([64, 16769])

    #     assert combined_out.shape[1] == self.pre_encoder.in_features, f"Expected input features {self.pre_encoder.in_features}, but got {combined_out.shape[1]}"

    #     # Final encoding
    #     combined_out = self.pre_encoder(combined_out)
    #     combined_out = self.encoder1(combined_out)
    #     combined_out = self.encoder2(combined_out)
    #     combined_out = self.softmax(combined_out)
    #     # # Debug
    #     # print(f"combined_out: {combined_out.shape}") # combined_out: torch.Size([64, 2])
    #     if not caTCR_tag:
    #         return combined_out
    #     else: # 测试时使用
    #         return combined_out, top_indices

    # 加入UMAP
    # 如果caTCR_tag=True，希望将max pool之前的每个样本的前k个TCR序列提取出来，作为得分最高的序列，现有代码是从dim=1提取的，逻辑上似乎不对。
    def forward(self, x, adj, frequency_features, v_gene_features, caTCR_tag=False, return_features=False):
        # x: TCR 序列特征, [batch_size, tcr_num, feature_dim]
        # adj: 邻接矩阵, [batch_size, tcr_num, tcr_num]
        # frequency_features: 频率特征
        # v_gene_features: V基因特征
        # caTCR_tag: 是否提取重要节点
        # return_features: 是否返回融合特征供可视化使用
        
        # 初始特征投掷到 GAT 层前进行dropout
        x = self.dropout(x)
        
        # 经过多层GAT处理节点特征
        for gat_layer in self.gat_layers:
            x = torch.cat([att(x, adj) for att in gat_layer], dim=-1)  # 每层GAT后特征拼接
        # # debug
        # print(f"x_GAT: {x.shape}") # x_GAT: torch.Size([64, 300, 128])
        
        # TODO：dim=1池化之前需将topk序列提取进行下游分析
        if not caTCR_tag:
            # 图级特征表示，全局最大池化
            x_max, _ = torch.max(x, dim=1)
            # # debug
            # print(f"x_after_max: {x_max.shape}") # x_after_max: torch.Size([64, 128])
        else:
            # 提取最重要节点特征及索引
            x_max, max_indices = torch.max(x, dim=1) # 维度应该是0
            # # debug
            # print(f"x_after_max: {x_max.shape}") # x_after_max: torch.Size([64, 128])
            batch_size, feature_dim = max_indices.shape
            node_counts = torch.zeros(batch_size, x.size(1))
            for i in range(batch_size):
                for j in range(feature_dim):
                    node_counts[i, max_indices[i, j]] += 1
            top_k = 5
            top_values, top_indices = torch.topk(node_counts, k=top_k, dim=1)
        
        # 频率特征处理
        batch_size, tcr_num = frequency_features.shape[:2]
        log_frequency_features = torch.log(frequency_features + 1e-10)
        # # debug
        # print(f"log_freq: {log_frequency_features.shape}") # log_freq: torch.Size([64, 300])
        log_frequency_features = log_frequency_features.view(batch_size * tcr_num, -1)
        # # debug
        # print(f"log_freq_view1: {log_frequency_features.shape}") # log_freq_view1: torch.Size([19200, 1])
        log_frequency_features = F.leaky_relu(self.fc_freq1(log_frequency_features))
        log_frequency_features = F.leaky_relu(self.fc_freq2(log_frequency_features))
        log_frequency_features = log_frequency_features.view(batch_size, tcr_num, -1)
        # # debug
        # print(f"log_freq_view2: {log_frequency_features.shape}") # log_freq_view2: torch.Size([64, 300, 128])
        log_frequency_features, _ = torch.max(log_frequency_features, dim=1)
        # # debug
        # print(f"freq_after_max: {log_frequency_features.shape}") # freq_after_max: torch.Size([64, 128])

        # V基因特征处理
        # # debug
        # print(f"v_gene: {v_gene_features.shape}") # v_gene: torch.Size([64, 300])
        v_gene_features = self.v_gene_embedding(v_gene_features)
        # # debug
        # print(f"v_gene_embed: {v_gene_features.shape}") # v_gene_embed: torch.Size([64, 300, 16])
        v_gene_features = v_gene_features.view(batch_size * tcr_num, -1)
        # # debug
        # print(f"v_gene_view1: {v_gene_features.shape}") # v_gene_view1: torch.Size([19200, 16])
        v_gene_features = F.elu(self.fc_vgene1(v_gene_features))
        v_gene_features = self.vgene_dropout(v_gene_features)
        v_gene_features = F.elu(self.fc_vgene2(v_gene_features))
        v_gene_features = self.vgene_dropout(v_gene_features)
        v_gene_features = v_gene_features.view(batch_size, tcr_num, -1) 
        # # debug
        # print(f"v_gene_view2: {v_gene_features.shape}") # v_gene_view2: torch.Size([64, 300, 128])
        v_gene_features, _ = torch.max(v_gene_features, dim=1)
        # # debug
        # print(f"v_gene_after_max: {v_gene_features.shape}") # v_gene_after_max: torch.Size([64, 128])

        # 模态融合逻辑（TCR 和 V基因特征融合）
        h1 = self.linear_h1(x_max)
        z1 = F.leaky_relu(self.linear_z1(x_max, v_gene_features))
        o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)

        h3 = self.linear_h3(v_gene_features)
        z3 = F.leaky_relu(self.linear_z3(v_gene_features, x_max))
        o3 = self.linear_o3(nn.Sigmoid()(z3) * h3)

        # 拼接融合结果
        device = o1.device
        o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=device)), 1)
        o3 = torch.cat((o3, torch.ones(o3.shape[0], 1, device=device)), 1)
        o13 = torch.bmm(o1.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o13)

        # 结合频率特征，送入全连接层之前
        combined_out_pre = torch.cat((out, log_frequency_features), dim=-1)
        # # debug
        # print(f"combined_out_pre: {combined_out_pre.shape}") # combined_out_pre: torch.Size([64, 16769])

        # 全连接层进行分类
        combined_out = self.pre_encoder(combined_out_pre)
        combined_out = self.encoder1(combined_out)
        combined_out = self.encoder2(combined_out)
        # # debug
        # print(f"combined_out: {combined_out.shape}") # combined_out: torch.Size([64, 2])
        combined_out = self.softmax(combined_out)

        # 进入全连接层前的特征
        if return_features:
            return combined_out, top_indices, combined_out_pre  # 额外返回融合后的高维特征
        
        if caTCR_tag:
            return combined_out, top_indices
        else:
            return combined_out

    # # 使用UMAP特征进行分类
    # def forward(self, x, adj, frequency_features, v_gene_features, caTCR_tag=False, return_features=False):
    #     # x: TCR 序列特征, [batch_size, tcr_num, feature_dim]
    #     # adj: 邻接矩阵, [batch_size, tcr_num, tcr_num]
    #     # frequency_features: 频率特征
    #     # v_gene_features: V基因特征
    #     # caTCR_tag: 是否提取重要节点
    #     # return_features: 是否返回融合特征供可视化使用
        
    #     # dropout
    #     x = self.dropout(x)
        
    #     # 经过多层GAT处理节点特征
    #     for gat_layer in self.gat_layers:
    #         x = torch.cat([att(x, adj) for att in gat_layer], dim=-1)  # 每层GAT后特征拼接
        
    #     if not caTCR_tag:
    #         # 图级特征表示，全局最大池化
    #         x_max, _ = torch.max(x, dim=1)# [b*n*d] [b*d]
    #     else:
    #         # 提取节点特征及索引
    #         x_max, max_indices = torch.max(x, dim=1)
    #         batch_size, feature_dim = max_indices.shape
    #         node_counts = torch.zeros(batch_size, x.size(1))
    #         for i in range(batch_size):
    #             for j in range(feature_dim):
    #                 node_counts[i, max_indices[i, j]] += 1
    #         top_k = 5
    #         top_values, top_indices = torch.topk(node_counts, k=top_k, dim=1)
        
    #     # 频率特征处理
    #     batch_size, tcr_num = frequency_features.shape[:2]
    #     log_frequency_features = torch.log(frequency_features + 1e-10)
    #     log_frequency_features = log_frequency_features.view(batch_size * tcr_num, -1)
    #     log_frequency_features = F.leaky_relu(self.fc_freq1(log_frequency_features))
    #     log_frequency_features = F.leaky_relu(self.fc_freq2(log_frequency_features))
    #     log_frequency_features = log_frequency_features.view(batch_size, tcr_num, -1)
    #     log_frequency_features, _ = torch.max(log_frequency_features, dim=1)

    #     # V基因特征处理
    #     v_gene_features = self.v_gene_embedding(v_gene_features)
    #     v_gene_features = v_gene_features.view(batch_size * tcr_num, -1)
    #     v_gene_features = F.elu(self.fc_vgene1(v_gene_features))
    #     v_gene_features = self.vgene_dropout(v_gene_features)
    #     v_gene_features = F.elu(self.fc_vgene2(v_gene_features))
    #     v_gene_features = self.vgene_dropout(v_gene_features)
    #     v_gene_features = v_gene_features.view(batch_size, tcr_num, -1)
    #     v_gene_features, _ = torch.max(v_gene_features, dim=1)

    #     # 模态融合逻辑（TCR 和 V基因特征融合）
    #     h1 = self.linear_h1(x_max)
    #     z1 = F.leaky_relu(self.linear_z1(x_max, v_gene_features))
    #     o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)

    #     h3 = self.linear_h3(v_gene_features)
    #     z3 = F.leaky_relu(self.linear_z3(v_gene_features, x_max))
    #     o3 = self.linear_o3(nn.Sigmoid()(z3) * h3)

    #     # 拼接融合结果
    #     device = o1.device
    #     o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=device)), 1)
    #     o3 = torch.cat((o3, torch.ones(o3.shape[0], 1, device=device)), 1)
    #     o13 = torch.bmm(o1.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
    #     out = self.post_fusion_dropout(o13)

    #     # 结合频率特征，送入全连接层之前
    #     combined_out_pre = torch.cat((out, log_frequency_features), dim=-1)
    #     combined_out_pre_np = combined_out_pre.cpu().detach().numpy()  # 转换为numpy供UMAP使用
    #     umap_features = self.umap.fit_transform(combined_out_pre_np)  # UMAP降维
    #     umap_features = torch.tensor(umap_features, device=x.device)  # 转换回tensor

    #     # 使用UMAP特征进行分类
    #     combined_out = self.encoder1(umap_features)
    #     combined_out = self.encoder2(combined_out)
    #     combined_out = self.softmax(combined_out)

    #     # 返回逻辑
    #     if return_features:
    #         return combined_out, 0, umap_features
    #     else:
    #         return combined_out


    # @staticmethod
    def validate_model(model, valid_sps, valid_lbs, valid_mat, frequency_features, v_gene_features, device, valid_features=None):
        # Validation logic to compute the accuracy
        model.eval()
        # 准备数据
        if valid_features is not None:
            features = valid_features
        else:
            tcrbert = TCR_Bert(model_path="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/tcr/")
            features = tcrbert.process_sps(valid_sps)  # 计算节点特征
        adjacencies = np.array(valid_mat)
        labels = np.array(valid_lbs)

        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        labels_tensor = torch.LongTensor(labels).to(device)
        frequency_features_tensor = torch.Tensor(frequency_features).to(device)
        v_gene_features_tensor = torch.LongTensor(v_gene_features).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(features_tensor, adjacencies_tensor, frequency_features_tensor, v_gene_features_tensor)

        # Get probabilities and predicted classes
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probabilities for class 1
        preds = (probs > 0.5).astype(int)  # Convert probabilities to 0 or 1 based on threshold

        # Calculate accuracy, AUC, and F1 score
        acc = accuracy_score(labels, preds)
        # auc = roc_auc_score(labels, probs)
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.0  # 或者 float('nan')
            print("[Warning] AUC not defined for single-class labels.")
        f1 = f1_score(labels, preds)

        return acc, auc, f1

    @staticmethod
    def v_gene_to_numeric(v_gene, v_gene_dict):
        return v_gene_dict.get(v_gene, v_gene_dict['<UNK>'])
    
    def training(sps, lbs, adjs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, v_gene_dict, shuffle=False, valid_sps=None, valid_lbs=None, valid_mat=None, valid_freq=None, valid_vgene=None):
        best_auc, best_f1 = 0, 0
        tcrbert = TCR_Bert(model_path="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 节点特征
        # 增加验证集节点特征计算
        if valid_sps is not None:
            valid_features = tcrbert.process_sps(valid_sps)
        else:
            valid_features = None

        adjacencies = np.array(adjs)  # 所有样本的邻接矩阵
        lbs = np.array(lbs)  # 样本标签
        
        frequency_features = np.array([[float(tcr[2]) for tcr in sample] for sample in sps])
        v_gene_features = np.array([[Mulgat_vgene_fusion_freq.v_gene_to_numeric(tcr[1], v_gene_dict) for tcr in sample] for sample in sps])

        features_tensor = torch.Tensor(features)
        adjacencies_tensor = torch.Tensor(adjacencies)
        labels_tensor = torch.LongTensor(lbs)
        frequency_features_tensor = torch.Tensor(frequency_features)
        v_gene_features_tensor = torch.LongTensor(v_gene_features)
        print(f"device:{device}")
        # Initialize the model
        model = Mulgat_vgene_fusion_freq(tcr_num=tcr_num, feature_dim=768, drop_out=dropout,  
                                        frequency_dim=1, v_gene_vocab_size=len(v_gene_dict), num_gat_layers=2).cuda()
        # import pdb
        
        # pdb.set_trace()
        # print("hello")
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss().to(device)
        
        # Check if a model file already exists and adjust the filename
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        valid_model_f = model_f + "temp.pth"

        # Print shapes for debugging
        # print(f"features_tensor shape: {features_tensor.shape}")
        # print(f"adjacencies_tensor shape: {adjacencies_tensor.shape}")
        # print(f"labels_tensor shape: {labels_tensor.shape}")
        # print(f"frequency_features_tensor shape: {frequency_features_tensor.shape}")
        # print(f"v_gene_features_tensor shape: {v_gene_features_tensor.shape}")

        # Create a dataset and loader
        dataset = TensorDataset(features_tensor, adjacencies_tensor, labels_tensor, frequency_features_tensor, v_gene_features_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=shuffle)

        epoch_losses = []
        max_acc = 0
        # Training loop
        for epoch in range(ep):
            total_loss = 0
            for batch_x, batch_adj, batch_y, batch_frequency, batch_vgene in loader:
                batch_x, batch_adj, batch_y, batch_frequency, batch_vgene = batch_x.to(device), batch_adj.to(device), batch_y.to(device), batch_frequency.to(device), batch_vgene.to(device)

                outputs = model(batch_x, batch_adj, batch_frequency, batch_vgene)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()

                # # 打印梯度
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(name, param.grad.norm())

                optimizer.step()

            epoch_losses.append(total_loss)
            print(f"Epoch {epoch + 1}, Total Loss = {total_loss}")
            # Logging
            if (epoch + 1) % log_inr == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss.item()))

            # Optional validation step
            if valid_sps is not None and valid_lbs is not None and valid_mat is not None and valid_freq is not None and valid_vgene is not None:
                valid_acc, valid_auc, valid_f1 = Mulgat_vgene_fusion_freq.validate_model(model, valid_sps, valid_lbs, valid_mat, valid_freq, valid_vgene, device, valid_features=valid_features)
                print('Validation Accuracy:', valid_acc)
                print('Validation AUC:', valid_auc)
                print('Validation f1 score:', valid_f1)
                # Save the best model
                if valid_acc > max_acc:
                    max_acc = valid_acc
                    best_auc = valid_auc
                    best_f1 = valid_f1
                    torch.save(model.state_dict(), valid_model_f)
                    if os.path.exists(model_f):
                        os.remove(model_f)
                    os.rename(valid_model_f, model_f)
                elif os.path.exists(valid_model_f):
                    os.remove(valid_model_f)

        # Save the final model
        # torch.save(model.state_dict(), model_f)
        # TODO:能否保存最佳模型进行最后的预测
        print("The trained model has been saved to: " + model_f)
        print(f"[Fold Validation Best] Acc: {max_acc:.4f}, AUC: {best_auc:.4f}, F1: {best_f1:.4f}")

        epoch_losses_df = pd.DataFrame(epoch_losses, columns=["Loss"])

        return epoch_losses_df, max_acc, best_auc, best_f1

    # @staticmethod
    # 增加相关TCR保存逻辑
    # def prediction(sps, dismat, model_f, tcr_num, device, v_gene_dict, true_labels):
    #     # 加载模型
    #     model = Mulgat_vgene_fusion_freq(tcr_num=tcr_num, feature_dim=768, attention_head_num=4, 
    #                         attention_hidden_size=128, drop_out=0.2, 
    #                         frequency_dim=1, 
    #                         v_gene_vocab_size=len(v_gene_dict), 
    #                         num_gat_layers=2
    #                         ).to(device)
    #     model.load_state_dict(torch.load(model_f, map_location=device))
    #     model.eval()

    #     # 准备测试数据
    #     tcrbert = TCR_Bert(model_path="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/tcr/")
    #     features = tcrbert.process_sps(sps)  # 计算节点特征
    #     adjacencies = np.array(dismat)

    #     frequency_features = np.array([[float(tcr[2]) for tcr in sample] for sample in sps])
    #     v_gene_features = np.array([[Mulgat_vgene_fusion_freq.v_gene_to_numeric(tcr[1], v_gene_dict) for tcr in sample] for sample in sps])

    #     features_tensor = torch.Tensor(features).to(device)
    #     adjacencies_tensor = torch.Tensor(adjacencies).to(device)
    #     frequency_features_tensor = torch.Tensor(frequency_features).to(device)
    #     v_gene_features_tensor = torch.LongTensor(v_gene_features).to(device)

    #     # forward
    #     with torch.no_grad():
    #         outputs, top_indices = model(features_tensor, adjacencies_tensor, frequency_features_tensor, v_gene_features_tensor, caTCR_tag=True)
    #     # print("贡献最大的节点索引:", top_indices)

    #     import json
    #     # 保存相关节点
    #     output_data = []
    #     for i, top_indice in enumerate(top_indices):
    #         label = true_labels[i]
    #         caTCRs = []
    #         for indice in top_indice:
    #             # 获取 sps 中的前三个值，分别作为 amino_acid, v_gene 和 freq
    #             amino_acid = sps[i][indice][0]
    #             v_gene = sps[i][indice][1]
    #             freq = float(sps[i][indice][2])  # 转换为浮点数
    #             # 创建 caTCR 字典并添加到 caTCRs 列表
    #             caTCRs.append({
    #                 "amino_acid": amino_acid,
    #                 "v_gene": v_gene,
    #                 "freq": freq
    #             })
    #         # 将 label 和 caTCRs 封装成一个字典
    #         entry = {
    #             "label": label,
    #             "caTCRs": caTCRs
    #         }
    #         # 添加到输出数据列表中
    #         output_data.append(entry)
    #     # 保存为 JSON 文件
    #     with open('caTCR_output.json', 'w') as json_file:
    #         json.dump(output_data, json_file, indent=4)

    #     # probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 获取类别1的概率
    #     # 如果模型已经输出了softmax后的概率分布，可以直接使用outputs[:, 1]
    #     probs = outputs[:, 1].cpu().numpy()  # 获取类别1的概率

    #     preds = (probs > 0.5).astype(int)  # 根据概率阈值0.5获取预测类别
    #     # 打印混淆矩阵
    #     cm = confusion_matrix(true_labels, preds)
    #     print("Confusion Matrix:")
    #     print(cm)

    #     return probs


    def prediction_with_umap(sps, dismat, model_f, tcr_num, device, v_gene_dict, true_labels):
        # 加载模型
        model = Mulgat_vgene_fusion_freq(
            tcr_num=tcr_num, feature_dim=768, attention_head_num=4, 
            attention_hidden_size=128, drop_out=0.2, 
            frequency_dim=1, 
            v_gene_vocab_size=len(v_gene_dict), 
            num_gat_layers=2
        ).to(device)
        model.load_state_dict(torch.load(model_f, map_location=device))
        model.eval()

        # 准备测试数据
        ## debug
        # print(f"sps[0]: {sps[0]}")
        tcrbert = TCR_Bert(model_path="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/model_path/", 
                        src_dir="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 计算节点特征
        print(f"features: {features.shape}")
        adjacencies = np.array(dismat)

        frequency_features = np.array([[float(tcr[2]) for tcr in sample] for sample in sps])
        v_gene_features = np.array([[Mulgat_vgene_fusion_freq.v_gene_to_numeric(tcr[1], v_gene_dict) for tcr in sample] for sample in sps])

        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        frequency_features_tensor = torch.Tensor(frequency_features).to(device)
        v_gene_features_tensor = torch.LongTensor(v_gene_features).to(device)

        batch_size = 64
        probs = []
        high_dim_features = []

        # 分批处理数据
        for i in range(0, len(features), batch_size):
            batch_features = features_tensor[i:i+batch_size]
            batch_adjacencies = adjacencies_tensor[i:i+batch_size]
            batch_frequency_features = frequency_features_tensor[i:i+batch_size]
            batch_v_gene_features = v_gene_features_tensor[i:i+batch_size]

            with torch.no_grad():
                # 运行模型，返回分类结果和融合特征
                outputs, top_indices, fused_features = model(
                    batch_features, batch_adjacencies, 
                    batch_frequency_features, batch_v_gene_features, 
                    caTCR_tag=True, return_features=True
                )

            batch_probs = outputs[:, 1].cpu().numpy()
            probs.extend(batch_probs)
            high_dim_features.append(fused_features.cpu().numpy())  # 保存融合特征

        # 转换为numpy数组
        probs = np.array(probs)
        high_dim_features = np.concatenate(high_dim_features, axis=0)

        # 基于概率阈值进行分类
        preds = (probs > 0.5).astype(int)
        cm = confusion_matrix(true_labels, preds)
        print("Confusion Matrix:")
        print(cm)

        # 进行UMAP降维
        # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        # embedding = reducer.fit_transform(high_dim_features)

        # # 可视化结果
        # plt.figure(figsize=(10, 8))
        # scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=true_labels, cmap='coolwarm', alpha=0.7)
        # plt.colorbar(scatter, label='True Labels')
        # plt.title('UMAP Visualization of Fused Node Features')
        # plt.xlabel('UMAP Dimension 1')
        # plt.ylabel('UMAP Dimension 2')
        # plt.savefig('/mnt/sdb/juhengwei/PDAC_286/Processed_PBMC/raw_data_PBMC/umap_lr0.0001.png')
        # plt.close()
        # # plt.show()

        # return probs, embedding
        return probs
    
    # # 原始predict代码
    # def prediction(sps, dismat, model_f, tcr_num, device, v_gene_dict, true_labels):
    #     # 加载模型
    #     model = Mulgat_vgene_fusion_freq(
    #         tcr_num=tcr_num, feature_dim=768, attention_head_num=4, 
    #         attention_hidden_size=128, drop_out=0.2, 
    #         frequency_dim=1, 
    #         v_gene_vocab_size=len(v_gene_dict), 
    #         num_gat_layers=2
    #     ).to(device)
    #     model.load_state_dict(torch.load(model_f, map_location=device))
    #     model.eval()

    #     # 准备测试数据
    #     # Debug
    #     # print(f"sps: {sps.shape}")
    #     print(f"sps[0]: {sps[0]}")
    #     tcrbert = TCR_Bert(model_path="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/model_path/", 
    #                     src_dir="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/tcr/")
    #     features = tcrbert.process_sps(sps)  # 计算节点特征
    #     print(f"features: {features.shape}")
    #     adjacencies = np.array(dismat)

    #     frequency_features = np.array([[float(tcr[2]) for tcr in sample] for sample in sps])
    #     v_gene_features = np.array([[Mulgat_vgene_fusion_freq.v_gene_to_numeric(tcr[1], v_gene_dict) for tcr in sample] for sample in sps])

    #     features_tensor = torch.Tensor(features).to(device)
    #     adjacencies_tensor = torch.Tensor(adjacencies).to(device)
    #     frequency_features_tensor = torch.Tensor(frequency_features).to(device)
    #     v_gene_features_tensor = torch.LongTensor(v_gene_features).to(device)

    #     # 分批推理的batch_size
    #     batch_size = 64  # 根据GPU的内存情况调整大小
    #     probs = []

    #     # 分批处理数据
    #     for i in range(0, len(features), batch_size):
    #         # 获取当前批次的所有数据
    #         batch_features = features_tensor[i:i+batch_size]
    #         batch_adjacencies = adjacencies_tensor[i:i+batch_size]
    #         batch_frequency_features = frequency_features_tensor[i:i+batch_size]
    #         batch_v_gene_features = v_gene_features_tensor[i:i+batch_size]

    #         # forward
    #         with torch.no_grad():
    #             outputs,  top_indices = model(batch_features, batch_adjacencies, batch_frequency_features, batch_v_gene_features, caTCR_tag=True)

    #         # print("贡献最大的节点索引:", top_indices)
    #         # 获取类别1的概率
    #         batch_probs = outputs[:, 1].cpu().numpy()
    #         probs.extend(batch_probs)

    #     # 转换为numpy数组
    #     probs = np.array(probs)

    #     # 预测类别，使用0.5作为阈值
    #     preds = (probs > 0.5).astype(int)  # 根据概率阈值0.5获取预测类别

    #     # 打印混淆矩阵
    #     cm = confusion_matrix(true_labels, preds)
    #     print("Confusion Matrix:")
    #     print(cm)

    #     return probs

###################################################################################################
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, adj, x):
        return torch.matmul(adj, x)

def top_rank(attention_score, graph_indicator, keep_ratio):
    '''基于给定的attention_score，对每个图进行pooling操作，为了直观地体现pooling过程，'''
    graph_id_list = list(set(graph_indicator.cpu().numpy()))
    mask = attention_score.new_empty((0,), dtype=torch.bool)
    for graph_id in graph_id_list:
        graph_attn_score = attention_score[graph_indicator == graph_id]
        graph_node_num = len(graph_attn_score)
        graph_mask = attention_score.new_zeros((graph_node_num,), dtype=torch.bool)
        keep_graph_node_num = int(keep_ratio * graph_node_num)
        _, sorted_index = graph_attn_score.sort(descending=True)
        graph_mask[sorted_index[:keep_graph_node_num]] = True
        mask = torch.cat((mask, graph_mask))
    return mask

def filter_adjacency(adjacency, mask):
    '''更新邻接矩阵，只保留被mask选中的节点'''
    device = adjacency.device
    mask = mask.cpu().numpy()
    indices = adjacency.coalesce().indices().cpu().numpy()
    num_nodes = adjacency.size(0)
    row, col = indices
    maskout_self_loop = row != col
    row = row[maskout_self_loop]
    col = col[maskout_self_loop]
    sparse_adjacency = torch.sparse_coo_tensor(
        torch.tensor([row, col]), torch.ones(len(row)), size=(num_nodes, num_nodes)
    ).to(device)
    return sparse_adjacency[mask, :][:, mask]

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, keep_ratio, activation=torch.tanh):
        super(SelfAttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.keep_ratio = keep_ratio
        self.activation = activation
        self.attn_gcn = GraphConvolutionLayer(input_dim, 1)  # 用GCN计算注意力分数

    def forward(self, adjacency, input_feature, graph_indicator):
        # 计算每个节点的注意力分数
        attn_score = self.attn_gcn(adjacency, input_feature).squeeze()
        attn_score = self.activation(attn_score)
        
        # 使用top_rank函数选择重要节点
        mask = top_rank(attn_score, graph_indicator, self.keep_ratio)
        
        # 更新特征矩阵，选择保留的节点
        hidden = input_feature[mask] * attn_score[mask].view(-1, 1)
        mask_graph_indicator = graph_indicator[mask]
        
        # 更新邻接矩阵
        mask_adjacency = filter_adjacency(adjacency, mask)
        return hidden, mask_graph_indicator, mask_adjacency

# 图全局平均和最大池化的实现
def global_max_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_max(x, graph_indicator, dim=0, dim_size=num)[0]

def global_mean_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_mean(x, graph_indicator, dim=0, dim_size=num)

class Mulgat_vgene_fusion_freq_Hierarchical(nn.Module): 
    def __init__(self, tcr_num=300, feature_dim=768, attention_head_num=4, attention_hidden_size=128, num_gat_layers=2, drop_out=0.4, frequency_dim=1, v_gene_vocab_size=100, sagpool_ratio=0.5):
        super(Mulgat_vgene_fusion_freq_Hierarchical, self).__init__()
        self.tcr_num = tcr_num
        self.feature_dim = feature_dim
        self.num_gat_layers = num_gat_layers
        self.frequency_dim = frequency_dim
        self.sagpool_ratio = sagpool_ratio
        
        # V gene embedding
        self.v_gene_embedding_dim = 16
        self.v_gene_vocab_size = v_gene_vocab_size
        self.v_gene_embedding = nn.Embedding(self.v_gene_vocab_size, self.v_gene_embedding_dim)
        
        # Attention-related parameters
        self.attention_head_num = attention_head_num
        self.attention_hidden_size = attention_hidden_size
        self.attention_head_size = int(attention_hidden_size / attention_head_num)
        
        # GAT layers
        self.gat_layers = nn.ModuleList([
            nn.ModuleList([GraphAttentionLayerV2(feature_dim if i == 0 else self.attention_hidden_size, self.attention_head_size, dropout=drop_out) for _ in range(attention_head_num)])
            for i in range(num_gat_layers)
        ])

        # FCN for frequency features
        self.fc_freq1 = nn.Linear(frequency_dim, attention_hidden_size * 2)
        self.fc_freq2 = nn.Linear(attention_hidden_size * 2, 128)

        # FCN for V gene features
        self.fc_vgene1 = nn.Linear(self.v_gene_embedding_dim, attention_hidden_size * 2)
        self.fc_vgene2 = nn.Linear(attention_hidden_size * 2, 128)
        self.vgene_dropout = nn.AlphaDropout(p=0.2)

        # Gating mechanism for feature fusion
        self.linear_h1 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU())
        self.linear_z1 = nn.Bilinear(attention_hidden_size, attention_hidden_size, attention_hidden_size)
        self.linear_o1 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU(), nn.Dropout(p=drop_out))

        # Define SAGPool for each GAT layer
        self.pool = SelfAttentionPooling(attention_hidden_size, sagpool_ratio)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(attention_hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, 2)

        self.dropout = nn.Dropout(p=drop_out)
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, x, adj, frequency_features, v_gene_features, caTCR_tag=False, return_features=False):
        pooled_features_list = []

        for gat_layer in self.gat_layers:
            # Apply GAT layer
            x = torch.cat([att(x, adj) for att in gat_layer], dim=-1)  # Concatenate attention heads

            # Perform SAGPool after each GAT layer
            pooled_x, pooled_graph_indicator, pooled_adj = self.pool(adj, x, graph_indicator=None)
            
            # Perform global pooling for the pooled features
            global_pool = torch.cat([global_mean_pool(pooled_x, pooled_graph_indicator), global_max_pool(pooled_x, pooled_graph_indicator)], dim=1)
            pooled_features_list.append(global_pool)  # Store the result of this layer

            # Update x to be the pooled features for the next layer
            x = pooled_x

        # Combine all pooled features from different layers
        combined_pooled_features = torch.cat(pooled_features_list, dim=1)

        # Correctly map frequency and V_gene features to the pooled nodes
        log_frequency_features = torch.log(frequency_features + 1e-10)
        log_frequency_features = self.fc_freq1(log_frequency_features)
        log_frequency_features = self.fc_freq2(log_frequency_features)

        # V_gene feature fusion, based on the retained nodes from pooling
        v_gene_features = self.v_gene_embedding(v_gene_features)
        v_gene_features = v_gene_features.view(x.size(0), x.size(1), -1)  # Adjust for pooled nodes
        v_gene_features, _ = torch.max(v_gene_features, dim=1)

        # Concatenate pooled features with frequency and V_gene features
        combined_features = torch.cat([combined_pooled_features, log_frequency_features, v_gene_features], dim=-1)

        # Final classification
        combined_out = self.fc1(combined_features)
        combined_out = self.fc2(combined_out)
        combined_out = self.softmax(combined_out)

        if return_features:
            return combined_out, combined_features

        return combined_out

    def training(sps, lbs, adjs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, v_gene_dict, shuffle=False, valid_sps=None, valid_lbs=None, valid_mat=None, valid_freq=None, valid_vgene=None):
        # Prepare features using TCR-Bert
        tcrbert = TCR_Bert(model_path="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # TCR sequence features

        adjacencies = np.array(adjs)  # Adjacency matrices
        lbs = np.array(lbs)  # Labels

        frequency_features = np.array([[float(tcr[2]) for tcr in sample] for sample in sps])
        v_gene_features = np.array([[Mulgat_vgene_fusion_freq.v_gene_to_numeric(tcr[1], v_gene_dict) for tcr in sample] for sample in sps])

        features_tensor = torch.Tensor(features)
        adjacencies_tensor = torch.Tensor(adjacencies)
        labels_tensor = torch.LongTensor(lbs)
        frequency_features_tensor = torch.Tensor(frequency_features)
        v_gene_features_tensor = torch.LongTensor(v_gene_features)
        
        print(f"device:{device}")
        
        # Initialize the model with the updated version that includes SAGPool
        model = Mulgat_vgene_fusion_freq_Hierarchical(tcr_num=tcr_num, feature_dim=768, drop_out=dropout,  
                                                    frequency_dim=1, v_gene_vocab_size=len(v_gene_dict), 
                                                    num_gat_layers=2, sagpool_ratio=0.5).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss().to(device)
        
        # Check if a model file already exists and adjust the filename
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        valid_model_f = model_f + "temp.pth"

        # Create a dataset and loader
        dataset = TensorDataset(features_tensor, adjacencies_tensor, labels_tensor, frequency_features_tensor, v_gene_features_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=shuffle)

        epoch_losses = []
        max_acc = 0
        
        # Training loop
        for epoch in range(ep):
            total_loss = 0
            for batch_x, batch_adj, batch_y, batch_frequency, batch_vgene in loader:
                batch_x, batch_adj, batch_y, batch_frequency, batch_vgene = batch_x.to(device), batch_adj.to(device), batch_y.to(device), batch_frequency.to(device), batch_vgene.to(device)

                # Forward pass
                outputs = model(batch_x, batch_adj, batch_frequency, batch_vgene)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_losses.append(total_loss)
            print(f"Epoch {epoch + 1}, Total Loss = {total_loss}")

            # Logging
            if (epoch + 1) % log_inr == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss.item()))

            # Optional validation step
            if valid_sps is not None and valid_lbs is not None and valid_mat is not None and valid_freq is not None and valid_vgene is not None:
                valid_acc, valid_auc, valid_f1 = Mulgat_fre_vgene_fusion.validate_model(model, valid_sps, valid_lbs, valid_mat, valid_freq, valid_vgene, device)
                print('Validation Accuracy:', valid_acc)
                print('Validation AUC:', valid_auc)
                print('Validation f1 score:', valid_f1)

                # Save the best model
                if valid_acc > max_acc:
                    max_acc = valid_acc
                    torch.save(model.state_dict(), valid_model_f)
                    if os.path.exists(model_f):
                        os.remove(model_f)
                    os.rename(valid_model_f, model_f)
                else:
                    os.remove(valid_model_f)

        # Save the final model
        torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)

        epoch_losses_df = pd.DataFrame(epoch_losses, columns=["Loss"])

        return epoch_losses_df


# 去掉max pooling，用mean pooling替代
class Mulgat_vgene_fusion_freq_meanpooling(nn.Module): 
    def __init__(self, tcr_num=300, feature_dim=768, attention_head_num=4, attention_hidden_size=128, num_gat_layers=2, drop_out=0.4, frequency_dim=1, v_gene_vocab_size=100):
        super(Mulgat_vgene_fusion_freq_meanpooling, self).__init__()
        self.tcr_num = tcr_num
        self.feature_dim = feature_dim
        self.num_gat_layers = num_gat_layers
        self.frequency_dim = frequency_dim
        
        # V gene embedding
        self.v_gene_embedding_dim = 16
        self.v_gene_vocab_size = v_gene_vocab_size
        self.v_gene_embedding = nn.Embedding(self.v_gene_vocab_size, self.v_gene_embedding_dim)
        
        # Attention-related parameters
        self.attention_head_num = attention_head_num
        self.attention_hidden_size = attention_hidden_size
        self.attention_head_size = int(attention_hidden_size / attention_head_num)
        
        self.gat_layers = nn.ModuleList([
            nn.ModuleList([
                GraphAttentionLayerV2(feature_dim if i == 0 else self.attention_hidden_size, self.attention_head_size, dropout=drop_out)
                for _ in range(attention_head_num)
            ]) for i in range(num_gat_layers)
        ])

        # FCN for frequency features
        """改进4：简化频率特征处理"""
        # # 1→256→128
        # self.fc_freq1 = nn.Linear(frequency_dim, attention_hidden_size * 2)
        # self.fc_freq2 = nn.Linear(attention_hidden_size * 2, 128)
        self.fc_freq1 = nn.Linear(1, 64)
        self.fc_freq2 = nn.Identity()

        # FCN for V gene features
        self.fc_vgene1 = nn.Linear(self.v_gene_embedding_dim, attention_hidden_size * 2)
        self.fc_vgene2 = nn.Linear(attention_hidden_size * 2, 128)
        self.vgene_dropout = nn.AlphaDropout(p=0.2)

        # Gating mechanism for feature fusion (TCR and V gene only)
        # self.linear_h1 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU())
        self.linear_h1 = nn.Sequential(nn.Linear(attention_hidden_size*2, attention_hidden_size), nn.LeakyReLU())
        self.linear_z1 = nn.Bilinear(attention_hidden_size*2, attention_hidden_size, attention_hidden_size)
        self.linear_o1 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU(), nn.Dropout(p=drop_out))

        self.linear_h3 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU())
        self.linear_z3 = nn.Bilinear(attention_hidden_size, attention_hidden_size*2, attention_hidden_size)
        self.linear_o3 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU(), nn.Dropout(p=drop_out))

        self.post_fusion_dropout = nn.Dropout(p=drop_out)
        # fused_dim = (attention_hidden_size + 1) ** 2  # 根据融合逻辑计算的维度
        # final_dim = fused_dim + 128  # 融合后的维度加上频率特征的维度
        fused_dim = attention_hidden_size
        final_dim = fused_dim + 64
        self.pre_encoder = nn.Linear(final_dim, 768)  # 确保输入维度匹配
        self.encoder1 = nn.Sequential(nn.Linear(768, attention_hidden_size), nn.LeakyReLU(), nn.Dropout(p=drop_out))
        # self.encoder2 = nn.Sequential(nn.Linear(attention_hidden_size, 2), nn.LeakyReLU(), nn.Dropout(p=drop_out))
        """改进3：分类器增强"""
        # self.encoder2 = nn.Sequential(nn.Linear(attention_hidden_size, 2))
        self.encoder2 = nn.Sequential(
            nn.Linear(attention_hidden_size, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 2)
        )

        # self.encoder1 = nn.Sequential(nn.Linear(2, attention_hidden_size), nn.LeakyReLU(), nn.Dropout(p=drop_out))
        # self.encoder2 = nn.Sequential(nn.Linear(attention_hidden_size, 2))

        self.dropout = nn.Dropout(p=drop_out)
        self.softmax = nn.Softmax(dim=1)  # 添加 softmax 激活函数

    # 加入UMAP
    def forward(self, x, adj, frequency_features, v_gene_features, caTCR_tag=False, return_features=False):
        # x: TCR 序列特征, [batch_size, tcr_num, feature_dim]
        # adj: 邻接矩阵, [batch_size, tcr_num, tcr_num]
        # frequency_features: 频率特征
        # v_gene_features: V基因特征
        # caTCR_tag: 是否提取重要节点
        # return_features: 是否返回融合特征供可视化使用
        
        # 初始特征投掷到 GAT 层前进行dropout
        x = self.dropout(x)
        
        # 经过多层GAT处理节点特征
        for gat_layer in self.gat_layers:
            x = torch.cat([att(x, adj) for att in gat_layer], dim=-1)  # 每层GAT后特征拼接
        # debug
        # print(f"x_GAT: {x.shape}") # x_GAT: torch.Size([64, 300, 128])
        
        if not caTCR_tag:
            # 图级特征表示，全局最大池化
            # x_max, _ = torch.max(x, dim=1)
            """修改1：改为混合池化"""
            x_max = torch.max(x, dim=1)[0]
            x_mean = torch.mean(x, dim=1)
            x_graph = torch.cat([x_max, x_mean], dim=1)
            # debug
            # print(f"x_after_max: {x_max.shape}") # x_after_max: torch.Size([64, 128])
        else:
            # 提取最重要节点特征及索引
            x_max, max_indices = torch.max(x, dim=1)
            # debug
            # print(f"x_after_max: {x_max.shape}") # x_after_max: torch.Size([64, 128])
            batch_size, feature_dim = max_indices.shape
            node_counts = torch.zeros(batch_size, x.size(1))
            for i in range(batch_size):
                for j in range(feature_dim):
                    node_counts[i, max_indices[i, j]] += 1
            top_k = 5
            top_values, top_indices = torch.topk(node_counts, k=top_k, dim=1)
        
        # 频率特征处理
        batch_size, tcr_num = frequency_features.shape[:2]
        log_frequency_features = torch.log(frequency_features + 1e-10)
        # debug
        # print(f"log_freq: {log_frequency_features.shape}") # log_freq: torch.Size([64, 300])
        log_frequency_features = log_frequency_features.view(batch_size * tcr_num, -1)
        # debug
        # print(f"log_freq_view1: {log_frequency_features.shape}") # log_freq_view1: torch.Size([19200, 1])
        log_frequency_features = F.leaky_relu(self.fc_freq1(log_frequency_features))
        log_frequency_features = F.leaky_relu(self.fc_freq2(log_frequency_features))
        log_frequency_features = log_frequency_features.view(batch_size, tcr_num, -1)
        # debug
        # print(f"log_freq_view2: {log_frequency_features.shape}") # log_freq_view2: torch.Size([64, 300, 128])
        log_frequency_features, _ = torch.max(log_frequency_features, dim=1)
        # debug
        # print(f"freq_after_max: {log_frequency_features.shape}") # freq_after_max: torch.Size([64, 128])
        # log_frequency_features, _ = torch.max(log_frequency_features, dim=1)
        # print(f"freq_after_max: {log_frequency_features.shape}") # freq_after_max: torch.Size([64, 128])

        # V基因特征处理
        # debug
        # print(f"v_gene: {v_gene_features.shape}") # v_gene: torch.Size([64, 300])
        v_gene_features = self.v_gene_embedding(v_gene_features)
        # debug
        # print(f"v_gene_embed: {v_gene_features.shape}") # v_gene_embed: torch.Size([64, 300, 16])
        v_gene_features = v_gene_features.view(batch_size * tcr_num, -1)
        # debug
        # print(f"v_gene_view1: {v_gene_features.shape}") # v_gene_view1: torch.Size([19200, 16])
        v_gene_features = F.elu(self.fc_vgene1(v_gene_features))
        v_gene_features = self.vgene_dropout(v_gene_features)
        v_gene_features = F.elu(self.fc_vgene2(v_gene_features))
        v_gene_features = self.vgene_dropout(v_gene_features)
        v_gene_features = v_gene_features.view(batch_size, tcr_num, -1) 
        # debug
        # print(f"v_gene_view2: {v_gene_features.shape}") # v_gene_view2: torch.Size([64, 300, 128])
        v_gene_features, _ = torch.max(v_gene_features, dim=1)
        # debug
        # print(f"v_gene_after_max: {v_gene_features.shape}") # v_gene_after_max: torch.Size([64, 128])

        # 模态融合逻辑（TCR 和 V基因特征融合）
        h1 = self.linear_h1(x_graph)
        z1 = F.leaky_relu(self.linear_z1(x_graph, v_gene_features))
        o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)

        h3 = self.linear_h3(v_gene_features)
        z3 = F.leaky_relu(self.linear_z3(v_gene_features, x_graph))
        o3 = self.linear_o3(nn.Sigmoid()(z3) * h3)

        # 拼接融合结果
        device = o1.device
        # o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=device)), 1)
        # o3 = torch.cat((o3, torch.ones(o3.shape[0], 1, device=device)), 1)
        # o13 = torch.bmm(o1.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        # out = self.post_fusion_dropout(o13)
        """修改2：简化融合计算"""
        # o13 = (o1 * o3).sum(dim=1)
        o13 = (o1 * o3)
        # print(f"o13: {o13.shape}")

        # 结合频率特征，送入全连接层之前
        combined_out_pre = torch.cat((o13, log_frequency_features), dim=-1)
        # debug
        # print(f"combined_out_pre: {combined_out_pre.shape}") # combined_out_pre: torch.Size([64, 16769])

        # 全连接层进行分类
        combined_out = self.pre_encoder(combined_out_pre)
        # combined_out = nn.Linear(combined_out_pre.shape[0], 768)
        combined_out = self.encoder1(combined_out)
        combined_out = self.encoder2(combined_out)
        # debug
        # print(f"combined_out: {combined_out.shape}") # combined_out: torch.Size([64, 2])
        combined_out = self.softmax(combined_out)

        # 进入全连接层前的特征
        if return_features:
            return combined_out, combined_out_pre  # 额外返回融合后的高维特征
        
        if caTCR_tag:
            return combined_out, top_indices
        else:
            return combined_out

    # @staticmethod
    def validate_model(model, valid_sps, valid_lbs, valid_mat, frequency_features, v_gene_features, device):
        # Validation logic to compute the accuracy
        model.eval()
        # 准备数据
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(valid_sps)  # 计算节点特征
        adjacencies = np.array(valid_mat)
        labels = np.array(valid_lbs)

        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        labels_tensor = torch.LongTensor(labels).to(device)
        frequency_features_tensor = torch.Tensor(frequency_features).to(device)
        v_gene_features_tensor = torch.LongTensor(v_gene_features).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(features_tensor, adjacencies_tensor, frequency_features_tensor, v_gene_features_tensor)

        # Get probabilities and predicted classes
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probabilities for class 1
        preds = (probs > 0.5).astype(int)  # Convert probabilities to 0 or 1 based on threshold

        # Calculate accuracy, AUC, and F1 score
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, preds)

        return acc, auc, f1

    @staticmethod
    def v_gene_to_numeric(v_gene, v_gene_dict):
        return v_gene_dict.get(v_gene, v_gene_dict['<UNK>'])
    
    def training(sps, lbs, adjs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, v_gene_dict, shuffle=False, valid_sps=None, valid_lbs=None, valid_mat=None, valid_freq=None, valid_vgene=None, pretrained_model_path=None):
        tcrbert = TCR_Bert(model_path="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 节点特征

        adjacencies = np.array(adjs)  # 所有样本的邻接矩阵
        lbs = np.array(lbs)  # 样本标签
        
        frequency_features = np.array([[float(tcr[2]) for tcr in sample] for sample in sps])
        v_gene_features = np.array([[Mulgat_vgene_fusion_freq_meanpooling.v_gene_to_numeric(tcr[1], v_gene_dict) for tcr in sample] for sample in sps])

        print(f"v_gene_dict: {len(v_gene_features)}")

        features_tensor = torch.Tensor(features)
        adjacencies_tensor = torch.Tensor(adjacencies)
        labels_tensor = torch.LongTensor(lbs)
        frequency_features_tensor = torch.Tensor(frequency_features)
        v_gene_features_tensor = torch.LongTensor(v_gene_features)
        print(f"device:{device}")
        # Initialize the model
        model = Mulgat_vgene_fusion_freq_meanpooling(tcr_num=tcr_num, feature_dim=768, drop_out=dropout,  
                                        frequency_dim=1, v_gene_vocab_size=len(v_gene_dict), num_gat_layers=2).cuda()
        # 加载现有模型
        if pretrained_model_path is not None:
            # 加载预训练的参数字典
            pretrained_dict = torch.load(pretrained_model_path, map_location=device)
            # 检查并调整 v_gene_embedding.weight 的大小
            if 'v_gene_embedding.weight' in pretrained_dict:
                pretrained_weight = pretrained_dict['v_gene_embedding.weight']
                current_vocab_size = model.v_gene_embedding.weight.size(0)
                # 如果预训练参数的词表大小与当前模型不一致
                if pretrained_weight.size(0) != current_vocab_size:
                    if pretrained_weight.size(0) > current_vocab_size:
                        # 如果预训练参数的行数多，则裁剪，只取前 current_vocab_size 行
                        pretrained_weight = pretrained_weight[:current_vocab_size, :]
                    else:
                        # 如果预训练参数的行数少，则进行填充，填充值为0
                        pad_size = current_vocab_size - pretrained_weight.size(0)
                        pad_tensor = torch.zeros(pad_size, pretrained_weight.size(1), device=pretrained_weight.device)
                        pretrained_weight = torch.cat([pretrained_weight, pad_tensor], dim=0)
                    # 更新参数字典中 v_gene_embedding 的权重
                    pretrained_dict['v_gene_embedding.weight'] = pretrained_weight
            # model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
            model.load_state_dict(pretrained_dict, strict=False)
            print(f"Loaded model from {pretrained_model_path}")
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss().to(device)
        
        # Check if a model file already exists and adjust the filename
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        valid_model_f = model_f + "temp.pth"

        # Create a dataset and loader
        dataset = TensorDataset(features_tensor, adjacencies_tensor, labels_tensor, frequency_features_tensor, v_gene_features_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=shuffle)

        epoch_losses = []
        max_acc = 0
        # Training loop
        for epoch in range(ep):
            total_loss = 0
            for batch_x, batch_adj, batch_y, batch_frequency, batch_vgene in loader:
                batch_x, batch_adj, batch_y, batch_frequency, batch_vgene = batch_x.to(device), batch_adj.to(device), batch_y.to(device), batch_frequency.to(device), batch_vgene.to(device)

                outputs = model(batch_x, batch_adj, batch_frequency, batch_vgene)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

            epoch_losses.append(total_loss)
            print(f"Epoch {epoch + 1}, Total Loss = {total_loss}")
            # Logging
            if (epoch + 1) % log_inr == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss.item()))

            # Optional validation step
            if valid_sps is not None and valid_lbs is not None and valid_mat is not None and valid_freq is not None and valid_vgene is not None:
                valid_acc, valid_auc, valid_f1 = Mulgat_fre_vgene_fusion.validate_model(model, valid_sps, valid_lbs, valid_mat, valid_freq, valid_vgene, device)
                print('Validation Accuracy:', valid_acc)
                print('Validation AUC:', valid_auc)
                print('Validation f1 score:', valid_f1)
                # Save the best model
                if valid_acc > max_acc:
                    max_acc = valid_acc
                    torch.save(model.state_dict(), valid_model_f)
                    if os.path.exists(model_f):
                        os.remove(model_f)
                    os.rename(valid_model_f, model_f)
                else:
                    os.remove(valid_model_f)

        # Save the final model
        torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)

        epoch_losses_df = pd.DataFrame(epoch_losses, columns=["Loss"])

        return epoch_losses_df

    def prediction_with_umap(sps, dismat, model_f, tcr_num, device, v_gene_dict, true_labels):
        # 加载模型
        model = Mulgat_vgene_fusion_freq_meanpooling(
            tcr_num=tcr_num, feature_dim=768, attention_head_num=4, 
            attention_hidden_size=128, drop_out=0.2, 
            frequency_dim=1, 
            v_gene_vocab_size=len(v_gene_dict), 
            num_gat_layers=2
        ).to(device)
        model.load_state_dict(torch.load(model_f, map_location=device))
        model.eval()

        # 准备测试数据
        # print(f"sps[0]: {sps[0]}")
        tcrbert = TCR_Bert(model_path="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/model_path/", 
                        src_dir="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 计算节点特征
        # print(f"features: {features.shape}")
        adjacencies = np.array(dismat)

        frequency_features = np.array([[float(tcr[2]) for tcr in sample] for sample in sps])
        v_gene_features = np.array([[Mulgat_vgene_fusion_freq.v_gene_to_numeric(tcr[1], v_gene_dict) for tcr in sample] for sample in sps])

        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        frequency_features_tensor = torch.Tensor(frequency_features).to(device)
        v_gene_features_tensor = torch.LongTensor(v_gene_features).to(device)

        batch_size = 64
        probs = []
        high_dim_features = []

        # 分批处理数据
        for i in range(0, len(features), batch_size):
            batch_features = features_tensor[i:i+batch_size]
            batch_adjacencies = adjacencies_tensor[i:i+batch_size]
            batch_frequency_features = frequency_features_tensor[i:i+batch_size]
            batch_v_gene_features = v_gene_features_tensor[i:i+batch_size]

            with torch.no_grad():
                # 运行模型，返回分类结果和融合特征
                outputs, fused_features = model(
                    batch_features, batch_adjacencies, 
                    batch_frequency_features, batch_v_gene_features, 
                    caTCR_tag=False, return_features=True
                )

            batch_probs = outputs[:, 1].cpu().numpy()
            probs.extend(batch_probs)
            high_dim_features.append(fused_features.cpu().numpy())  # 保存融合特征

        # 转换为numpy数组
        probs = np.array(probs)
        high_dim_features = np.concatenate(high_dim_features, axis=0)

        # 基于概率阈值进行分类
        preds = (probs > 0.5).astype(int)
        cm = confusion_matrix(true_labels, preds)
        print("Confusion Matrix:")
        print(cm)

        # 进行UMAP降维
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(high_dim_features)

        # 可视化结果
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=true_labels, cmap='coolwarm', alpha=0.7)
        plt.colorbar(scatter, label='True Labels')
        plt.title('UMAP Visualization of Fused Node Features')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.savefig('/mnt/sdb/juhengwei/小鼠/前列腺癌/processed_data/raw_data/test_few_shot.png')
        plt.close()

        return probs
    
# 多分类模型
# 着重修改
class Mulgat_vgene_fusion_freq_multi_task(nn.Module):
    def __init__(
            self, tcr_num=300, feature_dim=768, attention_head_num=4, attention_hidden_size=128, 
                 num_gat_layers=2, drop_out=0.4, frequency_dim=1, v_gene_vocab_size=100, num_classes=7
        ):
        super(Mulgat_vgene_fusion_freq_multi_task, self).__init__()
        self.tcr_num = tcr_num
        self.feature_dim = feature_dim
        self.num_gat_layers = num_gat_layers
        self.frequency_dim = frequency_dim
        
        # V gene embedding
        self.v_gene_embedding_dim = 16
        self.v_gene_vocab_size = v_gene_vocab_size
        self.v_gene_embedding = nn.Embedding(self.v_gene_vocab_size, self.v_gene_embedding_dim)
        
        # Attention-related parameters
        self.attention_head_num = attention_head_num
        self.attention_hidden_size = attention_hidden_size
        self.attention_head_size = int(attention_hidden_size / attention_head_num)
        
        self.gat_layers = nn.ModuleList([
            nn.ModuleList([
                GraphAttentionLayerV2(feature_dim if i == 0 else self.attention_hidden_size, self.attention_head_size, dropout=drop_out)
                for _ in range(attention_head_num)
            ]) for i in range(num_gat_layers)
        ])

        # FCN for frequency features
        # 1→256→128
        self.fc_freq1 = nn.Linear(frequency_dim, attention_hidden_size * 2)
        self.fc_freq2 = nn.Linear(attention_hidden_size * 2, 128)

        # FCN for V gene features
        self.fc_vgene1 = nn.Linear(self.v_gene_embedding_dim, attention_hidden_size * 2)
        self.fc_vgene2 = nn.Linear(attention_hidden_size * 2, 128)
        self.vgene_dropout = nn.AlphaDropout(p=0.2)

        # Gating mechanism for feature fusion (TCR and V gene only)
        self.linear_h1 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU())
        self.linear_z1 = nn.Bilinear(attention_hidden_size, attention_hidden_size, attention_hidden_size)
        self.linear_o1 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU(), nn.Dropout(p=drop_out))

        self.linear_h3 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU())
        self.linear_z3 = nn.Bilinear(attention_hidden_size, attention_hidden_size, attention_hidden_size)
        self.linear_o3 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU(), nn.Dropout(p=drop_out))

        self.post_fusion_dropout = nn.Dropout(p=drop_out)
        fused_dim = (attention_hidden_size + 1) ** 2  # 根据融合逻辑计算的维度
        final_dim = fused_dim + 128  # 融合后的维度加上频率特征的维度
        self.pre_encoder = nn.Linear(final_dim, 768)  # 确保输入维度匹配
        self.encoder1 = nn.Sequential(nn.Linear(768, attention_hidden_size), nn.LeakyReLU(), nn.Dropout(p=drop_out))
        # self.encoder2 = nn.Sequential(nn.Linear(attention_hidden_size, 2), nn.LeakyReLU(), nn.Dropout(p=drop_out))
        # # 修改输出层的神经元数量以支持多类别预测
        self.encoder2 = nn.Sequential(
            nn.Linear(attention_hidden_size, num_classes)
        ) # 

        self.dropout = nn.Dropout(p=drop_out)
        self.softmax = nn.Softmax(dim=1)  # 添加 softmax 激活函数

    def forward(self, x, adj, frequency_features, v_gene_features):
        # x: Node features, assumed to be [batch_size, tcr_num, feature_dim]
        # adj: Adjacency matrix, assumed to be [batch_size, tcr_num, tcr_num]
        x = self.dropout(x) 
        
        for gat_layer in self.gat_layers:
            x = torch.cat([att(x, adj) for att in gat_layer], dim=-1)
        x, _ = torch.max(x, dim=1)

        # Frequency features
        batch_size, tcr_num = frequency_features.shape[:2]
        log_frequency_features = torch.log(frequency_features + 1e-10)
        log_frequency_features = log_frequency_features.view(batch_size * tcr_num, -1)
        log_frequency_features = F.leaky_relu(self.fc_freq1(log_frequency_features))
        log_frequency_features = F.leaky_relu(self.fc_freq2(log_frequency_features)) 
        log_frequency_features = log_frequency_features.view(batch_size, tcr_num, -1)
        log_frequency_features, _ = torch.max(log_frequency_features, dim=1)

        # V gene features
        v_gene_features = self.v_gene_embedding(v_gene_features)
        v_gene_features = v_gene_features.view(batch_size * tcr_num, -1)
        v_gene_features = F.elu(self.fc_vgene1(v_gene_features))
        v_gene_features = self.vgene_dropout(v_gene_features)
        v_gene_features = F.elu(self.fc_vgene2(v_gene_features))
        v_gene_features = self.vgene_dropout(v_gene_features)
        v_gene_features = v_gene_features.view(batch_size, tcr_num, -1)
        v_gene_features, _ = torch.max(v_gene_features, dim=1)

        # Gating and fusion for TCR and V gene only
        h1 = self.linear_h1(x)
        z1 = F.leaky_relu(self.linear_z1(x, v_gene_features))
        o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)

        h3 = self.linear_h3(v_gene_features)
        z3 = F.leaky_relu(self.linear_z3(v_gene_features, x))
        o3 = self.linear_o3(nn.Sigmoid()(z3) * h3)

        # Fusion of TCR and V gene features
        device = o1.device
        o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=device)), 1)
        o3 = torch.cat((o3, torch.ones(o3.shape[0], 1, device=device)), 1)

        o13 = torch.bmm(o1.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o13)

        # Combine fused features with frequency features
        combined_out = torch.cat((out, log_frequency_features), dim=-1)

        assert combined_out.shape[1] == self.pre_encoder.in_features, f"Expected input features {self.pre_encoder.in_features}, but got {combined_out.shape[1]}"

        # Final encoding
        combined_out = self.pre_encoder(combined_out)
        combined_out = self.encoder1(combined_out)
        combined_out = self.encoder2(combined_out)

        combined_out = self.softmax(combined_out)

        return combined_out

    # @staticmethod
    def validate_model(model, valid_sps, valid_lbs, valid_mat, frequency_features, v_gene_features, device):
        # Validation logic to compute the accuracy
        model.eval()
        # 准备数据
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(valid_sps)  # 计算节点特征
        adjacencies = np.array(valid_mat)
        labels = np.array(valid_lbs)

        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        labels_tensor = torch.LongTensor(labels).to(device)
        frequency_features_tensor = torch.Tensor(frequency_features).to(device)
        v_gene_features_tensor = torch.LongTensor(v_gene_features).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(features_tensor, adjacencies_tensor, frequency_features_tensor, v_gene_features_tensor)

        # Get probabilities and predicted classes
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probabilities for class 1
        preds = (probs > 0.5).astype(int)  # Convert probabilities to 0 or 1 based on threshold

        # Calculate accuracy, AUC, and F1 score
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, preds)

        return acc, auc, f1

    @staticmethod
    def v_gene_to_numeric(v_gene, v_gene_dict):
        return v_gene_dict.get(v_gene, v_gene_dict['<UNK>'])
    
    def training(sps, lbs, adjs, tcr_num, lr, ep, dropout, log_inr, model_f, 
                 aa_f, device, v_gene_dict, shuffle=False, valid_sps=None, valid_lbs=None, 
                 valid_mat=None, valid_freq=None, valid_vgene=None, num_classes=7
                ):
        tcrbert = TCR_Bert(model_path="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 节点特征

        adjacencies = np.array(adjs)  # 所有样本的邻接矩阵
        lbs = np.array(lbs)  # 样本标签
        
        frequency_features = np.array([[float(tcr[2]) for tcr in sample] for sample in sps])
        v_gene_features = np.array([[Mulgat_vgene_fusion_freq.v_gene_to_numeric(tcr[1], v_gene_dict) for tcr in sample] for sample in sps])

        features_tensor = torch.Tensor(features)
        adjacencies_tensor = torch.Tensor(adjacencies)
        labels_tensor = torch.LongTensor(lbs) # # 确保标签是LongTensor，以适应CrossEntropyLoss
        frequency_features_tensor = torch.Tensor(frequency_features)
        v_gene_features_tensor = torch.LongTensor(v_gene_features)

        print(f"device:{device}")
        # Initialize the model
        model = Mulgat_vgene_fusion_freq_multi_task(tcr_num=tcr_num, feature_dim=768, drop_out=dropout,  
                                        frequency_dim=1, v_gene_vocab_size=len(v_gene_dict), num_gat_layers=4, num_classes=num_classes).to(device)
        # import pdb
        
        # pdb.set_trace()
        # print("hello")

        ######################################################################################################
        # # 类别不平衡的解决方法一
        # # 计算类别权重
        # class_weights = compute_class_weight(
        # class_weight='balanced',  # 自动计算平衡权重
        # classes=np.arange(num_classes),
        # y=lbs
        # )    
        # class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        # print(f"Class Weights: {class_weights}")  # 输出类别权重检查
        # # 使用带权重的 CrossEntropyLoss
        # criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        ######################################################################################################
    
        # 调整优化器和学习率调度
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        # 创建学习率调度器
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # 每 10 个 epoch 学习率乘以 0.5
        # 定义余弦退火调度器
        scheduler = CosineAnnealingLR(optimizer, T_max=600, eta_min=1e-6)  # eta_min 为最低学习率
        
        # # 初始优化器和损失
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss().to(device)
        
        # Check if a model file already exists and adjust the filename
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        valid_model_f = model_f + "temp.pth"

        # Print shapes for debugging
        # print(f"features_tensor shape: {features_tensor.shape}")
        # print(f"adjacencies_tensor shape: {adjacencies_tensor.shape}")
        # print(f"labels_tensor shape: {labels_tensor.shape}")
        # print(f"frequency_features_tensor shape: {frequency_features_tensor.shape}")
        # print(f"v_gene_features_tensor shape: {v_gene_features_tensor.shape}")

        # Create a dataset and loader
        dataset = TensorDataset(features_tensor, adjacencies_tensor, labels_tensor, frequency_features_tensor, v_gene_features_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=shuffle)

        epoch_losses = []
        max_acc = 0
        # Training loop
        for epoch in range(ep):
            total_loss = 0
            correct = 0
            total = 0

            for batch_x, batch_adj, batch_y, batch_frequency, batch_vgene in loader:
                batch_x, batch_adj, batch_y, batch_frequency, batch_vgene = batch_x.to(device), batch_adj.to(device), batch_y.to(device), batch_frequency.to(device), batch_vgene.to(device)

                outputs = model(batch_x, batch_adj, batch_frequency, batch_vgene)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()

                # # 打印梯度
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(name, param.grad.norm())

                optimizer.step()

            scheduler.step()  # 更新学习率

            epoch_losses.append(total_loss)
            epoch_acc = 100 * correct / total
            print(f"Epoch {epoch + 1}, Total Loss = {total_loss}, Accuracy = {epoch_acc:.2f}%")

            # Logging
            if (epoch + 1) % log_inr == 0:
                print(f'Epoch: {epoch + 1}, Loss: {total_loss:.6f}, Accuracy: {epoch_acc:.2f}%')

            # Optional validation step
            if valid_sps is not None and valid_lbs is not None and valid_mat is not None and valid_freq is not None and valid_vgene is not None:
                valid_acc, valid_auc, valid_f1 = Mulgat_fre_vgene_fusion.validate_model(model, valid_sps, valid_lbs, valid_mat, valid_freq, valid_vgene, device)
                print(f'Validation Accuracy: {valid_acc:.2f}%')
                print(f'Validation AUC: {valid_auc:.2f}')
                print(f'Validation F1 score: {valid_f1:.2f}')

                # Save the best model
                if valid_acc > max_acc:
                    max_acc = valid_acc
                    torch.save(model.state_dict(), valid_model_f)
                    if os.path.exists(model_f):
                        os.remove(model_f)
                    os.rename(valid_model_f, model_f)
                else:
                    os.remove(valid_model_f)

        # Save the final model
        torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)

        epoch_losses_df = pd.DataFrame(epoch_losses, columns=["Loss"])

        return epoch_losses_df

    #     return probs
    def prediction(sps, dismat, model_f, tcr_num, device, v_gene_dict, true_labels, num_classes=7):
        # 加载模型
        model = Mulgat_vgene_fusion_freq_multi_task(
            tcr_num=tcr_num, feature_dim=768, attention_head_num=4, 
            attention_hidden_size=128, drop_out=0.2, 
            frequency_dim=1, 
            v_gene_vocab_size=len(v_gene_dict), 
            num_gat_layers=3,
            num_classes=num_classes
        ).to(device)
        model.load_state_dict(torch.load(model_f, map_location=device))
        model.eval()

        # 准备测试数据
        tcrbert = TCR_Bert(model_path="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/model_path/", 
                        src_dir="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 计算节点特征
        adjacencies = np.array(dismat)

        frequency_features = np.array([[float(tcr[2]) for tcr in sample] for sample in sps])
        v_gene_features = np.array([[Mulgat_vgene_fusion_freq.v_gene_to_numeric(tcr[1], v_gene_dict) for tcr in sample] for sample in sps])

        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        frequency_features_tensor = torch.Tensor(frequency_features).to(device)
        v_gene_features_tensor = torch.LongTensor(v_gene_features).to(device)

        # 分批推理的batch_size
        batch_size = 60  # 根据GPU的内存情况调整大小
        probs = []

        # 分批处理数据
        for i in range(0, len(features), batch_size):
            # 获取当前批次的所有数据
            batch_features = features_tensor[i:i+batch_size]
            batch_adjacencies = adjacencies_tensor[i:i+batch_size]
            batch_frequency_features = frequency_features_tensor[i:i+batch_size]
            batch_v_gene_features = v_gene_features_tensor[i:i+batch_size]

            # forward
            with torch.no_grad():
                outputs = model(batch_features, batch_adjacencies, batch_frequency_features, batch_v_gene_features)
            
            # 输出是每个样本的所有类别的概率, outputs: [batch_size, num_classes]
            batch_probs = torch.softmax(outputs, dim=-1).cpu().numpy()  # 对输出进行softmax转化为概率
            probs.extend(batch_probs)

        # 转换为numpy数组
        probs = np.array(probs)

        # 获取预测的类别, 使用argmax选择概率最大的类别
        preds = np.argmax(probs, axis=1)  # 每个样本选择概率最大的类别

        # 计算并打印评估指标
        acc = accuracy_score(true_labels, preds)
        macro_f1 = f1_score(true_labels, preds, average='macro')
        weighted_f1 = f1_score(true_labels, preds, average='weighted')
        macro_p = precision_score(true_labels, preds, average='macro')
        weighted_p = precision_score(true_labels, preds, average='weighted')
        macro_r = recall_score(true_labels, preds, average='macro')
        weighted_r = recall_score(true_labels, preds, average='weighted')
        class_f1 = f1_score(true_labels, preds, average=None, labels=np.arange(num_classes))

        # # 打印混淆矩阵
        # cm = confusion_matrix(true_labels, preds)
        # print("Confusion Matrix:")
        # print(cm)

        print(f"Accuracy: {acc:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")
        print(f"Weighted-F1: {weighted_f1:.4f}")
        print(f"Macro Precision: {macro_p:.4f}")
        print(f"Weighted Precision: {weighted_p:.4f}")
        print(f"Macro Recall: {macro_r:.4f}")
        print(f"Weighted Recall: {weighted_r:.4f}")
        print(f"F1 per class: {class_f1}")

        return probs

# 三模态融合，有问题
class Mulgat_fre_vgene_fusion(nn.Module): 
    def __init__(self, tcr_num=300, feature_dim=768, attention_head_num=4, attention_hidden_size=128, num_gat_layers=2, drop_out=0.4, frequency_dim=1, v_gene_vocab_size=100):
        super(Mulgat_fre_vgene_fusion, self).__init__()
        self.tcr_num = tcr_num  # The number of TCR sequences per sample
        self.feature_dim = feature_dim  # Dimension of each TCR sequence feature (from TCRbert)
        self.num_gat_layers = num_gat_layers
        self.frequency_dim = frequency_dim
        
        # V gene embedding
        self.v_gene_embedding_dim = 16  # Dimension of V gene embedding
        self.v_gene_vocab_size = v_gene_vocab_size  # Vocabulary size for V gene
        self.v_gene_embedding = nn.Embedding(self.v_gene_vocab_size, self.v_gene_embedding_dim)
        
        # Attention-related parameters
        self.attention_head_num = attention_head_num  # Number of attention heads
        self.attention_hidden_size = attention_hidden_size  # Hidden size for attention
        self.attention_head_size = int(attention_hidden_size / attention_head_num)  # 100 Size of each attention head
        
        self.gat_layers = nn.ModuleList([
            nn.ModuleList([
                GraphAttentionLayerV2(feature_dim if i == 0 else self.attention_hidden_size, self.attention_head_size, dropout=drop_out)
                for _ in range(attention_head_num)
            ]) for i in range(num_gat_layers)
        ])
        print("hello")
        # FCN for frequency features
        # 1→256→128
        self.fc_freq1 = nn.Linear(frequency_dim, attention_hidden_size * 2)
        self.fc_freq2 = nn.Linear(attention_hidden_size * 2, 128)

        # FCN for V gene features
        # 16→256→128
        self.fc_vgene1 = nn.Linear(self.v_gene_embedding_dim, attention_hidden_size * 2)
        self.fc_vgene2 = nn.Linear(attention_hidden_size * 2, 128)
        self.vgene_dropout = nn.AlphaDropout(p=0.2)

        # Gating mechanism for feature fusion
        self.linear_h1 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU())
        self.linear_z1 = nn.Bilinear(attention_hidden_size, attention_hidden_size, attention_hidden_size)
        self.linear_o1 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU(), nn.Dropout(p=drop_out))

        self.linear_h2 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU())
        self.linear_z2 = nn.Bilinear(attention_hidden_size, attention_hidden_size, attention_hidden_size)
        self.linear_o2 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU(), nn.Dropout(p=drop_out))

        self.linear_h3 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU())
        self.linear_z3 = nn.Bilinear(attention_hidden_size, attention_hidden_size, attention_hidden_size)
        self.linear_o3 = nn.Sequential(nn.Linear(attention_hidden_size, attention_hidden_size), nn.LeakyReLU(), nn.Dropout(p=drop_out))
        print("hello")
        # import pdb
        # pdb.set_trace()
        self.post_fusion_dropout = nn.Dropout(p=drop_out)
        # self.pre_encoder = nn.Linear(38700, 768)
        # self.pre_encoder = nn.Linear(attention_hidden_size * 3, 768)
        self.pre_encoder = nn.Linear((attention_hidden_size + 1) ** 3, 768)
        self.encoder1 = nn.Sequential(nn.Linear(768, attention_hidden_size), nn.LeakyReLU(), nn.Dropout(p=drop_out))
        self.encoder2 = nn.Sequential(nn.Linear(attention_hidden_size, 2), nn.LeakyReLU(), nn.Dropout(p=drop_out))

        self.dropout = nn.Dropout(p=drop_out)
        print("hello")


    def forward(self, x, adj, frequency_features, v_gene_features):
        # x: Node features, assumed to be [batch_size, tcr_num, feature_dim]
        # adj: Adjacency matrix, assumed to be [batch_size, tcr_num, tcr_num]
        # The values in adj are not binary but represent the weighted distance between nodes.

        x = self.dropout(x) 
        
        for gat_layer in self.gat_layers:
            x = torch.cat([att(x, adj) for att in gat_layer], dim=-1) # torch.Size([3, 300, 256])
        # 最大池化
        x, _ = torch.max(x, dim=1)
        print(f"x_maxpool_shape:{x.shape}")

        # Frequency features
        batch_size, tcr_num = frequency_features.shape[:2]
        log_frequency_features = torch.log(frequency_features + 1e-10)
        log_frequency_features = log_frequency_features.view(batch_size * tcr_num, -1)
        log_frequency_features = F.leaky_relu(self.fc_freq1(log_frequency_features))
        log_frequency_features = F.leaky_relu(self.fc_freq2(log_frequency_features)) 
        log_frequency_features = log_frequency_features.view(batch_size, tcr_num, -1) # torch.Size([3, 300, 256])
        # 最大池化
        log_frequency_features, _ = torch.max(log_frequency_features, dim=1)
        print(f"log_frequency_features_maxpool_shape:{log_frequency_features.shape}")

        # V gene features
        v_gene_features = self.v_gene_embedding(v_gene_features) # embedding        
        v_gene_features = v_gene_features.view(batch_size * tcr_num, -1)
        # 过两个线性层：FCN + elu + Alphadropout, FCN + elu + Alphadropout
        v_gene_features = F.elu(self.fc_vgene1(v_gene_features))
        v_gene_features = self.vgene_dropout(v_gene_features)
        v_gene_features = F.elu(self.fc_vgene2(v_gene_features))
        v_gene_features = self.vgene_dropout(v_gene_features)
        v_gene_features = v_gene_features.view(batch_size, tcr_num, -1) # torch.Size([3, 300, 256])
        # 最大池化
        v_gene_features, _ = torch.max(v_gene_features, dim=1)
        print(f"v_gene_features_maxpool_shape:{v_gene_features.shape}") 

        # del x, adj, frequency_features, v_gene_features

        # Degug
        # print("111111111111111111111111111111111111111111111111111111111111111")
        # print(f"x_shape:{x.shape}") # x_shape:torch.Size([3, 300, 256])
        # print(f"log_frequency_features_shapr:{log_frequency_features.shape}") # log_frequency_features_shapr:torch.Size([3, 300, 256])
        # print(f"v_gene_features_shape:{v_gene_features.shape}") # v_gene_features_shape:torch.Size([3, 300, 256])

        # Gating and fusion
        # 减少线性层的使用，看看结果
        # 检查发现这里都没有激活函数
        h1 = self.linear_h1(x) # 线性层 * 1
        # h1 = x
        z1 = F.leaky_relu(self.linear_z1(x, log_frequency_features)) # seq, freq 线性层 * 1
        o1 = self.linear_o1(nn.Sigmoid()(z1) * h1) # 线性层 * 1
        # o1 = nn.Sigmoid()(z1) * h1

        h2 = self.linear_h2(log_frequency_features)
        # h2 = log_frequency_features
        z2 = F.leaky_relu(self.linear_z2(log_frequency_features, v_gene_features))
        o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        # o2 = nn.Sigmoid()(z2) * h2

        h3 = self.linear_h3(v_gene_features)
        # h3 = v_gene_features
        z3 = F.leaky_relu(self.linear_z3(v_gene_features, x))
        o3 = self.linear_o3(nn.Sigmoid()(z3) * h3)
        # o3 = nn.Sigmoid()(z3) * h3

        # Debug
        # print("222222222222222222222222222222222222222222222222222222222222222")
        # print(f"h1_shape:{h1.shape}") # h1_shape:torch.Size([3, 300, 256])
        # print(f"z1_shape:{z1.shape}") # z1_shape:torch.Size([3, 300, 256])
        # print(f"o1_shape:{o1.shape}\n") # o1_shape:torch.Size([3, 300, 256])

        # print(f"h2_shape:{h2.shape}") # h2_shape:torch.Size([3, 300, 256])
        # print(f"z2_shape:{z2.shape}") # z2_shape:torch.Size([3, 300, 256])
        # print(f"o2_shape:{o2.shape}\n") # o2_shape:torch.Size([3, 300, 256])

        # print(f"h3_shape:{h3.shape}") # h3_shape:torch.Size([3, 300, 256])
        # print(f"z3_shape:{z3.shape}") # z3_shape:torch.Size([3, 300, 256])
        # print(f"o3_shape:{o3.shape}") # o3_shape:torch.Size([3, 300, 256])

        # Fusion
        # o1 = torch.cat((o1, torch.ones(o1.shape[0], o1.shape[1], 1, device=o1.device)), 2)
        # o2 = torch.cat((o2, torch.ones(o2.shape[0], o2.shape[1], 1, device=o2.device)), 2)
        # o3 = torch.cat((o3, torch.ones(o3.shape[0], o3.shape[1], 1, device=o3.device)), 2)

        # Debug
        # print("333333333333333333333333333333333333333333333333333333333333333")
        # print(f"o1_shape:{o1.shape}") # o1_shape:torch.Size([3, 300, 257]) ([batch_size, tcr_num, feature_dim])
        # print(f"o2_shape:{o2.shape}") # o2_shape:torch.Size([3, 300, 257]) ([batch_size, tcr_num, feature_dim])
        # print(f"o3_shape:{o3.shape}") # o3_shape:torch.Size([3, 300, 257]) ([batch_size, tcr_num, feature_dim])

        # Fusion, bmm, 三维的情况
        # o12 = torch.bmm(o1.permute(0, 2, 1), o2)
        # # print(f"o12:{o12.shape}") # o12:torch.Size([3, 257, 257])
        # o12_permuted = o12.permute(0, 2, 1) # 调整 o12 的维度，以便进行第二步矩阵乘法
        # # print(f"o12:{o12.shape}") # torch.Size([3, 257, 257])
        # o123 = torch.bmm(o12_permuted, o3.permute(0, 2, 1))
        # # print("o12 形状:", o12.shape) # torch.Size([3, 257, 257])
        # # print("o123 形状:", o123.shape) # torch.Size([3, 257, 300])
        # o123 = o123.flatten(start_dim=1)
        # # print("o123 形状:", o123.shape)
        # out = self.post_fusion_dropout(o123)

        # fusion, 二维
        device = o1.device
        o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=device)), 1)
        o2 = torch.cat((o2, torch.ones(o2.shape[0], 1, device=device)), 1)
        o3 = torch.cat((o3, torch.ones(o3.shape[0], 1, device=device)), 1)

        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o123 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        # o1o2 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1))
        # o12 = o1o2.view(o1o2.size(0), -1)

        # o12o3 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1))
        # o123 = o12o3.view(o12o3.size(0), -1)

        out = self.post_fusion_dropout(o123)

        # 门控之后直接拼接
        # out = torch.cat((o1, o2, o3), dim=-1) # # [batch_size, tcr_num, attention_hidden_size * 3]
        # out = out.view(len(out), -1) # # [batch_size, tcr_num * attention_hidden_size * 3]
        # out = self.post_fusion_dropout(out)
        # print("out 形状:", out.shape) # torch.Size([3, 38700])
        # 这里out过了三个线性层
        out = self.pre_encoder(out)
        out = self.encoder1(out)
        out = self.encoder2(out)

        return out
    import pdb
    # @staticmethod
    def validate_model(model, valid_sps, valid_lbs, valid_mat, frequency_features, v_gene_features, device):
        # Validation logic to compute the accuracy
        model.eval()
        # 准备数据
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(valid_sps)  # 计算节点特征
        adjacencies = np.array(valid_mat)
        labels = np.array(valid_lbs)

        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        labels_tensor = torch.LongTensor(labels).to(device)
        frequency_features_tensor = torch.Tensor(frequency_features).to(device)
        v_gene_features_tensor = torch.LongTensor(v_gene_features).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(features_tensor, adjacencies_tensor, frequency_features_tensor, v_gene_features_tensor)

        # Get probabilities and predicted classes
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probabilities for class 1
        preds = (probs > 0.5).astype(int)  # Convert probabilities to 0 or 1 based on threshold

        # Calculate accuracy, AUC, and F1 score
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, preds)

        return acc, auc, f1

    @staticmethod
    def v_gene_to_numeric(v_gene, v_gene_dict):
        return v_gene_dict.get(v_gene, v_gene_dict['<UNK>'])
    
    def training(sps, lbs, adjs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, v_gene_dict, shuffle=False, valid_sps=None, valid_lbs=None, valid_mat=None, valid_freq=None, valid_vgene=None):
        tcrbert = TCR_Bert(model_path="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/juhengwei/GAT_all/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 节点特征

        adjacencies = np.array(adjs)  # 所有样本的邻接矩阵
        lbs = np.array(lbs)  # 样本标签
        
        frequency_features = np.array([[float(tcr[2]) for tcr in sample] for sample in sps])
        v_gene_features = np.array([[Mulgat_fre_vgene_fusion.v_gene_to_numeric(tcr[1], v_gene_dict) for tcr in sample] for sample in sps])

        features_tensor = torch.Tensor(features)
        adjacencies_tensor = torch.Tensor(adjacencies)
        labels_tensor = torch.LongTensor(lbs)
        frequency_features_tensor = torch.Tensor(frequency_features)
        v_gene_features_tensor = torch.LongTensor(v_gene_features)
        print(f"device:{device}")
        # Initialize the model
        model = Mulgat_fre_vgene_fusion(tcr_num=tcr_num, feature_dim=768, drop_out=dropout,  
                                        frequency_dim=1, v_gene_vocab_size=len(v_gene_dict), num_gat_layers=2).cuda()
        # import pdb
        
        # pdb.set_trace()
        print("hello")
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss().to(device)
        
        # Check if a model file already exists and adjust the filename
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        valid_model_f = model_f + "temp.pth"

        # Print shapes for debugging
        print(f"features_tensor shape: {features_tensor.shape}")
        print(f"adjacencies_tensor shape: {adjacencies_tensor.shape}")
        print(f"labels_tensor shape: {labels_tensor.shape}")
        print(f"frequency_features_tensor shape: {frequency_features_tensor.shape}")
        print(f"v_gene_features_tensor shape: {v_gene_features_tensor.shape}")

        # Create a dataset and loader
        dataset = TensorDataset(features_tensor, adjacencies_tensor, labels_tensor, frequency_features_tensor, v_gene_features_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=shuffle)

        epoch_losses = []
        max_acc = 0
        # Training loop
        for epoch in range(ep):
            total_loss = 0
            for batch_x, batch_adj, batch_y, batch_frequency, batch_vgene in loader:
                batch_x, batch_adj, batch_y, batch_frequency, batch_vgene = batch_x.to(device), batch_adj.to(device), batch_y.to(device), batch_frequency.to(device), batch_vgene.to(device)

                outputs = model(batch_x, batch_adj, batch_frequency, batch_vgene)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()

                # 打印梯度
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(name, param.grad.norm())

                optimizer.step()

            epoch_losses.append(total_loss)
            print(f"Epoch {epoch + 1}, Total Loss = {total_loss}")
            # Logging
            if (epoch + 1) % log_inr == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss.item()))

            # Optional validation step
            if valid_sps is not None and valid_lbs is not None and valid_mat is not None and valid_freq is not None and valid_vgene is not None:
                valid_acc, valid_auc, valid_f1 = Mulgat_fre_vgene_fusion.validate_model(model, valid_sps, valid_lbs, valid_mat, valid_freq, valid_vgene, device)
                print('Validation Accuracy:', valid_acc)
                print('Validation AUC:', valid_auc)
                print('Validation f1 score:', valid_f1)
                # Save the best model
                if valid_acc > max_acc:
                    max_acc = valid_acc
                    torch.save(model.state_dict(), valid_model_f)
                    if os.path.exists(model_f):
                        os.remove(model_f)
                    os.rename(valid_model_f, model_f)
                else:
                    os.remove(valid_model_f)

        # Save the final model
        torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)

        epoch_losses_df = pd.DataFrame(epoch_losses, columns=["Loss"])

        return epoch_losses_df

    # @staticmethod
    def prediction(sps, dismat, model_f, tcr_num, device, v_gene_dict, true_labels):
        # 加载模型
        model = Mulgat_fre_vgene_fusion(tcr_num=tcr_num, feature_dim=768, attention_head_num=4, 
                            attention_hidden_size=128, drop_out=0.2, 
                            frequency_dim=1, 
                            v_gene_vocab_size=len(v_gene_dict), 
                            num_gat_layers=2
                            ).to(device)
        model.load_state_dict(torch.load(model_f, map_location=device))
        model.eval()

        # 准备测试数据
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 计算节点特征
        adjacencies = np.array(dismat)

        frequency_features = np.array([[float(tcr[2]) for tcr in sample] for sample in sps])
        v_gene_features = np.array([[Mulgat_fre_vgene_fusion.v_gene_to_numeric(tcr[1], v_gene_dict) for tcr in sample] for sample in sps])

        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        frequency_features_tensor = torch.Tensor(frequency_features).to(device)
        v_gene_features_tensor = torch.LongTensor(v_gene_features).to(device)

        # forward
        with torch.no_grad():
            outputs = model(features_tensor, adjacencies_tensor, frequency_features_tensor, v_gene_features_tensor)

        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 获取类别1的概率

        preds = (probs > 0.5).astype(int)  # 根据概率阈值0.5获取预测类别
        # 打印混淆矩阵
        cm = confusion_matrix(true_labels, preds)
        print("Confusion Matrix:")
        print(cm)

        return probs

# # GAT
# class GraphAttentionLayer(nn.Module):
#     def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2):
#         super(GraphAttentionLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.dropout = dropout
#         self.alpha = alpha

#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)

#         self.leakyrelu = nn.LeakyReLU(self.alpha)

#     def forward(self, h, adj):
#         Wh = torch.matmul(h, self.W)  # [batch_size, tcr_num, out_features]

#         # Efficient attention mechanism
#         f_1 = torch.einsum('ijk,kl->ijl', Wh, self.a[:self.out_features, :])  # [batch_size, tcr_num, 1]
#         f_2 = torch.einsum('ijk,kl->ijl', Wh, self.a[self.out_features:, :])  # [batch_size, tcr_num, 1]

#         e = self.leakyrelu(f_1 + f_2.transpose(1, 2))  # [batch_size, tcr_num, tcr_num]

#         # 转换adj矩阵，使得距离小的值赋予更大的权重
#         adj_exp = torch.exp(-adj)  # 使用负指数函数转换距离
#         # Integrate the adjacency matrix with attention scores
#         # 使用乘法调整权重
#         e = e * adj_exp

#         # Applying softmax to normalize attention scores
#         attention = F.softmax(e, dim=2)
#         attention = F.dropout(attention, self.dropout, training=self.training)

#         # Apply attention and aggregate
#         h_prime = torch.matmul(attention, Wh)
#         # print(f"h_prime:{h_prime.shape}") # torch.Size([8, 100, 100])

#         return h_prime

# GATv2
class GraphAttentionLayerV2(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2):
        super(GraphAttentionLayerV2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # 初始化权重矩阵W和注意力参数a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # 线性变换
        Wh = torch.matmul(h, self.W)  # [batch_size, tcr_num, out_features]
        # print(f"Wh_size:{Wh.shape}")
        # 现在首先计算所有对的特征组合
        a_input = self._prepare_attentional_mechanism_input(Wh)
        # print(f"a_input_size:{a_input.shape}")
        e = torch.matmul(a_input, self.a).squeeze(2)

        # 激活函数应用于注意力得分的计算
        e = self.leakyrelu(e)
        e = e.squeeze(-1) # 移除最后一个维度

        # 使用负指数函数转换邻接矩阵
        adj_exp = torch.exp(-adj)
        
        # print(f"e_size:{e.shape}")
        # print(f"adj_size:{adj_exp.shape}")
        # 乘以邻接矩阵的指数转换版本
        e = e * adj_exp

        # 归一化处理
        attention = F.softmax(e, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 使用注意力权重聚合
        h_prime = torch.matmul(attention, Wh)

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        batch_size, num_nodes, _ = Wh.shape
        Wh_expanded = Wh.unsqueeze(2).repeat(1, 1, num_nodes, 1)
        Wh_repeated = Wh.unsqueeze(1).repeat(1, num_nodes, 1, 1)
        all_combinations_matrix = torch.cat([Wh_expanded, Wh_repeated], dim=3)
        return all_combinations_matrix

"""
    torch_geometric库的一个缺点是如果需要加入边的信息，则需要自定义代码，实现较为复杂。
    改进后的GAT
    1. 注意力机制的优化：参考现有的公式和keras代码
    2. 对边信息融合的优化
    3. 加入multi-head
    4. 加入early stopping
"""
class DeepLion2_GATv2(nn.Module):
    def __init__(self, tcr_num=100, feature_dim=768, nhid=8, nclass=2, attention_head_num=1, alpha=0.2, attention_hidden_size=100, drop_out=0.4):
        super(DeepLION2_GAT, self).__init__()
        # self.tcr_num = tcr_num  # The number of TCR sequences per sample
        # self.feature_dim = feature_dim  # Dimension of each TCR sequence feature (from TCRbert)
        
        # # Attention-related parameters
        # self.attention_head_num = attention_head_num  # Number of attention heads
        # self.attention_hidden_size = attention_hidden_size  # Hidden size for attention
        # self.attention_head_size = int(attention_hidden_size / attention_head_num)  # 100 Size of each attention head
        
        # # Graph Attention Layers (we will define these layers below)
        # self.attentions = nn.ModuleList([
        #     GraphAttentionLayer(feature_dim, self.attention_head_size, dropout=drop_out)
        #     for _ in range(attention_head_num)
        # ])
        
        # # Output fully connected layer
        # # self.out_fc = nn.Linear(feature_dim * attention_head_num, 2)  # Assume binary classification
        # self.out_fc = nn.Linear(tcr_num, 2)
        

        # self.dropout = nn.Dropout(drop_out)
        self.dropout = drop_out
        # 创建一个由多个GraphAttentionLayer实例组成的列表，每个实例对应一个注意力“头”。这些层将输入特征从nfeat转换到nhid
        self.attentions = [GraphAttentionLayer2(feature_dim, nhid, dropout=drop_out, alpha=alpha, concat=True) for _ in range(attention_head_num)] # 需要定义GraphAttentionLayer
        # 为每个注意力层注册一个唯一的模块名，使得它们可以被PyTorch的模型管理和优化框架正确处理。
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        # 设置一个单独的输出注意力层，这个层的任务是将多个头的输出合并，并将其转换为最终的类别数nclass。此层不使用拼接（concat=False）
        self.out_att = GraphAttentionLayer2(nhid * attention_head_num, nclass, dropout=drop_out, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # x: Node features, assumed to be [batch_size, tcr_num, feature_dim]
        # adj: Adjacency matrix, assumed to be [batch_size, tcr_num, tcr_num]
        # The values in adj are not binary but represent the weighted distance between nodes.

        # Node features are initially of size [batch_size, tcr_num, feature_dim]
        # print(f"Initial x shape: {x.shape}")  # Debug: check the shape of the input features # torch.Size([8, 100, 768])
        # print(f"Adjacency matrix shape: {adj.shape}")  # Debug: check the shape of the adjacency matrix # torch.Size([8, 100, 100])
 
        # Apply dropout to input features
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.dropout(x)
        
        # Apply the graph attention layers
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        # After concatenating the output of each attention head, x should have shape:
        # [batch_size, tcr_num, feature_dim * attention_head_num]

        # print(f"Shape after attention: {x.shape}")  # Debug: check the shape after attention

        # Mean pooling across the node dimension to aggregate node features into a single vector per sample
        x = torch.mean(x, dim=1)  # [batch_size, feature_dim * attention_head_num]
        # print(f"x:{x.shape}")
        # Apply a final linear transformation
        x = self.out_fc(x)  # [batch_size, 2] x:torch.Size([8, 2])

        # Assuming you want a single probability per sample, apply softmax and select one class's probability,
        # typically, this would be the probability of class 1 for binary classification

        return x
    @staticmethod
    def validate_model(model, valid_sps, valid_lbs, valid_mat, device):
        # Validation logic to compute the accuracy
        model.eval()
        # 准备数据
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(valid_sps) # 计算节点特征
        adjacencies = np.array(valid_mat)
        labels = np.array(valid_lbs)

        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)
        labels_tensor = torch.LongTensor(labels).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(features_tensor, adjacencies_tensor)

        # Get probabilities and predicted classes
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probabilities for class 1
        preds = (probs > 0.5).astype(int)  # Convert probabilities to 0 or 1 based on threshold

        # Calculate accuracy, AUC, and F1 score
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, preds)

        # print(f"Validation Accuracy: {acc:.4f}")
        # print(f"Validation AUC: {auc:.4f}")
        # print(f"Validation F1 Score: {f1:.4f}")

        return acc, auc, f1

    @staticmethod
    def training(sps, lbs, adjs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, shuffle=False, valid_sps=None, valid_lbs=None, valid_mat=None):
        # Assume sps includes both features and adjacency matrices for each sample
        # Assume lbs is a list of labels corresponding to each sample in sps

        # Convert features and adjacency matrices to tensors and pack them
        # sps已经包含了节点特征和邻接矩阵
        # features = [sp['features'] for sp in sps]
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps) # 节点特征
        # adjacencies = [sp['adjacency'] for sp in sps]
        adjacencies = np.array(adjs) # 所有样本的邻接矩阵
        lbs = np.array(lbs) # 样本标签
        features_tensor = torch.Tensor(features)
        adjacencies_tensor = torch.Tensor(adjacencies)
        labels_tensor = torch.LongTensor(lbs)

        # Create a dataset and loader
        dataset = TensorDataset(features_tensor, adjacencies_tensor, labels_tensor)
        loader = DataLoader(dataset, batch_size=len(features_tensor), shuffle=shuffle)

        # Initialize the model
        model = DeepLion2_GATv2(tcr_num=tcr_num, feature_dim=768, drop_out=dropout).to(device) # 这里是模型的init，需要修改
        optimizer = optim.Adam(model.parameters(), lr=lr) # 优化器
        criterion = nn.CrossEntropyLoss().to(device) # 损失函数

        # Check if a model file already exists and adjust the filename
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        valid_model_f = model_f + "temp.pth"

        # Training loop
        for epoch in range(ep):
            for batch_x, batch_adj, batch_y in loader:
                batch_x, batch_adj, batch_y = batch_x.to(device), batch_adj.to(device), batch_y.to(device)

                # print(f"batch_x:{batch_x.shape}") # torch.Size([8, 100, 768])
                # print(f"batch_adj:{batch_adj.shape}") # torch.Size([8, 100, 100])
                # print(f"batch_y:{batch_y.shape}")
                # Forward pass
                outputs = model(batch_x, batch_adj) # 这里是前向传播，主要的改动在这里
                # print(f"outputs:{outputs.shape}")
                # print(outputs)
                # print(batch_y)
                loss = criterion(outputs, batch_y) # 损失函数也要改
                print(f"epoch:{epoch}, loss = {loss}")

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Logging
                if (epoch + 1) % log_inr == 0:
                    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss.item()))

            print("valid_sps:", valid_sps)
            # Optional validation step
            # 目前训练时没有验证，这里可以加入early stopping
            if valid_sps is not None and valid_lbs is not None and valid_mat is not None:
                valid_acc, valid_auc, valid_f1 = DeepLION2_GAT.validate_model(model, valid_sps, valid_lbs, valid_mat, device)
                print('Validation Accuracy:', valid_acc)
                print('Validation AUC:', valid_auc)
                print('Validation f1 score:', valid_f1)
                # Save the best model
                if valid_acc > max_acc:
                    max_acc = valid_acc
                    torch.save(model.state_dict(), valid_model_f)
                    if os.path.exists(model_f):
                        os.remove(model_f)
                    os.rename(valid_model_f, model_f)
                else:
                    os.remove(valid_model_f)

        # Save the final model
        torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)

        return 0

    @staticmethod
    def prediction(sps, dismat, model_f, tcr_num, device):
        # 加载模型
        model = DeepLION2_GAT(tcr_num=tcr_num, feature_dim=768, attention_head_num=1, attention_hidden_size=100, drop_out=0.4).to(device)
        model.load_state_dict(torch.load(model_f, map_location=device))
        model.eval()

        # 准备测试数据
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 计算节点特征
        adjacencies = np.array(dismat)
        features_tensor = torch.Tensor(features).to(device)
        adjacencies_tensor = torch.Tensor(adjacencies).to(device)

        # forward
        with torch.no_grad():
            outputs = model(features_tensor, adjacencies_tensor)

        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 获取类别1的概率

        return probs    
    
class GraphAttentionLayer2(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        """
            in_features：输入特征的维度
            out_features：输出特征的维度
            alpha：LeakyReLU激活函数的负斜率参数
            concat：一个布尔值，决定是否在多头注意力中合并输出
        """
        super(GraphAttentionLayer2, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # 初始化一个形状为（in_features, out_features）的权重矩阵W，用于将输入特征转换到新的特征空间
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # 初始化一个形状为（2 * out_features, 1）的参数向量a，用于计算注意力系数
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # 定义一个LeakyReLU激活函数，负斜率由alpha参数提供
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
            h：输入的节点特征矩阵，形状为(N, in_features)
            adj：邻接矩阵，表示图中节点的连接关系
        """
        # 使用权重矩阵W对输入特征矩阵h进行线性变换，得到新的特征矩阵Wh
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # 使用Wh和参数向量a计算原始的注意力系数e
        e = self._prepare_attentional_mechanism_input(Wh)

        # 使用torch.where将邻接矩阵中不连接的节点对的注意力系数设置为一个非常小的值（接近负无穷），以便在应用softmax时这些位置的值接近0。
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        # 对注意力系数进行softmax归一化，使得每个节点的所有输入注意力系数之和为1。
        attention = F.softmax(attention, dim=1)
        # 对归一化后的注意力系数应用dropout。
        attention = F.dropout(attention, self.dropout, training=self.training)
        # 使用注意力权重的加权和计算新的节点特征矩阵h_prime
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    # 计算注意力机制的输入
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        # 分别计算Wh和参数向量a的前半部分和后半部分的矩阵乘积，得到Wh1和Wh2。
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        # 将Wh1和Wh2的转置相加，应用LeakyReLU激活函数。
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class DeepLION2_GCN(nn.Module):
    def __init__(self, tcr_num=200, feature_dim=768, nhid=128, dropout_rate=0.4, pooling_ratio=0.20):
        super(DeepLION2_GCN, self).__init__()
        self.graph_layer = GraphConvolutionLayer(feature_dim, nhid, dropout=dropout_rate, pooling_ratio=pooling_ratio)
        self.out_fc = nn.Linear(nhid * 2, 2)  # 注意调整这里的输入维度以适应池化层后的输出
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, adj):
        x, adj = self.graph_layer(x, adj)
        x = torch.mean(x, dim=1)  # 如果是批量处理，确保这里正确处理维度
        x = self.out_fc(x)
        return x
    
    @staticmethod
    def training(sps, lbs, adjs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, shuffle=False, valid_sps=None, valid_lbs=None, valid_mat=None):
        # Assume sps includes both features and adjacency matrices for each sample
        tcrbert = TCR_Bert(model_path="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/model_path/", src_dir="/home/cc/cc/TCR/tcr-bert/tcr-bert-main/tcr/")
        features = tcrbert.process_sps(sps)  # 获取特征
        adjacencies = np.array(adjs)
        lbs = np.array(lbs)
        features_tensor = torch.Tensor(features)
        adjacencies_tensor = torch.Tensor(adjacencies)
        labels_tensor = torch.LongTensor(lbs)

        dataset = TensorDataset(features_tensor, adjacencies_tensor, labels_tensor)
        loader = DataLoader(dataset, batch_size=100, shuffle=shuffle)

        model = DeepLION2_GCN(tcr_num=tcr_num, feature_dim=768, dropout_rate=dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss().to(device)

        # Check and handle model file operations
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        valid_model_f = model_f + "temp.pth"

        max_acc = 0
        for epoch in range(ep):
            for batch_x, batch_adj, batch_y in loader:
                batch_x, batch_adj, batch_y = batch_x.to(device), batch_adj.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_x, batch_adj)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % log_inr == 0:
                    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss.item()))

            if valid_sps is not None and valid_lbs is not None and valid_mat is not None:
                valid_acc, valid_auc, valid_f1 = DeepLION2_GCN.validate_model(model, valid_sps, valid_lbs, valid_mat, device)
                print('Validation Accuracy:', valid_acc)
                print('Validation AUC:', valid_auc)
                print('Validation f1 score:', valid_f1)
                if valid_acc > max_acc:
                    max_acc = valid_acc
                    torch.save(model.state_dict(), valid_model_f)
                    if os.path.exists(model_f):
                        os.remove(model_f)
                    os.rename(valid_model_f, model_f)
                else:
                    os.remove(valid_model_f)

        torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)

        return 0

# # 自定义的可以处理边权重的SAGEConv类
# class CustomSAGEConv(SAGEConv):
#     def __init__(self, in_channels, out_channels):
#         super(CustomSAGEConv, self).__init__(in_channels, out_channels, normalize=False, root_weight=False)

#     def forward(self, x, edge_index, edge_weight=None, size=None):
#         edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1, num_nodes=x.size(0))
#         if edge_weight is not None:
#             edge_weight = softmax(edge_weight, edge_index[0])
#         return super().forward(x, edge_index, edge_weight, size=size)

# class GraphConvolutionLayer(nn.Module):
#     def __init__(self, in_features, nhid, dropout=0.5, pooling_ratio=0.20):
#         super(GraphConvolutionLayer, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.conv1 = CustomSAGEConv(in_features, nhid)
#         self.pool1 = SAGPooling(nhid, ratio=pooling_ratio)
#         self.conv2 = CustomSAGEConv(nhid, nhid)
#         self.pool2 = SAGPooling(nhid, ratio=pooling_ratio)
#         self.conv3 = CustomSAGEConv(nhid, nhid)
#         self.pool3 = SAGPooling(nhid, ratio=pooling_ratio)

#     def forward(self, x, adj, batch=None):
#         results = []
#         for i in range(adj.shape[0]):  # iterate over each graph
#             single_adj = adj[i]
#             edge_index, edge_weight = dense_to_sparse(single_adj)
#             single_x = x[i]  # 直接获取当前图的节点特征

#             single_x = self.dropout(single_x)

#             size = (single_x.size(0), single_x.size(0))

#             # Apply graph convolution and pooling
#             single_x = self.conv1(single_x, edge_index, edge_weight, size=size)
#             single_x, edge_index, _, batch, _ = self.pool1(single_x, edge_index, None, batch)
#             x1 = torch.cat([gmp(single_x, batch), gap(single_x, batch)], dim=1) if batch is not None else single_x

#             single_x = self.conv2(single_x, edge_index, edge_weight, size=size)
#             single_x, edge_index, _, batch, _ = self.pool2(single_x, edge_index, None, batch)
#             x2 = torch.cat([gmp(single_x, batch), gap(single_x, batch)], dim=1) if batch is not None else single_x

#             single_x = self.conv3(single_x, edge_index, edge_weight, size=size)
#             single_x, edge_index, _, batch, _ = self.pool3(single_x, edge_index, None, batch)
#             x3 = torch.cat([gmp(single_x, batch), gap(single_x, batch)], dim=1) if batch is not None else single_x

#             single_x = x1 + x2 + x3
#             results.append(single_x)

#         final_x = torch.stack(results, dim=0)  # Stack to recreate batch dimension
#         return final_x, edge_index



# class GraphConvolutionLayer(nn.Module):
#     def __init__(self, in_features, nhid, dropout=0.5, pooling_ratio=0.20):
#         super(GraphConvolutionLayer, self).__init__()
#         self.dropout = nn.Dropout(dropout)

#         # Define the graph convolution and pooling layers
#         self.conv1 = SAGEConv(in_features, nhid)
#         self.pool1 = SAGPooling(nhid, ratio=pooling_ratio)
#         self.conv2 = SAGEConv(nhid, nhid)
#         self.pool2 = SAGPooling(nhid, ratio=pooling_ratio)
#         self.conv3 = SAGEConv(nhid, nhid)
#         self.pool3 = SAGPooling(nhid, ratio=pooling_ratio)

#     def forward(self, x, adj, batch=None):
#         x = self.dropout(x)
        
#         # First graph convolution and pooling
#         x = F.relu(self.conv1(x, adj)) # 这一步中，SAGEConv不接受邻接矩阵
#         x, adj, _, batch, _ = self.pool1(x, adj, None, batch)
#         x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) if batch is not None else x

#         # Second graph convolution and pooling
#         x = F.relu(self.conv2(x, adj))
#         x, adj, _, batch, _ = self.pool2(x, adj, None, batch)
#         x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) if batch is not None else x

#         # Third graph convolution and pooling
#         x = F.relu(self.conv3(x, adj))
#         x, adj, _, batch, _ = self.pool3(x, adj, None, batch)
#         x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) if batch is not None else x

#         # Combine features from all levels
#         x = x1 + x2 + x3

#         return x, adj