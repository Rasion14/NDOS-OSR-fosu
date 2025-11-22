import json
import os
import random
import copy
import csv
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm,trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn.cluster import KMeans
import torch
from torch.distributions import Beta
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss,MSELoss,BCEWithLogitsLoss
from transformers import (
    BertModel,
    BertPreTrainedModel,
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup)
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler
)
from torch.optim import AdamW,Adam
from datetime import datetime


def F_measure(cm):
    idx = 0
    rs, ps, fs = [], [], []
    n_class = cm.shape[0]

    for idx in range(n_class):
        TP = cm[idx][idx]
        r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0
        rs.append(r * 100)
        ps.append(p * 100)
        fs.append(f * 100)

    f = np.mean(fs).round(4)
    f_seen = np.mean(fs[:-1]).round(4)
    f_unseen = round(fs[-1], 4)
    r_unseen = round(rs[-1],4)
    p_unseen = round(ps[-1],4)
    result = {}
    result['F1_Known'] = f_seen
    result['F1_Open'] = f_unseen
    result['R_Open'] = r_unseen
    result['P_Open'] = p_unseen
    result['F1-score'] = f

    return result
