import numpy as np
import math
import torch
import torch.nn.functional as F


class EarlyStopping(object):
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_metric = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, criteria):
        is_best = False
        if self.best_loss is None:
            is_best = True
            self.best_metric = criteria
            self.best_loss = loss
        elif criteria < self.best_metric:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if criteria > self.best_metric:
                is_best = True
            self.best_loss = np.min((loss, self.best_loss))
            self.best_metric = np.max((criteria, self.best_metric))
            self.counter = 0
        return self.early_stop, is_best


class Evaluate:
    def __init__(self, max_length):
        self.idcg_list = self.get_idcg_list(max_length)

    @staticmethod
    def get_idcg_list(max_length):
        idcg_list = [0]
        idcg = 0.0
        for idx in range(max_length):
            idcg = idcg + math.log(2) / math.log(idx + 2)
            idcg_list.append(idcg)
        return idcg_list

    def get_hit_ndcg(self, positive_predict_list, negative_predict_list, top_k):
        positive_predict_list = list(positive_predict_list)
        negative_predict_list = list(negative_predict_list)
        positive_length = len(positive_predict_list)
        target_length = min(positive_length, top_k)

        all_predict_list = positive_predict_list
        all_predict_list.extend(negative_predict_list)
        sort_index = np.argsort(all_predict_list)[::-1]

        hit_k = 0.0
        dcg_k = 0.0
        for idx in range(min(len(sort_index), top_k)):
            ranking = sort_index[idx]
            if ranking < positive_length:
                hit_k = hit_k + 1.0
                dcg_k = dcg_k + math.log(2) / math.log(idx + 2)

        hit_k = hit_k / target_length
        idcg = self.idcg_list[target_length]
        ndcg_k = dcg_k / idcg

        return hit_k, ndcg_k


def compute_pre_distill_loss(pre_a, pre_b):
    distill_loss = - torch.mean(pre_b.detach() * torch.log(pre_a) + (1 - pre_b.detach()) * torch.log(1 - pre_a))
    return distill_loss
