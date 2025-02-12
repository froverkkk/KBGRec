import datetime
import numpy as np
import random
import torch
import dgl
import math
from collections import defaultdict
from tqdm import tqdm
import gc


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_date_postfix():
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix


def get_source_target_form_pair_set(pair_set):
    source_ids = []
    target_ids = []
    for id_1, id_2 in pair_set:
        source_ids.append(id_1)
        target_ids.append(id_2)
    return source_ids, target_ids


def reverse_pair_set(pair_set):
    rev_pair_set = set()
    for (id_1, id_2) in pair_set:
        rev_pair_set.add((id_2, id_1))
    return rev_pair_set


def reverse_pair_list(pair_list):
    rev_pair_list = list()
    for (id_1, id_2) in pair_list:
        rev_pair_list.append((id_2, id_1))
    return rev_pair_list


def construct_hetero_graphs(pair_sets, edge_types, num_nodes_dict):
    data_dict = {}
    for idx, pair_set in enumerate(pair_sets):
        source_ids, target_ids = get_source_target_form_pair_set(pair_set)
        data_dict[edge_types[idx]] = (source_ids, target_ids)
    graph = dgl.heterograph(data_dict, num_nodes_dict)
    return graph


def fuse_dict_with_set(dict_list):
    ret_dict = defaultdict(set)
    for dic in dict_list:
        for key, val in dic.items():
            ret_dict[key] = ret_dict[key] | val
    return ret_dict


def split_pair_dict_thr(pair_dict, thr_1, thr_2):
    train_dict = defaultdict(set)
    valid_dict = defaultdict(set)
    test_dict = defaultdict(set)
    for key, val in pair_dict.items():
        if len(val) >= thr_1:
            train_dict[key] = val
        elif len(val) >= thr_2:
            valid_dict[key] = val
        else:
            test_dict[key] = val

    return train_dict, valid_dict, test_dict


def split_pair_dict_user_ratio(pair_dict, test_ratio, valid_ratio, train_ratio):
    assert test_ratio + valid_ratio + train_ratio <= 1.0
    train_dict = defaultdict(set)
    valid_dict = defaultdict(set)
    test_dict = defaultdict(set)
    for key, val in pair_dict.items():
        user_item_count = len(val)
        thr_0 = int(user_item_count * test_ratio)
        thr_1 = int(user_item_count * (test_ratio + valid_ratio))
        thr_2 = int(user_item_count * (test_ratio + valid_ratio + train_ratio))
        item_list = list(val)
        random.shuffle(item_list)

        test_item_set = set(item_list[: thr_0])
        valid_item_set = set(item_list[thr_0: thr_1])
        train_item_set = set(item_list[thr_1: thr_2])

        test_dict[key] = test_item_set
        valid_dict[key] = valid_item_set
        train_dict[key] = train_item_set

    return train_dict, valid_dict, test_dict


def split_pair_dict_random_ratio(pair_set, test_ratio, valid_ratio, train_ratio):
    set_random_seed(2021)
    assert test_ratio + valid_ratio + train_ratio <= 1.0
    train_dict = defaultdict(set)
    valid_dict = defaultdict(set)
    test_dict = defaultdict(set)
    for user, item in pair_set:
        thr_0 = test_ratio
        thr_1 = test_ratio + valid_ratio
        thr_2 = test_ratio + valid_ratio + train_ratio

        rand_val = random.uniform(0.0, 1.0)
        if 0 <= rand_val < thr_0:
            test_dict[user].add(item)
        elif thr_0 <= rand_val < thr_1:
            valid_dict[user].add(item)
        elif thr_1 <= rand_val < thr_2:
            train_dict[user].add(item)

    return train_dict, valid_dict, test_dict


def get_pair_set_from_pair_dict(pair_dict):
    pair_set = set()
    for key, val in pair_dict.items():
        for pair in val:
            pair_set.add((key, pair))
    return pair_set


def get_statistic_list(sim_user_pair_list, raw_user_pair):
    print('Begin')
    x_point = np.linspace(start=0, stop=len(sim_user_pair_list), num=1000, endpoint=True)
    sim_list = []
    social_list = []
    val_list = []
    cur_idx = 0
    cur_point_idx = 1
    cur_sim_count = 0
    for val, (f_user, s_user) in tqdm(sim_user_pair_list):
        if (f_user, s_user) in raw_user_pair:
            cur_sim_count += 1
        cur_idx += 1
        if cur_idx >= x_point[cur_point_idx]:
            # print(len(x_point), cur_point_idx, x_point[-1], cur_idx)
            # print(val)
            cur_point_idx += 1
            sim_list.append(cur_sim_count)
            social_list.append(cur_idx)
            val_list.append(val)

    return sim_list, social_list, val_list


def construct_iu_pair_dict(ui_pair_dict):
    iu_pair_dict = defaultdict(set)
    for user, user_item_set in ui_pair_dict.items():
        for item in user_item_set:
            iu_pair_dict[item].add(user)
    return iu_pair_dict


def get_u2i_norm_dict(ui_pair_dict, iu_pair_dict, norm_mode='mean'):
    user_norm_dict = defaultdict(float)
    item_norm_dict = defaultdict(float)
    if norm_mode == 'mean':
        for user, user_item_set in ui_pair_dict.items():
            user_norm_dict[user] = 1
        for item, item_user_set in iu_pair_dict.items():
            item_norm_dict[item] = 1 / float(len(item_user_set))
    elif norm_mode == 'raw':
        for user, user_item_set in ui_pair_dict.items():
            user_norm_dict[user] = 1 / math.sqrt(float(len(user_item_set)))
        for item, item_user_set in iu_pair_dict.items():
            item_norm_dict[item] = 1 / math.sqrt(float(len(item_user_set)))
    else:
        raise NotImplementedError('Invalid norm mode')
    return user_norm_dict, item_norm_dict


def get_i2u_norm_dict(ui_pair_dict, iu_pair_dict, norm_mode='mean'):
    user_norm_dict = defaultdict(float)
    item_norm_dict = defaultdict(float)
    if norm_mode == 'mean':
        for item, item_user_set in iu_pair_dict.items():
            item_norm_dict[item] = 1
        for user, user_item_set in ui_pair_dict.items():
            user_norm_dict[user] = 1 / float(len(user_item_set))
    elif norm_mode == 'raw':
        for item, item_user_set in iu_pair_dict.items():
            item_norm_dict[item] = 1 / math.sqrt(float(len(item_user_set)))
        for user, user_item_set in ui_pair_dict.items():
            user_norm_dict[user] = 1 / math.sqrt(float(len(user_item_set)))
    else:
        raise NotImplementedError('Invalid norm mode')
    return user_norm_dict, item_norm_dict


def construct_ui_sparse_matrix(ui_pair_dict, user_norm_dict, item_norm_dict, num_user, num_item):
    user_node_list = []
    item_node_list = []
    val_list = []
    for user, user_item_set in ui_pair_dict.items():
        user_norm = user_norm_dict[user]
        for item in user_item_set:
            item_norm = item_norm_dict[item]
            user_node_list.append(user)
            item_node_list.append(item)
            val_list.append(user_norm * item_norm)
    ui_sparse_matrix = torch.sparse_coo_tensor([user_node_list, item_node_list], val_list, (num_user, num_item))
    return ui_sparse_matrix


def construct_iu_sparse_matrix(iu_pair_dict, user_norm_dict, item_norm_dict, num_user, num_item):
    item_node_list = []
    user_node_list = []
    val_list = []
    for item, item_user_set in iu_pair_dict.items():
        item_norm = item_norm_dict[item]
        for user in item_user_set:
            user_norm = user_norm_dict[user]
            item_node_list.append(item)
            user_node_list.append(user)
            val_list.append(item_norm * user_norm)
    iu_sparse_matrix = torch.sparse_coo_tensor([item_node_list, user_node_list], val_list, (num_item, num_user))
    return iu_sparse_matrix


def get_sorted_user_pair_list(user_pair, user_pair_val):
    user_pair_list = list(zip(user_pair_val, list(zip(user_pair[0], user_pair[1]))))
    random.shuffle(user_pair_list)
    sorted_user_pair_list = sorted(user_pair_list, key=lambda x: x[0], reverse=True)
    return sorted_user_pair_list


def out_file(data_list, path):
    with open(path, 'w') as f:
        for data in data_list:
            f.write(str(data) + '\n')

