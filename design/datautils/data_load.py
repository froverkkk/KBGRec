import os
from scipy.io import loadmat
from datautils.utils import *


def load_data_v1(conf):
    data_rating = loadmat(os.path.join(conf.data_dir, conf.data_name, 'rating.mat'))
    data_social = loadmat(os.path.join(conf.data_dir, conf.data_name, 'trustnetwork.mat'))
    user_pair_set = read_social_data_v1(data_social['trustnetwork'])
    reverse_user_pair_set = reverse_pair_set(user_pair_set)
    all_user_pair_set = user_pair_set | reverse_user_pair_set
    all_ui_pair_dict = read_rating_data_v1(data_rating['rating'])

    return all_user_pair_set, all_ui_pair_dict


def read_social_data_v1(social_data):
    user_pair_set = set()
    for row in range(social_data.shape[0]):
        u1_id, u2_id = int(social_data[row][0]) - 1, int(social_data[row][1]) - 1
        user_pair_set.add((u1_id, u2_id))
    return user_pair_set


def read_rating_data_v1(rating_data):
    ui_pair_dict = defaultdict(set)
    for row in range(rating_data.shape[0]):
        rating_item = rating_data[row]
        user_id, item_id = int(rating_item[0]) - 1, int(rating_item[1]) - 1,
        category, score = int(rating_item[2]), int(rating_item[3])
        ui_pair_dict[user_id].add(item_id)
    return ui_pair_dict


def load_data_v2(conf):
    dir_path = os.path.join(conf.data_dir, conf.data_name)
    social_filename = os.path.join(dir_path, conf.data_name + '.links')
    rating_filename = os.path.join(dir_path, conf.data_name + '.rating')
    user_pair_set = read_social_data_v2(social_filename)
    reverse_user_pair_set = reverse_pair_set(user_pair_set)
    all_user_pair_set = user_pair_set | reverse_user_pair_set
    all_ui_pair_dict = read_rating_data_v2(rating_filename)

    return all_user_pair_set, all_ui_pair_dict


def read_social_data_v2(filename):
    with open(filename) as f:
        user_pair_set = set()
        for line in f:
            arr = line.split("\t")
            u1_id, u2_id = int(arr[0]), int(arr[1])
            user_pair_set.add((u1_id, u2_id))
        return user_pair_set


def read_rating_data_v2(filename):
    with open(filename) as f:
        ui_pair_dict = defaultdict(set)
        for line in f:
            arr = line.split("\t")
            user_id, item_id = int(arr[0]), int(arr[1])
            ui_pair_dict[user_id].add(item_id)
        return ui_pair_dict
