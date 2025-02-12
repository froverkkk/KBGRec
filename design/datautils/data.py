from datautils.data_load import *
from datautils.dataset import *
from datautils.utils import *
from torch.utils.data import DataLoader
import numpy as np

class Data:
    def __init__(self, conf):
        self.conf = conf
        self.num_users = conf.num_users
        self.num_items = conf.num_items
        if conf.data_name == 'ciao' or conf.data_name == 'epinions':
            self.all_user_pair_set, self.all_ui_pair_dict = load_data_v1(self.conf)
        elif conf.data_name == 'yelp' or conf.data_name == 'flickr':
            self.all_user_pair_set, self.all_ui_pair_dict = load_data_v2(self.conf)
        else:
            raise NotImplementedError('Wrong data name')
        self.all_ui_pair_set = get_pair_set_from_pair_dict(self.all_ui_pair_dict)

        self.user_set = set()
        self.item_set = set()
        for user, item in self.all_ui_pair_set:
            self.user_set.add(user)
            self.item_set.add(item)
        print('Num users', len(self.user_set))
        print('Num items', len(self.item_set))

        self.train_ui_pair_dict, self.valid_ui_pair_dict, self.test_ui_pair_dict = \
            split_pair_dict_random_ratio(self.all_ui_pair_set, 0.15, 0.05, conf.train_ratio)

        self.train_user_set = set(self.train_ui_pair_dict.keys())
        self.valid_user_set = set(self.valid_ui_pair_dict.keys())
        self.test_user_set = set(self.test_ui_pair_dict.keys())
        self.train_ui_pair_set = get_pair_set_from_pair_dict(self.train_ui_pair_dict)
        self.valid_ui_pair_set = get_pair_set_from_pair_dict(self.valid_ui_pair_dict)
        self.test_ui_pair_set = get_pair_set_from_pair_dict(self.test_ui_pair_dict)
        self.train_iu_pair_set = reverse_pair_set(self.train_ui_pair_set)

        print('Num users: train:{}, valid:{}, test:{}'.format(len(self.train_ui_pair_dict),
                                                              len(self.valid_ui_pair_dict),
                                                              len(self.test_ui_pair_dict)))
        print('Num ui pairs: train:{}, valid:{}, test:{}'.format(len(self.train_ui_pair_set),
                                                                 len(self.valid_ui_pair_set),
                                                                 len(self.test_ui_pair_set)))

        self.data_graph = construct_hetero_graphs(
            [self.all_user_pair_set, self.train_ui_pair_set, self.train_iu_pair_set],
            [('user', 'friend', 'user'), ('user', 'like', 'item'), ('item', 'rev_like', 'user')],
            {'user': conf.num_users, 'item': conf.num_items}
        ).to(self.conf.device)

        self.train_ui_pair2eid = self.get_pair2eid(self.train_ui_pair_set)
        self.train_iu_pair2eid = self.get_pair2eid(self.train_iu_pair_set)

        # train
        self.train_dataset = Train_dataset(conf, self.train_ui_pair_set)
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=conf.train_batch_size,
                                            collate_fn=self.collate_train, shuffle=True)

        # eval positive
        self.train_eval_user_idx_dict, self.train_eval_user_list, self.train_eval_item_list = \
            self.get_eval_positive_dataset(self.train_ui_pair_set)
        self.valid_eval_user_idx_dict, self.valid_eval_user_list, self.valid_eval_item_list = \
            self.get_eval_positive_dataset(self.valid_ui_pair_set)
        self.test_eval_user_idx_dict, self.test_eval_user_list, self.test_eval_item_list = \
            self.get_eval_positive_dataset(self.test_ui_pair_set)

        # eval negative
        self.train_data_dict = Eval_dataset(conf, self.train_user_set, self.train_ui_pair_dict)
        self.valid_data_dict = Eval_dataset(conf, self.valid_user_set, self.valid_ui_pair_dict)
        self.test_data_dict = Eval_dataset(conf, self.test_user_set, self.test_ui_pair_dict)
        self.train_neg_data_loader_train = DataLoader(self.train_data_dict, batch_size=conf.train_batch_size,
                                                      collate_fn=self.collate_eval_neg_train, shuffle=False)
        self.valid_neg_data_loader_train = DataLoader(self.valid_data_dict, batch_size=conf.train_batch_size,
                                                      collate_fn=self.collate_eval_neg_train, shuffle=False)
        self.test_neg_data_loader_train = DataLoader(self.test_data_dict, batch_size=conf.train_batch_size,
                                                     collate_fn=self.collate_eval_neg_train, shuffle=False)

        self.train_neg_data_loader_test = DataLoader(self.train_data_dict, batch_size=conf.test_batch_size,
                                                     collate_fn=self.collate_eval_neg_test, shuffle=False)
        self.valid_neg_data_loader_test = DataLoader(self.valid_data_dict, batch_size=conf.test_batch_size,
                                                     collate_fn=self.collate_eval_neg_test, shuffle=False)
        self.test_neg_data_loader_test = DataLoader(self.test_data_dict, batch_size=conf.test_batch_size,
                                                    collate_fn=self.collate_eval_neg_test, shuffle=False)
        print('Finish data initialize')

    def get_positive_social_pairs(self):

        pos_pairs = np.array(list(self.all_user_pair_set))

        np.random.shuffle(pos_pairs)
        return pos_pairs

    def get_negative_social_pairs(self, num_negatives=None):

        if num_negatives is None:

            num_negatives = len(self.all_user_pair_set)

        neg_pairs = []
        existing_pairs = self.all_user_pair_set

        while len(neg_pairs) < num_negatives:

            u1 = np.random.randint(0, self.num_users)
            u2 = np.random.randint(0, self.num_users)
            if u1 == u2:
                continue

            if (u1, u2) not in existing_pairs and (u2, u1) not in existing_pairs:
                neg_pairs.append([u1, u2])

        neg_pairs = np.array(neg_pairs)
        return neg_pairs


    def get_social_train_batch(self):

        pos_pairs = self.get_positive_social_pairs()
        neg_pairs = self.get_negative_social_pairs()

        all_pairs = np.concatenate([pos_pairs, neg_pairs])
        all_labels = np.concatenate([
            np.ones(len(pos_pairs)),
            np.zeros(len(neg_pairs))
        ])

        indices = np.arange(len(all_pairs))
        np.random.shuffle(indices)

        for start_idx in range(0, len(indices), self.conf.pretrain_batch_size):
            batch_indices = indices[start_idx:start_idx + self.conf.pretrain_batch_size]
            batch_pairs = all_pairs[batch_indices]
            batch_labels = all_labels[batch_indices]

            yield batch_pairs[:, 0], batch_pairs[:, 1], batch_labels

    @staticmethod
    def read_social_data(social_data):
        user_pair_set = set()
        for row in range(social_data.shape[0]):
            u1_id, u2_id = int(social_data[row][0]) - 1, int(social_data[row][1]) - 1
            user_pair_set.add((u1_id, u2_id))
        return user_pair_set

    @staticmethod
    def read_rating_data(rating_data):
        ui_pair_dict = defaultdict(set)
        for row in range(rating_data.shape[0]):
            rating_item = rating_data[row]
            user_id, item_id = int(rating_item[0]) - 1, int(rating_item[1]) - 1,
            category, score = int(rating_item[2]), int(rating_item[3])
            ui_pair_dict[user_id].add(item_id)
        return ui_pair_dict

    # generate train data
    def collate_train(self, samples):
        user_list, item_list = map(list, zip(*samples))
        collected_user_list = []
        collected_item_list = []
        collected_label_list = []
        for idx in range(len(user_list)):
            collected_user_list.append(user_list[idx])
            collected_item_list.append(item_list[idx])
            collected_label_list.append(1)
            collected_user_list.extend([user_list[idx]] * self.conf.num_train_negatives)
            for _ in range(self.conf.num_train_negatives):
                negative_item_id = np.random.randint(self.conf.num_items)
                while (user_list[idx], negative_item_id) in self.train_ui_pair_set:
                    negative_item_id = np.random.randint(self.conf.num_items)
                collected_item_list.append(negative_item_id)
            collected_label_list.extend([0] * self.conf.num_train_negatives)

        return collected_user_list, collected_item_list, collected_label_list

    # generate eval data
    @staticmethod
    def get_eval_positive_dataset(ui_pair_set):
        user_list = []
        item_list = []
        user_idx_dict = defaultdict(list)
        for idx, (user_id, item_id) in enumerate(ui_pair_set):
            user_list.append(user_id)
            item_list.append(item_id)
            user_idx_dict[user_id].append(idx)
        return user_idx_dict, user_list, item_list

    def collate_eval_neg_train(self, samples):
        user_list, item_list = map(list, zip(*samples))
        collected_user_list = []
        collected_item_list = []
        for idx in range(len(user_list)):
            collected_user_list.extend([user_list[idx]] * self.conf.num_eval_negatives)
            for _ in range(self.conf.num_eval_negatives):
                negative_item_id = np.random.randint(self.conf.num_items)
                while (user_list[idx], negative_item_id) in self.all_ui_pair_set:
                    negative_item_id = np.random.randint(self.conf.num_items)
                collected_item_list.append(negative_item_id)
        return user_list, collected_user_list, collected_item_list

    def collate_eval_neg_test(self, samples):
        user_list, item_list = map(list, zip(*samples))
        collected_user_list = []
        collected_item_list = []
        for idx in range(len(user_list)):
            collected_user_list.extend([user_list[idx]] * self.conf.num_test_negatives)
            for _ in range(self.conf.num_test_negatives):
                negative_item_id = np.random.randint(self.conf.num_items)
                while (user_list[idx], negative_item_id) in self.all_ui_pair_set:
                    negative_item_id = np.random.randint(self.conf.num_items)
                collected_item_list.append(negative_item_id)

        return user_list, collected_user_list, collected_item_list

    @staticmethod
    def get_pair2eid(pair_list):
        pair2eid = {}
        for eid, pair in enumerate(pair_list):
            pair2eid[pair] = eid
        return pair2eid
