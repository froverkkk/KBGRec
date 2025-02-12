from torch.utils.data  import Dataset


class Eval_dataset(Dataset):
    def __init__(self, conf, user_set, ui_pair_dict):
        super(Eval_dataset, self).__init__()
        self.conf = conf
        self.user_list = list(user_set)
        self.item_list = self.get_item_label_list(ui_pair_dict)

    def get_item_label_list(self, ui_pair_dict):
        item_list = []
        for user in self.user_list:
            user_item_list = list(ui_pair_dict[user])
            item_list.append(user_item_list)
        return item_list

    def __getitem__(self, idx):
        return self.user_list[idx], self.item_list[idx]

    def __len__(self):
        return len(self.user_list)


class Train_dataset(Dataset):
    def __init__(self, conf, ui_pair_set):
        super(Train_dataset, self).__init__()
        self.conf = conf
        self.ui_pair_list = list(ui_pair_set)

    def __getitem__(self, idx):
        return self.ui_pair_list[idx]

    def __len__(self):
        return len(self.ui_pair_list)
