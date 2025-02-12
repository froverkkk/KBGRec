import time
import os
from model.socialgcn import SocialGCN
from model.twingcn import TwinGCN
from model.ratinggcn import RatingGCN
from model.model_utils import *
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Process, Queue
import numpy as np
from datetime import datetime

class Trainer:
    def __init__(self, conf, data):
        self.conf = conf
        self.data = data
        self.model_dict = self.get_model_dict(conf.model_list)
        self.evaluate = Evaluate(conf.top_k)

        if hasattr(conf, 'do_pretrain') and conf.do_pretrain:
            self.pretrain_social_model = SocialGCN(conf, data).to(conf.device)
            self.pretrain_rating_model = RatingGCN(conf, data).to(conf.device)
            self.pretrain_social_optimizer = torch.optim.Adam(
                self.pretrain_social_model.parameters(),
                lr=conf.pretrain_lr
            )
            self.pretrain_rating_optimizer = torch.optim.Adam(
                self.pretrain_rating_model.parameters(),
                lr=conf.pretrain_lr
            )

        self.results_dir = os.path.join(self.conf.cur_out_path, 'results')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)


    def pretrain_social_view(self, epochs=10):

        print("Start pretraining social view...")
        for epoch in range(epochs):
            self.pretrain_social_model.train()
            total_loss = 0
            num_batches = 0

            for users1, users2, labels in self.data.get_social_train_batch():
                users1 = torch.LongTensor(users1).to(self.conf.device)
                users2 = torch.LongTensor(users2).to(self.conf.device)
                labels = torch.FloatTensor(labels).to(self.conf.device)

                self.pretrain_social_optimizer.zero_grad()

                predict, _, _ = self.pretrain_social_model(users1, users2, mode='train')

                if isinstance(predict, tuple):
                    predict = predict[0]

                loss = self.model_dict['social']['criterion'](predict, labels)

                loss.backward()
                self.pretrain_social_optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Social View Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    def pretrain_rating_view(self, epochs=10):
        print("Start pretraining rating view...")
        for epoch in range(epochs):
            self.pretrain_rating_model.train()
            total_loss = 0
            num_batches = 0

            for users, items, labels in self.data.train_data_loader:
                users = torch.LongTensor(users).to(self.conf.device)
                items = torch.LongTensor(items).to(self.conf.device)
                labels = torch.FloatTensor(labels).to(self.conf.device)

                self.pretrain_rating_optimizer.zero_grad()

                predict, _, _ = self.pretrain_rating_model(users, items, users_list=users.cpu().numpy(),
                                                           items_list=items.cpu().numpy(), mode='train')


                if isinstance(predict, tuple):
                    predict = predict[0]

                loss = self.model_dict['rating']['criterion'](predict, labels)

                loss.backward()
                self.pretrain_rating_optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Rating View Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
    def load_pretrained_params(self):

        twin_model = self.model_dict['twin']['model']

        social_dict = {}
        for name, param in self.pretrain_social_model.named_parameters():
            if 'social_layer' in name:
                social_dict[name] = param

        rating_dict = {}
        for name, param in self.pretrain_rating_model.named_parameters():
            if 'rating_layer' in name:
                rating_dict[name] = param

        twin_model.load_state_dict(social_dict, strict=False)
        twin_model.load_state_dict(rating_dict, strict=False)

    def train(self):

        if hasattr(self.conf, 'do_pretrain') and self.conf.do_pretrain:
            print("Starting pretraining phase...")
            self.pretrain_social_view(epochs=self.conf.pretrain_epochs)
            self.pretrain_rating_view(epochs=self.conf.pretrain_epochs)
            self.load_pretrained_params()
            print("Pretraining completed.")

        print("Starting main training phase...")
        for epoch in range(1, self.conf.epochs + 1):
            print('epoch: ', epoch)

            if hasattr(self.conf, 'transfer_weight_warmup'):
                current_transfer_weight = min(
                    epoch * self.conf.transfer_weight / self.conf.transfer_weight_warmup,
                    self.conf.transfer_weight
                )
                self.conf.current_transfer_weight = current_transfer_weight

            self.train_epoch(self.data, self.model_dict)

            if epoch % self.conf.eval_epochs == 0:
                self.eval_epoch(epoch, self.data, self.model_dict, save_model=True)
    def save_embeddings(self, model, model_name):

        model.eval()
        with torch.no_grad():

            all_users = torch.arange(self.conf.num_users).to(self.conf.device)

            dummy_items = torch.zeros_like(all_users).to(self.conf.device)
            _, user_emb, _ = model(all_users, dummy_items, mode='eval')
            all_items = torch.arange(self.conf.num_items).to(self.conf.device)
            dummy_users = torch.zeros_like(all_items).to(self.conf.device)
            _, _, item_emb = model(dummy_users, all_items, mode='eval')

            user_emb = user_emb.cpu().numpy()
            item_emb = item_emb.cpu().numpy()

            embedding_dir = os.path.join(self.conf.cur_out_path, 'embeddings')
            if not os.path.exists(embedding_dir):
                os.makedirs(embedding_dir)

            user_emb_path = os.path.join(embedding_dir, f'{model_name}_user_embeddings.npy')
            item_emb_path = os.path.join(embedding_dir, f'{model_name}_item_embeddings.npy')

            np.save(user_emb_path, user_emb)
            np.save(item_emb_path, item_emb)

            print(f"Saved embeddings for {model_name}:")
            print(f"User embeddings shape: {user_emb.shape}")
            print(f"Item embeddings shape: {item_emb.shape}")
            print(f"Files saved at: {embedding_dir}")

    def get_embeddings(self, users, items):

        user_gcn_emb, item_gcn_emb = self.generate_graphemb(self.norm_adj)
        return user_gcn_emb, item_gcn_emb


    @staticmethod
    def get_model(conf, data, model_name):
        if model_name == 'twin':
            return TwinGCN(conf, data)
        elif model_name == 'social':
            return SocialGCN(conf, data)
        elif model_name == 'rating':
            return RatingGCN(conf, data)
        else:
            raise NotImplementedError('Invalid model', conf.model)

    def get_model_dict(self, model_list):
        model_dict = {}
        for model_name in model_list:
            model = self.get_model(self.conf, self.data, model_name).to(self.conf.device)
            model_dict[model_name] = {
                'model': model,
                'model_path': os.path.join(self.conf.cur_out_path, model_name + '.pkt'),
                'optimizer': torch.optim.Adam(model.parameters(), lr=self.conf.lr),
                'criterion': torch.nn.BCELoss(),
                'best_perform': {
                    'train_hit': 0.0,
                    'train_ndcg': 0.0,
                    'valid_hit': 0.0,
                    'valid_ndcg': 0.0,
                    'test_hit': 0.0,
                    'test_ndcg': 0.0,
                },
                'cur_perform': {
                    'train_hit': 0.0,
                    'train_ndcg': 0.0,
                    'valid_hit': 0.0,
                    'valid_ndcg': 0.0,
                    'test_hit': 0.0,
                    'test_ndcg': 0.0,
                }
            }
        return model_dict

    def train_epoch(self, data, model_dict):
        data_loader = data.train_data_loader
        for step, batch_data in enumerate(tqdm(data_loader, desc="Iteration")):
            users_list, items_list, labels = batch_data
            users = torch.tensor(users_list).to(self.conf.device).long()
            items = torch.tensor(items_list).to(self.conf.device).long()
            labels = torch.tensor(labels).to(self.conf.device).float()

            batch_loss = 0
            pre_dict = {}

            for model_name in model_dict.keys():
                model = model_dict[model_name]['model']
                if model_name == 'twin':

                    predict, user_emb, item_emb, cl_social, cl_fused, cl_rating, consistency_loss = \
                        model_dict[model_name]['model'](users, items, users_list, items_list, mode='train')

                    pred_loss = model_dict[model_name]['criterion'](predict, labels)

                    if hasattr(self.conf, 'ssl_reg') and self.conf.ssl_reg > 0:

                        cl_loss_social = model_dict[model_name]['model'].contrastive_loss(
                            cl_social,
                            cl_fused,
                            users,
                            items
                        )

                        cl_loss_rating = model_dict[model_name]['model'].contrastive_loss(
                            cl_rating,
                            cl_fused,
                            users,
                            items
                        )

                        alignment_weight = getattr(self.conf, 'alignment_weight', 0.6)
                        batch_loss += (
                                pred_loss +
                                alignment_weight * consistency_loss +
                                self.conf.ssl_reg * (cl_loss_social + cl_loss_rating)
                        )
                    else:
                        batch_loss += pred_loss + alignment_weight * consistency_loss

                    pre_dict[model_name] = {
                        'pre': predict,
                        'user_emb': user_emb,
                        'item_emb': item_emb
                    }

                else:
                    predict, user_emb, item_emb = \
                        model(users, items, users_list, items_list, mode='train')
                    model_loss = model_dict[model_name]['criterion'](predict, labels)
                    batch_loss += model_loss
                    pre_dict[model_name] = {
                        'pre': predict,
                        'user_emb': user_emb,
                        'item_emb': item_emb
                    }

            for model_name in model_dict.keys():
                model_dict[model_name]['optimizer'].zero_grad()
            batch_loss.backward()
            for model_name in model_dict.keys():
                model_dict[model_name]['optimizer'].step()
    @staticmethod
    def get_eval_metrics_single_process(message_q, user_list, positive_predict_dict, negative_predict_dict,
                                        evaluate, top_k):
        hit_k_list = []
        ndcg_k_list = []
        for user in user_list:
            hit_k, ndcg_k = evaluate.get_hit_ndcg(positive_predict_dict[user], negative_predict_dict[user], top_k)
            hit_k_list.append(hit_k)
            ndcg_k_list.append(ndcg_k)
        mean_hit_k = np.mean(hit_k_list)
        mean_ndcg_k = np.mean(ndcg_k_list)
        message_q.put((mean_hit_k, mean_ndcg_k, len(hit_k_list)))

    def get_eval_metrics_multi_process(self, user_list, positive_predict_dict, negative_predict_dict):
        message_q = Queue()

        batch_size = len(user_list) // self.conf.num_proc + 1
        index = 0
        process_list = []
        for _ in range(self.conf.num_proc):
            if index + batch_size < len(user_list):
                batch_user_list = user_list[index:index + batch_size]
                index = index + batch_size
            else:
                batch_user_list = user_list[index:len(user_list)]
            p = Process(target=self.get_eval_metrics_single_process,
                        args=(message_q, batch_user_list, positive_predict_dict, negative_predict_dict, self.evaluate, self.conf.top_k))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()

        hit_k_sum = 0.0
        ndcg_k_sum = 0.0
        num_user_sum = 0.0
        for _ in range(self.conf.num_proc):
            mean_hit_k, mean_ndcg_k, num_user = message_q.get()
            hit_k_sum += mean_hit_k * num_user
            ndcg_k_sum += mean_ndcg_k * num_user
            num_user_sum += num_user
        mean_hit_k = hit_k_sum / num_user_sum
        mean_ndcg_k = ndcg_k_sum / num_user_sum

        return mean_hit_k, mean_ndcg_k

    def eval_net(self, net, user_idx_dict, user_list, item_list, neg_data_loader, criterion, mode='train'):
        net.eval()
        if mode == 'train':
            num_negatives = self.conf.num_eval_negatives
        else:
            num_negatives = self.conf.num_test_negatives

        with torch.no_grad():
            positive_users = torch.tensor(user_list).to(self.conf.device).long()
            positive_items = torch.tensor(item_list).to(self.conf.device).long()
            positive_labels = torch.ones_like(positive_users).to(self.conf.device).float()
            positive_predicts, _, _ = net(positive_users, positive_items, mode='eval')
            positive_loss = criterion(positive_predicts, positive_labels).item()
            positive_predicts = positive_predicts.cpu().numpy()

            positive_predict_dict = defaultdict(list)
            negative_predict_dict = defaultdict(list)

            for user_id in user_list:
                positive_predict_dict[user_id] = positive_predicts[user_idx_dict[user_id]]

            negative_loss = 0.0
            num_negative_users = 0
            for step, data in enumerate(tqdm(neg_data_loader, desc="Iteration")):
                users_idx_list, negative_users, negative_items = data
                negative_users = torch.tensor(negative_users).to(self.conf.device).long()
                negative_items = torch.tensor(negative_items).to(self.conf.device).long()
                negative_labels = torch.zeros_like(negative_users).to(self.conf.device).float()
                negative_predicts, _, _ = net(negative_users, negative_items, mode='eval')
                negative_loss += criterion(negative_predicts, negative_labels).item() * len(negative_users)
                num_negative_users += len(negative_users)
                negative_predicts = negative_predicts.cpu().numpy().reshape(-1, num_negatives)

                for idx, user_id in enumerate(users_idx_list):
                    negative_predict_dict[user_id] = negative_predicts[idx]
            negative_loss = negative_loss / num_negative_users

            t1 = time.time()
            mean_hit_k, mean_ndcg_k = self.get_eval_metrics_multi_process(
                user_list, positive_predict_dict, negative_predict_dict
            )
            t2 = time.time()
            print('Eval_time', t2 - t1)

        return positive_loss, negative_loss, mean_hit_k, mean_ndcg_k

    def eval_epoch(self, epoch, data, model_dict, save_model=False):
        with open(self.conf.log_path, 'a') as f:
            f.write('epoch: {}\n'.format(epoch))
        for model in model_dict.keys():
            train_pos_loss, train_neg_loss, train_hit, train_ndcg = self.eval_net(
                model_dict[model]['model'],
                data.train_eval_user_idx_dict,
                data.train_eval_user_list,
                data.train_eval_item_list,
                data.train_neg_data_loader_test,
                model_dict[model]['criterion'],
                mode='test',
            )
            valid_pos_loss, valid_neg_loss, valid_hit, valid_ndcg = self.eval_net(
                model_dict[model]['model'],
                data.valid_eval_user_idx_dict,
                data.valid_eval_user_list,
                data.valid_eval_item_list,
                data.valid_neg_data_loader_test,
                model_dict[model]['criterion'],
                mode='train')
            test_pos_loss, test_neg_loss, test_hit, test_ndcg = self.eval_net(
                model_dict[model]['model'],
                self.data.test_eval_user_idx_dict,
                self.data.test_eval_user_list,
                self.data.test_eval_item_list,
                self.data.test_neg_data_loader_test,
                model_dict[model]['criterion'],
                mode='train')

            model_dict[model]['cur_perform'] = {
                'train_hit': train_hit,
                'train_ndcg': test_ndcg,
                'valid_hit': valid_hit,
                'valid_ndcg':valid_ndcg,
                'test_hit': test_hit,
                'test_ndcg': test_ndcg
            }
            if model_dict[model]['cur_perform']['valid_ndcg'] >= model_dict[model]['best_perform']['valid_ndcg']:
                model_dict[model]['best_perform'] = {
                    'train_hit': train_hit,
                    'train_ndcg': test_ndcg,
                    'valid_hit': valid_hit,
                    'valid_ndcg':valid_ndcg,
                    'test_hit': test_hit,
                    'test_ndcg': test_ndcg
                }
                if save_model:
                    torch.save(model_dict[model]['model'].state_dict(), model_dict[model]['model_path'])
            print(('Model: {}, '
                        'train_loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}, '
                        'valid_loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}, '
                        'test_loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}\n'
                        .format(model, train_pos_loss, train_neg_loss, train_hit, train_ndcg,
                                valid_pos_loss, valid_neg_loss, valid_hit, valid_ndcg,
                                test_pos_loss, test_neg_loss, test_hit, test_ndcg)))

            with open(self.conf.log_path, 'a') as f:
                f.write('Model: {}, '
                        'train_loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}, '
                        'valid_loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}, '
                        'test_loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}\n'
                        .format(model, train_pos_loss, train_neg_loss, train_hit, train_ndcg,
                                valid_pos_loss, valid_neg_loss, valid_hit, valid_ndcg,
                                test_pos_loss, test_neg_loss, test_hit, test_ndcg))

    def _generate_timestamp_filename(self, prefix, extension):

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{prefix}_{timestamp}.{extension}"
        return os.path.join(self.results_dir, filename)

    def _save_test_results(self, results_data, model_metrics, best_performance):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_filepath = os.path.join(self.results_dir, f'test_results_{timestamp}.txt')

        with open(results_filepath, 'w', encoding='utf-8') as f:
            f.write('Test Results\n')
            f.write('=' * 50 + '\n\n')
            f.write(f"Generated Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write('Model Results:\n')
            f.write('-' * 30 + '\n')
            for result in results_data:
                f.write(result + '\n')

            f.write('\nBest Performance Summary:\n')
            f.write('-' * 30 + '\n')
            f.write(f"Best Model: {best_performance['model']}\n")
            f.write(f"Hit@K: {best_performance['hit']:.4f}\n")
            f.write(f"NDCG@K: {best_performance['ndcg']:.4f}\n")
            f.write(f"Loss values (positive/negative): ({best_performance['pos_loss']:.4f}, {best_performance['neg_loss']:.4f})\n")

            # Add separator
            f.write('\n' + '=' * 50 + '\n\n')
            f.write('Training Log\n')
            f.write('=' * 50 + '\n\n')

            # Append training log content
            try:
                with open(self.conf.log_path, 'r', encoding='utf-8') as log_file:
                    log_content = log_file.read()
                    f.write(log_content)
            except Exception as e:
                f.write(f"Failed to read training log file: {str(e)}\n")

        print(f"\nResults and training log have been saved to: {results_filepath}")
        return results_filepath

    def test(self):
        results_data = []
        best_performance = {
            'model': None,
            'hit': 0.0,
            'ndcg': 0.0,
            'pos_loss': float('inf'),
            'neg_loss': float('inf')
        }

        model_metrics = {}

        for model in self.model_dict.keys():
            self.model_dict[model]['model'].load_state_dict(
                torch.load(self.model_dict[model]['model_path'])
            )

            self.save_embeddings(self.model_dict[model]['model'], model)

            test_pos_loss, test_neg_loss, test_hit, test_ndcg = self.eval_net(
                self.model_dict[model]['model'],
                self.data.test_eval_user_idx_dict,
                self.data.test_eval_user_list,
                self.data.test_eval_item_list,
                self.data.test_neg_data_loader_test,
                self.model_dict[model]['criterion'],
                mode='test'
            )

            # Store metrics
            model_metrics[model] = {
                'pos_loss': test_pos_loss,
                'neg_loss': test_neg_loss,
                'hit': test_hit,
                'ndcg': test_ndcg
            }

            result_str = ('Model: {} Test Set - Average Loss: ({:.4f}, {:.4f}), '
                          'Hit@K: {:.4f}, NDCG@K: {:.4f}').format(
                model, test_pos_loss, test_neg_loss, test_hit, test_ndcg
            )
            results_data.append(result_str)
            print(result_str)

            if test_ndcg > best_performance['ndcg'] or \
                    (test_ndcg == best_performance['ndcg'] and test_hit > best_performance['hit']):
                best_performance = {
                    'model': model,
                    'hit': test_hit,
                    'ndcg': test_ndcg,
                    'pos_loss': test_pos_loss,
                    'neg_loss': test_neg_loss
                }

        saved_filepath = self._save_test_results(results_data, model_metrics, best_performance)

        print('\n' + '=' * 50)
        print('Best Performance Summary:')
        print('Best Model: {}'.format(best_performance['model']))
        print('Hit@K: {:.4f}'.format(best_performance['hit']))
        print('NDCG@K: {:.4f}'.format(best_performance['ndcg']))
        print('Loss values (positive/negative): ({:.4f}, {:.4f})'.format(
            best_performance['pos_loss'],
            best_performance['neg_loss']
        ))
        print('=' * 50)
        print(f'\nDetailed results have been saved to: {saved_filepath}')
