import torch
import torch.nn as nn
from model.layers import SocialConv, RatingConv
import torch.nn.functional as F


class FeatureTransfer(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.layer_norm = nn.LayerNorm(emb_dim)
        self.batch_norm = nn.BatchNorm1d(emb_dim)


        self.transfer_net = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim, emb_dim)
        )


        self.projection = nn.Linear(emb_dim, emb_dim)


        self.residual_weight = nn.Parameter(torch.FloatTensor([0.5]))

    def forward(self, source_emb, target_emb):

        source_emb = self.layer_norm(source_emb)
        target_emb = self.layer_norm(target_emb)


        combined = torch.cat([source_emb, target_emb], dim=-1)


        transfer_features = self.transfer_net(combined)
        transfer_features = self.batch_norm(transfer_features)


        residual = self.projection(source_emb)


        output = residual + self.residual_weight * transfer_features

        return output




class DomainAdapter(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.domain_aligner = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, emb_dim)
        )

    def forward(self, social_view, rating_view):

        aligned_social = self.domain_aligner(social_view)
        aligned_rating = self.domain_aligner(rating_view)


        consistency_loss = F.mse_loss(aligned_social, aligned_rating)

        return aligned_social, aligned_rating, consistency_loss

class MetaLearner(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        self.decay = 0.99
        self.register_buffer('moving_average', torch.zeros(emb_dim))

    def forward(self, social_view, rating_view):
        meta_knowledge = self.meta_net(
            torch.cat([social_view, rating_view], dim=-1)
        )

        with torch.no_grad():
            self.moving_average = self.decay * self.moving_average + (1 - self.decay) * meta_knowledge.mean(0)
        return meta_knowledge * F.sigmoid(self.moving_average)


class GradientReversal(torch.autograd.Function):


    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class TwinGCN(nn.Module):
    def __init__(self, conf, data):
        super(TwinGCN, self).__init__()
        self.conf = conf
        self.data = data


        self.dropout = nn.Dropout(p=getattr(conf, 'dropout', 0.4))
        self.user_emb = nn.Parameter(torch.normal(mean=0, std=0.01, size=(conf.num_users, conf.emb_dim)))
        self.item_emb = nn.Parameter(torch.normal(mean=0, std=0.01, size=(conf.num_items, conf.emb_dim)))
        self.reg_weight = getattr(conf, 'reg_weight', 1e-4)
        self.social_layer = SocialConv()
        self.rating_layer = RatingConv(conf, conf.emb_dim)
        self.temp = getattr(conf, 'temp', 0.07)


        self.feature_transfer = FeatureTransfer(conf.emb_dim)
        self.domain_adapter = DomainAdapter(conf.emb_dim)
        self.meta_learner = MetaLearner(conf.emb_dim)


        self.alpha = nn.Parameter(torch.FloatTensor([0.7]))
        self.beta = nn.Parameter(torch.FloatTensor([0.1]))
        self.gamma = nn.Parameter(torch.FloatTensor([0.1]))
        self.delta = nn.Parameter(torch.FloatTensor([0.1]))


        self.weight_norm = nn.Softmax(dim=0)


    def get_social_rating_views(self, graph, mode='test', user_emb=None, item_emb=None):

        if user_emb is None:
            user_emb = self.user_emb
        if item_emb is None:
            item_emb = self.item_emb


        user_deg = graph.out_degrees(etype='like').float().to(self.conf.device).unsqueeze(1)
        u_sw = 1 - user_deg / (user_deg + 1e-8)
        item_deg = graph.out_degrees(etype='rev_like').float().to(self.conf.device).unsqueeze(1)
        i_sw = 1 - item_deg / (item_deg + 1e-8)


        social_user_emb_gnn1 = self.social_layer(graph, user_emb)
        if mode == 'train':
            social_user_emb_gnn1 = self.dropout(social_user_emb_gnn1)
        social_user_emb_gnn2 = self.social_layer(graph, social_user_emb_gnn1)
        if mode == 'train':
            social_user_emb_gnn2 = self.dropout(social_user_emb_gnn2)


        rating_user_emb_gnn1, rating_item_emb_gnn1 = self.rating_layer(
            graph, user_emb, item_emb, u_sw, i_sw
        )
        if mode == 'train':
            rating_user_emb_gnn1 = self.dropout(rating_user_emb_gnn1)
            rating_item_emb_gnn1 = self.dropout(rating_item_emb_gnn1)

        rating_user_emb_gnn2, rating_item_emb_gnn2 = self.rating_layer(
            graph, rating_user_emb_gnn1, rating_item_emb_gnn1, u_sw, i_sw
        )
        if mode == 'train':
            rating_user_emb_gnn2 = self.dropout(rating_user_emb_gnn2)
            rating_item_emb_gnn2 = self.dropout(rating_item_emb_gnn2)


        return {
            'social': (social_user_emb_gnn1, social_user_emb_gnn2),
            'rating': (rating_user_emb_gnn1, rating_user_emb_gnn2,
                       rating_item_emb_gnn1, rating_item_emb_gnn2)
        }

    def get_embeddings(self, mode='test'):

        graph = self.data.data_graph.local_var()

        if mode == 'train':
            user_emb = self.dropout(self.user_emb)
            item_emb = self.dropout(self.item_emb)
        else:
            user_emb = self.user_emb
            item_emb = self.item_emb


        views = self.get_social_rating_views(graph, mode, user_emb, item_emb)
        social_gnn1, social_gnn2 = views['social']
        rating_gnn1, rating_gnn2, rating_item_gnn1, rating_item_gnn2 = views['rating']


        final_user_emb = 0
        if 0 in self.conf.l_user:
            final_user_emb = final_user_emb + user_emb
        if 1 in self.conf.l_user:
            user_emb_gnn1 = 0.8 * social_gnn1 + 0.2 * rating_gnn1
            final_user_emb = final_user_emb + user_emb_gnn1
        if 2 in self.conf.l_user:
            user_emb_gnn2 = 0.8 * social_gnn2 + 0.2 * rating_gnn2
            final_user_emb = final_user_emb + user_emb_gnn2


        if self.conf.use_transfer:

            transferred_social = self.feature_transfer(social_gnn2, rating_gnn2)
            transferred_rating = self.feature_transfer(rating_gnn2, social_gnn2)


            meta_knowledge = self.meta_learner(
                transferred_social,
                transferred_rating
            )

            weights = torch.stack([self.alpha, self.beta, self.gamma, self.delta])
            normalized_weights = self.weight_norm(weights)

            final_user_emb = (
                    normalized_weights[0] * final_user_emb +
                    normalized_weights[1] * transferred_social +
                    normalized_weights[2] * transferred_rating +
                    normalized_weights[3] * meta_knowledge
            )

            final_user_emb = F.normalize(final_user_emb, p=2, dim=1)

            aligned_social, aligned_rating, consistency_loss = self.domain_adapter(
                transferred_social,
                transferred_rating
            )

        fused_view = torch.cat([final_user_emb, item_emb], dim=0)
        social_view = torch.cat([social_gnn2, item_emb], dim=0)
        rating_view = torch.cat([rating_gnn2, rating_item_gnn2], dim=0)

        if mode == 'train':
            return social_view, fused_view, rating_view,aligned_social, aligned_rating, consistency_loss
        return social_view, fused_view, rating_view

    def contrastive_loss(self, view1_emb, view2_emb, users, items):

        view1_emb = F.normalize(view1_emb, dim=1)
        view2_emb = F.normalize(view2_emb, dim=1)

        user_v1 = view1_emb[users]
        user_v2 = view2_emb[users]

        with torch.no_grad():
            sim_matrix = torch.mm(user_v1, user_v2.t())
            neg_mask = sim_matrix > torch.diagonal(sim_matrix).unsqueeze(1)  # 找出比正样本更相似的负样本

        sim = torch.mm(user_v1, user_v2.t()) / self.temp
        exp_sim = torch.exp(sim)

        neg_sim = (exp_sim * neg_mask.float()).sum(dim=1)
        pos_sim = torch.diagonal(exp_sim)

        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8)).mean()
        return loss


    def forward(self, users, items, users_list=None, items_list=None, mode='test'):

        graph = self.data.data_graph.local_var()

        if mode == 'train' and users_list is not None:

            remove_iu_eid_list = []
            remove_ui_eid_list = []
            for idx in range(len(users_list)):
                iu_eid = self.data.train_iu_pair2eid.get((items_list[idx], users_list[idx]), -1)
                ui_eid = self.data.train_ui_pair2eid.get((users_list[idx], items_list[idx]), -1)
                if iu_eid >= 0:
                    remove_iu_eid_list.append(iu_eid)
                if ui_eid >= 0:
                    remove_ui_eid_list.append(ui_eid)
            graph.remove_edges(remove_iu_eid_list, 'rev_like')
            graph.remove_edges(remove_ui_eid_list, 'like')

        if mode == 'train':
            social_view, fused_view, rating_view, aligned_social, aligned_rating, consistency_loss = self.get_embeddings(
                mode)
        else:
            social_view, fused_view, rating_view = self.get_embeddings(mode)

        num_users = self.conf.num_users
        social_user_emb = social_view[:num_users]
        rating_user_emb = rating_view[:num_users]
        fused_user_emb = fused_view[:num_users]
        final_item_emb = fused_view[num_users:]

        latest_user_emb = fused_user_emb[users]
        latest_item_emb = final_item_emb[items]

        predict = torch.sigmoid(torch.sum(torch.mul(latest_user_emb, latest_item_emb), dim=1))

        if mode == 'train':
            return predict, latest_user_emb, latest_item_emb, social_user_emb, fused_user_emb, rating_user_emb, consistency_loss
        else:
            return predict, latest_user_emb, latest_item_emb