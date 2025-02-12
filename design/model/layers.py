import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class SocialConv(nn.Module):
    def __init__(self):
        super(SocialConv, self).__init__()

    def forward(self, graph, user_emb):
        graph = graph.local_var()
        graph.nodes['user'].data['feat'] = user_emb
        graph.update_all(fn.copy_u('feat', 'n_feat'), fn.mean('n_feat', 'new_feat'), etype='friend')
        return graph.nodes['user'].data['new_feat']


class RatingConv(nn.Module):
    def __init__(self, conf, emb_dim):
        super(RatingConv, self).__init__()
        self.conf = conf

    def forward(self, graph, user_emb, item_emb, u_sw, i_sw):
        graph = graph.local_var()
        graph.nodes['user'].data['feat'] = user_emb
        graph.nodes['item'].data['feat'] = item_emb

        # ************************************************************************* #
        graph.update_all(fn.copy_u('feat', 'n_feat'), fn.mean('n_feat', 'new_f'), etype='like')
        graph.update_all(fn.copy_u('feat', 'n_feat'), fn.mean('n_feat', 'new_f'), etype='rev_like')
        # ************************************************************************* #

        graph.nodes['user'].data['feat'] = graph.nodes['user'].data['new_f'] + graph.nodes['user'].data['feat'] * u_sw
        graph.nodes['item'].data['feat'] = graph.nodes['item'].data['new_f'] + graph.nodes['item'].data['feat'] * i_sw

        # ************************************************************************* #
        graph.update_all(fn.copy_u('feat', 'n_feat'), fn.mean('n_feat', 'new_f'), etype='like')
        graph.update_all(fn.copy_u('feat', 'n_feat'), fn.mean('n_feat', 'new_f'), etype='rev_like')
        # ************************************************************************* #

        graph.nodes['user'].data['feat'] = graph.nodes['user'].data['new_f'] + graph.nodes['user'].data['feat'] * u_sw
        graph.nodes['item'].data['feat'] = graph.nodes['item'].data['new_f'] + graph.nodes['item'].data['feat'] * i_sw

        return graph.nodes['user'].data['feat'], graph.nodes['item'].data['feat']
