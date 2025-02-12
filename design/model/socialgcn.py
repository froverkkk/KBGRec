import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import SocialConv, RatingConv

import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class DenoisingLayer(nn.Module):
    def __init__(self, conf):
        super(DenoisingLayer, self).__init__()
        self.emb_dim = conf.emb_dim
        self.device = conf.device

        # Feature transformation layers
        self.nb_layer = nn.Linear(conf.emb_dim, conf.emb_dim)
        self.self_layer = nn.Linear(conf.emb_dim, conf.emb_dim)

        # Attention layer
        self.attention = nn.Linear(2 * conf.emb_dim, 1)

        # Move modules to the correct device
        self.to(conf.device)

    @torch.no_grad()
    def normalize_adj(self, indices, values, size):
        """Normalize adjacency matrix"""
        adj = torch.sparse_coo_tensor(indices, values, size).coalesce()
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-7), -0.5)

        row, col = indices
        norm_values = values * deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Check and clean invalid values
        norm_values = torch.nan_to_num(norm_values, nan=0.0, posinf=1.0, neginf=0.0)
        return norm_values

    def forward(self, adj, features):
        # Input check
        if torch.isnan(features).any():
            print("Warning: NaN in input features")
            features = torch.nan_to_num(features, nan=0.0)

        indices = adj._indices()
        values = adj._values()

        # Get features of source and target nodes
        src_features = features[indices[0]]
        dst_features = features[indices[1]]

        # Feature transformation
        src_trans = self.nb_layer(src_features)
        src_trans = torch.nan_to_num(src_trans, nan=0.0)

        dst_trans = self.self_layer(dst_features)
        dst_trans = torch.nan_to_num(dst_trans, nan=0.0)

        # Attention calculation
        attention_features = torch.cat([src_trans, dst_trans], dim=1)
        attention_weights = self.attention(attention_features)

        # Prevent invalid values
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        attention_weights = torch.clamp(attention_weights, min=-10.0, max=10.0)

        if self.training:
            edge_weights = self.hard_concrete_sample(attention_weights).view(-1)
        else:
            edge_weights = torch.sigmoid(attention_weights).view(-1)

        # Clean invalid values
        edge_weights = torch.nan_to_num(edge_weights, nan=0.0)
        edge_weights = torch.clamp(edge_weights, min=0.0, max=1.0)

        # Update edge weights
        new_values = values * edge_weights

        # Normalization
        norm_values = self.normalize_adj(indices, new_values, adj.size())

        # Use sparse matrix multiplication
        norm_adj = torch.sparse_coo_tensor(indices, norm_values, adj.size())
        output = torch.sparse.mm(norm_adj, features)

        # Final check
        output = torch.nan_to_num(output, nan=0.0)
        return output

    def hard_concrete_sample(self, log_alpha, beta=1.0):
        gamma = -0.1
        zeta = 1.1

        try:
            u = torch.rand_like(log_alpha, device=self.device)
            noise = torch.log(u.clamp(min=1e-7)) - torch.log((1 - u).clamp(min=1e-7))
            gate_inputs = torch.sigmoid((log_alpha + noise) / beta)

            # Clean invalid values
            gate_inputs = torch.nan_to_num(gate_inputs, nan=0.5)
            gate_inputs = torch.clamp(gate_inputs, min=0.0, max=1.0)

            stretched = gate_inputs * (zeta - gamma) + gamma
            return torch.clamp(stretched, 0.0, 1.0)

        except Exception as e:
            print(f"Error in hard_concrete_sample: {e}")
            return torch.ones_like(log_alpha)


class SocialGCN(nn.Module):
    def __init__(self, conf, data):
        super(SocialGCN, self).__init__()
        self.conf = conf
        self.data = data
        self.device = conf.device

        # Initialize embeddings
        self.user_emb = nn.Parameter(torch.normal(mean=0, std=0.01, size=(conf.num_users, conf.emb_dim)))
        self.item_emb = nn.Parameter(torch.normal(mean=0, std=0.01, size=(conf.num_items, conf.emb_dim)))

        # Social layer and denoising layer
        self.social_layer = SocialConv()
        self.denoising = DenoisingLayer(conf)

        # L0 regularization parameter
        self.lambda_l0 = conf.lambda_l0 if hasattr(conf, 'lambda_l0') else 0.1

        # Cache
        self.cached_adj = None
        self.cache_name = None

    def get_adj_from_edges(self, graph):
        """
        Get adjacency matrix from graph
        Returns: adjacency matrix in torch.sparse_coo_tensor format
        """
        all_edges_src = []
        all_edges_dst = []

        # Collect all edges
        for etype in graph.canonical_etypes:
            src, dst = graph.edges(etype=etype)
            src = src.to(self.device)
            dst = dst.to(self.device)
            all_edges_src.append(src)
            all_edges_dst.append(dst)

        # Merge all edges
        if len(all_edges_src) > 0:
            src = torch.cat(all_edges_src)
            dst = torch.cat(all_edges_dst)

            # Create sparse adjacency matrix
            total_nodes = self.conf.num_users + self.conf.num_items
            indices = torch.stack([src, dst])
            values = torch.ones(src.size(0), device=self.device)

            adj = torch.sparse_coo_tensor(
                indices,
                values,
                size=(total_nodes, total_nodes),
                device=self.device
            ).coalesce()

            return adj
        else:
            # If no edges, create an empty sparse matrix
            total_nodes = self.conf.num_users + self.conf.num_items
            indices = torch.empty((2, 0), dtype=torch.long, device=self.device)
            values = torch.empty(0, device=self.device)
            return torch.sparse_coo_tensor(
                indices,
                values,
                size=(total_nodes, total_nodes),
                device=self.device
            ).coalesce()

    def forward(self, users, items, users_list=None, items_list=None, mode='test'):
        try:
            fuse_user_emb = self.user_emb
            fuse_item_emb = self.item_emb

            # Check embeddings
            if torch.isnan(fuse_user_emb).any() or torch.isnan(fuse_item_emb).any():
                print("Warning: NaN in fused embeddings")
                fuse_user_emb = torch.nan_to_num(fuse_user_emb, nan=0.0)
                fuse_item_emb = torch.nan_to_num(fuse_item_emb, nan=0.0)

            # Combine embeddings
            all_embeddings = torch.cat([fuse_user_emb, fuse_item_emb], dim=0)

            # Get graph structure
            graph = self.data.data_graph.local_var()
            adj = self.get_adj_from_edges(graph)

            # Ensure adjacency matrix is not None
            if adj is None:
                raise ValueError("Adjacency matrix is None")

            # Apply denoising layer
            embeddings_gnn1 = self.denoising(adj, all_embeddings)
            embeddings_gnn2 = self.denoising(adj, embeddings_gnn1)

            # Check GNN outputs
            embeddings_gnn1 = torch.nan_to_num(embeddings_gnn1, nan=0.0)
            embeddings_gnn2 = torch.nan_to_num(embeddings_gnn2, nan=0.0)

            # Separate user and item embeddings
            user_emb_gnn1 = embeddings_gnn1[:self.conf.num_users]
            user_emb_gnn2 = embeddings_gnn2[:self.conf.num_users]

            # Aggregate representations from different layers
            final_user_emb = torch.zeros_like(fuse_user_emb)
            if hasattr(self.conf, 'l_user'):
                if 0 in self.conf.l_user:
                    final_user_emb.add_(fuse_user_emb)
                if 1 in self.conf.l_user:
                    final_user_emb.add_(user_emb_gnn1)
                if 2 in self.conf.l_user:
                    final_user_emb.add_(user_emb_gnn2)
            else:
                # If no l_user config, use all layers
                final_user_emb = fuse_user_emb + user_emb_gnn1 + user_emb_gnn2

            final_item_emb = fuse_item_emb

            # Final checks
            final_user_emb = torch.nan_to_num(final_user_emb, nan=0.0)
            final_item_emb = torch.nan_to_num(final_item_emb, nan=0.0)

            # Get embeddings of target users and items
            latest_user_emb = final_user_emb[users]
            latest_item_emb = final_item_emb[items]

            # Final check before prediction
            latest_user_emb = torch.nan_to_num(latest_user_emb, nan=0.0)
            latest_item_emb = torch.nan_to_num(latest_item_emb, nan=0.0)

            # Use clamp for numerical stability
            pred_scores = torch.sum(latest_user_emb * latest_item_emb, dim=1)
            pred_scores = torch.clamp(pred_scores, min=-10.0, max=10.0)
            predict = torch.sigmoid(pred_scores)

            # Final prediction value check
            predict = torch.nan_to_num(predict, nan=0.5)
            predict = torch.clamp(predict, min=1e-7, max=1 - 1e-7)

            return predict, latest_user_emb, latest_item_emb

        except Exception as e:
            print(f"Error in forward pass: {e}")
            # Return safe defaults on error
            batch_size = len(users)
            predict = torch.full((batch_size,), 0.5, device=self.device)
            default_emb = torch.zeros((batch_size, self.conf.emb_dim), device=self.device)
            return predict, default_emb, default_emb
