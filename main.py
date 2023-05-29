# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from sklearn.metrics import f1_score
from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay)
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy import dot
from tqdm import tqdm
import torch.optim as optim
import dgl.function as fn
import dgl.nn as dglnn
import torch.nn.functional as F
import torch.nn as nn
from dgl.dataloading import DataLoader
from scipy.sparse import coo_matrix
import dgl
import torch
import pandas as pd
import numpy as np
import pickle as pkl
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import pdb
from argparse import ArgumentParser

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('rawdata/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


def read_txt(file):
    res_list = list()
    with open(file, "r") as f:
        line_list = f.readlines()
    for line in line_list:
        res_list.append(list(map(int, line.strip().split(' '))))
    return res_list


base_path = "rawdata/"

cite_file = "paper_file_ann.txt"
train_ref_file = "bipartite_train_ann.txt"
test_ref_file = "bipartite_test_ann.txt"
coauthor_file = "author_file_ann.txt"
feature_file = "feature.pkl"

citation = read_txt(os.path.join(base_path, cite_file))
existing_refs = read_txt(os.path.join(base_path, train_ref_file))
refs_to_pred = read_txt(os.path.join(base_path, test_ref_file))
coauthor = read_txt(os.path.join(base_path, coauthor_file))

feature_file = os.path.join(base_path, feature_file)
with open(feature_file, 'rb') as f:
    paper_feature = pkl.load(f)


print("Number of citation edges: {}\n\
Number of existing references: {}\n\
Number of author-paper pairs to predict: {}\n\
Number of coauthor edges: {}\n\
Shape of paper features: {}"
      .format(len(citation), len(existing_refs), len(refs_to_pred), len(coauthor), paper_feature.shape))


cite_edges = pd.DataFrame(citation, columns=['source', 'target'])
cite_edges = cite_edges.set_index(
    "c-" + cite_edges.index.astype(str)
)

ref_edges = pd.DataFrame(existing_refs, columns=['source', 'target'])
ref_edges = ref_edges.set_index(
    "r-" + ref_edges.index.astype(str)
)

coauthor_edges = pd.DataFrame(coauthor, columns=['source', 'target'])
coauthor_edges = coauthor_edges.set_index(
    "a-" + coauthor_edges.index.astype(str)
)

cite_edges.head()


node_tmp = pd.concat([cite_edges.loc[:, 'source'],
                     cite_edges.loc[:, 'target'], ref_edges.loc[:, 'target']])
node_papers = pd.DataFrame(index=pd.unique(node_tmp))

node_tmp = pd.concat(
    [ref_edges['source'], coauthor_edges['source'], coauthor_edges['target']])
node_authors = pd.DataFrame(index=pd.unique(node_tmp))

print("Number of paper nodes: {}, number of author nodes: {}".format(
    len(node_papers), len(node_authors)))


# sample a subset (at a proportion of sample_frac) of the reference edges as training data
# the rest are used as test true data
# the same number of false data are sampled from the rest of the nodes
# the test data are then combined with the false data to form the test data
sample_frac = 0.9

train_refs = ref_edges.sample(frac=sample_frac, random_state=0, axis=0)
train_true_refs = train_refs.copy()
train_true_refs.loc[:, 'label'] = 1
false_source = node_authors.sample(
    frac=train_refs.shape[0]/node_authors.shape[0], random_state=0, replace=True, axis=0)
false_target = node_papers.sample(
    frac=train_refs.shape[0]/node_papers.shape[0], random_state=0, replace=True, axis=0)
false_source = false_source.reset_index()
false_target = false_target.reset_index()
train_false_refs = pd.concat([false_source, false_target], axis=1)
train_false_refs.columns = ['source', 'target']
train_false_refs = train_false_refs[train_false_refs.isin(ref_edges) == False]
train_false_refs.loc[:, 'label'] = 0
train_refs_with_labels = pd.concat(
    [train_true_refs, train_false_refs.iloc[:min(len(false_source), len(false_target))]])
train_refs_with_labels = train_refs_with_labels.sample(
    frac=1, random_state=0, axis=0)

test_true_refs = ref_edges[~ref_edges.index.isin(train_refs.index)]
test_true_refs.loc[:, 'label'] = 1

false_source = node_authors.sample(
    frac=test_true_refs.shape[0]/node_authors.shape[0], random_state=0, replace=True, axis=0)
false_target = node_papers.sample(
    frac=test_true_refs.shape[0]/node_papers.shape[0], random_state=0, replace=True, axis=0)
false_source = false_source.reset_index()
false_target = false_target.reset_index()
test_false_refs = pd.concat([false_source, false_target], axis=1)
test_false_refs.columns = ['source', 'target']
test_false_refs = test_false_refs[test_false_refs.isin(ref_edges) == False]
test_false_refs.loc[:, 'label'] = 0

test_refs = pd.concat(
    [test_true_refs, test_false_refs.iloc[:min(len(false_source), len(false_target))]])
test_refs = test_refs.sample(frac=1, random_state=0, axis=0)
test_refs.head()


os.environ['DGLBACKEND'] = 'pytorch'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_ref_tensor = torch.from_numpy(train_refs.values)
cite_tensor = torch.from_numpy(cite_edges.values)
coauthor_tensor = torch.from_numpy(coauthor_edges.values)
rel_list = [('author', 'ref', 'paper'), ('paper', 'cite', 'paper'),
            ('author', 'coauthor', 'author'), ('paper', 'beref', 'author')]
graph_data = {
    rel_list[0]: (train_ref_tensor[:, 0], train_ref_tensor[:, 1]),
    rel_list[1]: (torch.cat([cite_tensor[:, 0], cite_tensor[:, 1]]), torch.cat([cite_tensor[:, 1], cite_tensor[:, 0]])),
    rel_list[2]: (torch.cat([coauthor_tensor[:, 0], coauthor_tensor[:, 1]]), torch.cat([coauthor_tensor[:, 1], coauthor_tensor[:, 0]])),
    rel_list[3]: (train_ref_tensor[:, 1], train_ref_tensor[:, 0])
}
hetero_graph = dgl.heterograph(graph_data)
node_features = {'author': torch.rand(
    node_authors.shape[0], paper_feature.shape[1]), 'paper': paper_feature}
hetero_graph.ndata['features'] = node_features
hetero_graph = hetero_graph.to(device)
print(hetero_graph)


# def build_graph(edges: pd.DataFrame, undirected = False, device=device):
#     src, dst = torch.tensor(edges['source'].values), torch.tensor(edges['target'].values)
#     g = dgl.graph((src, dst),idtype=torch.int32,device=device)
#     if undirected:
#         g = dgl.to_bidirected(g)
#     return g

# # Build the Graphs
# u_i_sp_mat = coo_matrix((np.ones(len(ref_edges)), (ref_edges['source'], ref_edges['target'])))
# g_i = dgl.bipartite_from_scipy(u_i_sp_mat, utype='Author', etype='Ref', vtype='Paper', idtype=torch.int32, device=device) # interest graph (user-item)
# g_s = build_graph(coauthor_edges, undirected=True) # social graph (user-user)
# g_p = build_graph(cite_edges) # paper graph (item-item)

# print("Social graph: {}\n\
#       Paper graph: {}\n\
#       Interest graph: {}".format(g_s, g_p, g_i))


# from torch.utils.data import Dataset, DataLoader


class LightGCNLayer(nn.Module):
    def __init__(self, n_layers, keep_prob=0.6):
        super(LightGCNLayer, self).__init__()
        self.n_layers = n_layers
        # self.graph = graph
        # self.graph = self._convert_sp_mat_to_sp_tensor(self.graph)
        # self.graph = self.graph.coalesce().to(device)
        self.keep_prob = keep_prob

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def create_adj_mat(self, adj_mat):
        assert adj_mat.shape[0] == adj_mat.shape[1]
        adj = adj_mat + torch.eye(adj_mat.shape[0])

        # D^-1 * A
        rowsum = torch.sum(adj, axis=1)

        d_inv = torch.pow(rowsum, -1).flatten()
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.sparse.spdiags(d_inv, torch.tensor(0), adj.shape)

        norm_adj = torch.sparse.mm(d_mat_inv, adj)
        return norm_adj

    def __dropout(self, keep_prob):
        graph = self.__dropout_x(self.graph, keep_prob)
        return graph

    def forward(self, graph, user_embedding, item_embedding):
        users_num, items_num = graph.num_src_nodes(
            'author'), graph.num_dst_nodes('paper')
        assert users_num == user_embedding.shape[0] and items_num == item_embedding.shape[0]
        # u_id,v_id = graph.adj_sparse('coo')
        u_id, v_id = graph.edges()
        u_id_r, v_id_r = torch.cat((u_id, v_id)), torch.cat((v_id, u_id))
        assert u_id.shape[0] == v_id.shape[0]
        # vs = torch.tensor([1.0]*u_id.shape[0]*2).to(device)

        # add self loop
        u_degrees, v_degrees = graph.out_degrees()+1, graph.in_degrees()+1
        assert torch.where(u_degrees[u_id] == 0)[0].shape[0] == 0 and torch.where(
            v_degrees[v_id] == 0)[0].shape[0] == 0
        vs = torch.cat((1.0/u_degrees[u_id], 1.0/v_degrees[v_id]))

        eye_arange = torch.arange(0,users_num+items_num).to(device)
        ones_arange = torch.ones((users_num+items_num,),dtype=torch.float32,device=device)
        ones_inv = ones_arange/torch.cat((u_degrees, v_degrees))
        eye_matrix = torch.sparse.FloatTensor(torch.stack(
            (eye_arange, eye_arange)), ones_inv, torch.Size([users_num+items_num, items_num+users_num]))
        
        # pdb.set_trace()
        adj_mat = torch.sparse.FloatTensor(torch.stack(
            (u_id_r, v_id_r)), vs, torch.Size([users_num+items_num, items_num+users_num]))
        # adj_mat = self.create_adj_mat(adj_mat)
        adj_mat = adj_mat + eye_matrix
        self.graph = adj_mat.coalesce()

        all_emb = torch.cat([user_embedding, item_embedding])
        embs = [all_emb]
        if self.training:
            g_droped = self.__dropout(self.keep_prob)
        else:
            g_droped = self.graph
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [users_num, items_num])
        return users, items


class GraphSage(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout=0.5):
        super(GraphSage, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for _ in range(n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation

    def forward(self, g, features):
        h = features  # (u,v)
        u, v = (h[0], h[1]), (h[1], h[0])
        assert len(h) == 2
        for i, layer in enumerate(self.layers):
            if i != 0:
                u = (self.activation(h[0]), self.activation(h[1]))
                v = (self.activation(h[1]), self.activation(h[0]))
            # h = self.dropout(h[0]), sSelf.dropout(h[1])
            h = layer(dgl.reverse(g), v), layer(g, u)
        return h[1]


class GraphGAT(nn.Module):
    def __init__(self, in_feats, n_hidden, num_heads, n_layers, activation=None, dropout=0.5):
        super(GraphGAT, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            # self.layers.append(dglnn.GATConv(in_feats, n_hidden, num_heads))
            self.layers.append(dglnn.DotGatConv(in_feats, n_hidden, num_heads))
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.activation(h)
            h = self.dropout(h)
            h = layer(g, h).flatten(1)
        return h


class MyModel(nn.Module):
    def __init__(self,
                 etypes,
                 num_nodes,
                 in_feats=512,
                 out_feats=64,
                 num_layers=2,
                 emb_size=64,
                 gcn_layers=3,
                 gat_layers=2,
                 hidden_size=None):
        super().__init__()

        self.etypes = etypes  # ['ref', 'cite', 'coauthor', 'beref']
        self.num_nodes = num_nodes  # {'author': 6611, 'paper': 79937}
        self.num_layers = num_layers
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.emb_size = emb_size
        self.gcn_layers = gcn_layers
        self.gat_layers = gat_layers

        self.hidden_size = hidden_size if hidden_size else self.emb_size
        assert self.emb_size % self.hidden_size == 0

        self.user_embedding = nn.Embedding(
            self.num_nodes['author'], self.emb_size)
        self.item_embedding = nn.Embedding(
            self.num_nodes['paper'], self.emb_size)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        self.reduce_dimension_layer = nn.Linear(self.in_feats, self.emb_size)

        self.influence_diffusion_layer = nn.ModuleList([GraphGAT(self.emb_size, self.hidden_size, int(
            self.emb_size/self.hidden_size), n_layers=self.gat_layers, activation=F.relu) for _ in range(self.num_layers)])
        # self.influence_diffusion_layer = nn.ModuleList([dglnn.GATConv(self.emb_size, self.hidden_size, num_heads=int(
        # self.emb_size/self.hidden_size))for _ in range(self.num_layers)])
        self.user_interest_diffusion_layer = [LightGCNLayer(n_layers=self.gcn_layers) for _ in
                                              range(self.num_layers)]
        # self.user_interest_diffusion_layer = nn.ModuleList([GraphSage(
        # self.emb_size, self.emb_size, self.emb_size, self.gcn_layers, F.relu, 0.5) for _ in range(self.num_layers)])
        self.item_interest_diffusion_layer = nn.ModuleList([GraphSage(
            self.emb_size, self.emb_size, self.emb_size, self.gcn_layers, F.relu, 0.5) for _ in range(self.num_layers)])
        # self.user_interest_diffusion_layer = nn.ModuleList([dglnn.SAGEConv(
        # self.emb_size, self.emb_size, aggregator_type='pool')for _ in range(self.num_layers)])
        # self.item_interest_diffusion_layer = nn.ModuleList([dglnn.SAGEConv(
        # self.emb_size, self.emb_size, aggregator_type='pool')for _ in range(self.num_layers)])
        # self.citation_diffusion_layer = nn.ModuleList([dglnn.GATConv(self.emb_size, self.hidden_size, num_heads=int(
        # self.emb_size/self.hidden_size))for _ in range(self.num_layers)])
        self.citation_diffusion_layer = nn.ModuleList([GraphGAT(self.emb_size, self.hidden_size, int(
            self.emb_size/self.hidden_size), n_layers=self.gat_layers, activation=F.relu)for _ in range(self.num_layers)])

        self.graph_u_i_attention_layer = nn.ModuleList([nn.Sequential(nn.Linear(self.emb_size*2, self.emb_size),
                                                                      nn.Tanh(),
                                                                      nn.Linear(self.emb_size,1),
                                                                      nn.LeakyReLU())
                                                        for i in range(self.num_layers)])
        self.graph_i_u_attention_layer = nn.ModuleList([nn.Sequential(nn.Linear(self.emb_size*2, self.emb_size),
                                                                      nn.Tanh(),
                                                                      nn.Linear(self.emb_size,1),
                                                                      nn.LeakyReLU())
                                                        for i in range(self.num_layers)])
        self.graph_u_u_attention_layer = nn.ModuleList([nn.Sequential(nn.Linear(self.emb_size*2, self.emb_size),
                                                                      nn.Tanh(),
                                                                      nn.Linear(self.emb_size,1),
                                                                      nn.LeakyReLU())
                                                        for i in range(self.num_layers)])
        self.graph_i_i_attention_layer = nn.ModuleList([nn.Sequential(nn.Linear(self.emb_size*2, self.emb_size),
                                                                      nn.Tanh(),
                                                                      nn.Linear(self.emb_size,1),
                                                                      nn.LeakyReLU())
                                                        for i in range(self.num_layers)])

        self.final_user_embedding = torch.nn.Parameter(torch.randn(
            self.num_nodes['author'], self.emb_size*(self.num_layers+2)), requires_grad=True)
        self.final_item_embedding = torch.nn.Parameter(torch.randn(
            self.num_nodes['paper'], self.emb_size*(self.num_layers+2)), requires_grad=True)

        self.config = {
            'num_layers': self.num_layers,
            'in_feats': self.in_feats,
            'out_feats': self.out_feats,
            'emb_size': self.emb_size,
            'hidden_size': self.hidden_size,
            'gcn_layers': self.gcn_layers,
            'gat_layers': self.gat_layers
        }

    def convertDistribution(self, x):
        mean, std = torch.mean(x), torch.std(x)
        y = (x - mean) * 0.1 / std
        return y

    def forward_fusion_layer(self, user_idx, item_idx, features):
        assert item_idx.shape[0] == features['paper'].shape[0] and user_idx.shape[0] == features['author'].shape[0]

        item_first_coverted_fetaures = self.convertDistribution(
            features['paper'])
        user_first_coverted_fetaures = self.convertDistribution(
            features['author'])
        item_reduce_dimension = self.reduce_dimension_layer(
            item_first_coverted_fetaures)
        user_reduce_dimension = self.reduce_dimension_layer(
            user_first_coverted_fetaures)
        item_second_coverted_fetaures = self.convertDistribution(
            item_reduce_dimension)
        user_second_coverted_fetaures = self.convertDistribution(
            user_reduce_dimension)
        user_emb, item_emb = self.user_embedding(
            user_idx), self.item_embedding(item_idx)
        fusion_item_embedding = item_emb + item_second_coverted_fetaures
        fusion_user_embedding = user_emb + user_second_coverted_fetaures
        return fusion_user_embedding, fusion_item_embedding, user_emb, item_emb, user_second_coverted_fetaures, item_second_coverted_fetaures

    def predict(self, u, v):
        # u: tensor(batch_size, emb_size*n)
        # v: tensor(batch_size, emb_size*n)
        return torch.sigmoid(torch.sum(u*v, 1, keepdim=True))

    def prediction(self, edge_subgraph, h, author_idx, paper_idx):
        # edge_subgraph: {'ref': tensor(batch_ref_num,2), 'beref': tensor(batch_beref_num,2), 'cite': tensor(batch_cite_num,2), 'coauthor': tensor(batch_coauthor_num,2)}
        # h: {'author': tensor(batch_author_num,emb_size), 'paper': tensor(batch_paper_num,emb_size)}
        with edge_subgraph.local_scope():
            h_ = {}
            for ntype in ['author', 'paper']:
                h_[ntype] = h[ntype][self.mask_idx(locals()[ntype+'_idx'],
                                                   edge_subgraph.nodes[ntype].data[dgl.NID]) == 1]
            edge_subgraph.ndata['h'] = h_
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return edge_subgraph.edata['score']

    def mask_idx(self, long_ids, short_ids):
        assert long_ids.shape[0] >= short_ids.shape[0] and long_ids.dim(
        ) == 1 and short_ids.dim() == 1
        mask = torch.zeros_like(
            long_ids, dtype=torch.int, device=long_ids.device)
        mask_ = torch.zeros_like(
            long_ids, dtype=torch.int, device=short_ids.device)
        long_ids, ori_pos = torch.sort(long_ids)
        short_ids, _ = torch.sort(short_ids)
        mask[torch.searchsorted(long_ids, short_ids)] = 1
        mask_[ori_pos] = mask
        return mask_

    def forward(self, positive_graph, negative_graph, blocks, x):
        # x: {'author': tensor(batch_author_num,in_feature), 'paper': tensor(batch_paper_num,in_feature)}
        # blocks[0]: {'author': tensor(batch_author_num,in_feature), 'paper': tensor(batch_paper_num,in_feature)}
        item_idx = blocks[0].nodes['paper'].data[dgl.NID]
        user_idx = blocks[0].nodes['author'].data[dgl.NID]
        assert item_idx.shape[0] == x['paper'].shape[0] and user_idx.shape[0] == x['author'].shape[0]

        # fusion layer
        fusion_user_embedding, fusion_item_embedding,\
            user_emb, item_emb, user_second_coverted_fetaures, item_second_coverted_fetaures\
            = self.forward_fusion_layer(user_idx, item_idx, x)

        user_embeddings = [user_emb, user_second_coverted_fetaures]
        item_embeddings = [item_emb, item_second_coverted_fetaures]

        for i in range(self.num_layers):
            # diffusion layer
            # user-item graph, item-user graph, item-item graph, user-user graph
            g_u_i, g_i_u, g_i_i, g_u_u = (blocks[i].edge_type_subgraph([etype])
                                          for etype in ['ref', 'beref', 'cite', 'coauthor'])

            g_i_i = dgl.node_type_subgraph(dgl.add_self_loop(g_i_i), ['paper'])
            g_u_u = dgl.node_type_subgraph(
                dgl.add_self_loop(g_u_u), ['author'])
            graphs = {"g_u_i": g_u_i, "g_i_u": g_i_u,
                      "g_i_i": g_i_i, "g_u_u": g_u_u}

            node_masks = {g_name: (self.mask_idx(user_idx, graph.nodes['author'].data[dgl.NID]) if 'author' in graph.ntypes else torch.zeros_like(user_idx),
                                   self.mask_idx(item_idx, graph.nodes['paper'].data[dgl.NID]) if 'paper' in graph.ntypes else torch.zeros_like(item_idx))
                          for g_name, graph in graphs.items()}

            g_u_i = dgl.node_subgraph(g_u_i, {'author': torch.where(node_masks["g_u_i"][0] == 1)[
                                      0], 'paper': torch.where(node_masks["g_u_i"][1] == 1)[0]})
            g_i_u = dgl.node_subgraph(g_i_u, {'paper': torch.where(node_masks["g_i_u"][1] == 1)[
                                      0], 'author': torch.where(node_masks["g_i_u"][0] == 1)[0]})
            graphs.update({"g_u_i": g_u_i, "g_i_u": g_i_u})

            user_embedding_from_interest_diffusion = fusion_user_embedding.clone()
            item_embedding_from_interest_diffusion = fusion_item_embedding.clone()
            user_embedding_from_interest_diffusion[node_masks["g_u_i"][0] == 1], item_embedding_from_interest_diffusion[node_masks["g_u_i"][1] == 1] = \
                self.user_interest_diffusion_layer[i](
                g_u_i, fusion_user_embedding[node_masks["g_u_i"][0] == 1], fusion_item_embedding[node_masks["g_u_i"][1] == 1])
            # user_embedding_from_interest_diffusion[node_masks["g_i_u"][0] == 1] = self.user_interest_diffusion_layer[i](
            # g_i_u, fusion_item_embedding[node_masks["g_i_u"][1] == 1], fusion_user_embedding[node_masks["g_i_u"][0] == 1]).flatten(1)
            # user_embedding_from_interest_diffusion[node_masks["g_i_u"][0] == 1] = self.user_interest_diffusion_layer[i](
            # g_i_u, (fusion_item_embedding[node_masks["g_i_u"][1] == 1], fusion_user_embedding[node_masks["g_i_u"][0] == 1])).flatten(1)
            user_embedding_from_influence_diffusion = fusion_user_embedding.clone()
            user_embedding_from_influence_diffusion[node_masks["g_u_u"][0] == 1] = self.influence_diffusion_layer[i](
                g_u_u, fusion_user_embedding[node_masks["g_u_u"][0] == 1]).flatten(1)
            user_items_attention = torch.exp(self.graph_u_i_attention_layer[i](torch.concat(
                [fusion_user_embedding, user_embedding_from_interest_diffusion], 1))) + 0.7
            user_users_attention = torch.exp(self.graph_u_u_attention_layer[i](torch.concat(
                [fusion_user_embedding, user_embedding_from_influence_diffusion], 1))) + 0.3
            user_items_attention = user_items_attention / \
                (user_items_attention + user_users_attention)
            user_users_attention = 1 - user_items_attention

            # item_embedding_from_interest_diffusion = fusion_item_embedding.clone()
            # item_embedding_from_interest_diffusion[node_masks["g_u_i"][1] == 1] = self.item_interest_diffusion_layer[i](
            # g_u_i, (fusion_user_embedding[node_masks["g_u_i"][0] == 1], fusion_item_embedding[node_masks["g_u_i"][1] == 1])).flatten(1)
            item_embedding_from_citation_diffusion = fusion_item_embedding.clone()
            item_embedding_from_citation_diffusion[node_masks["g_i_i"][1] == 1] = self.citation_diffusion_layer[i](
                g_i_i, fusion_item_embedding[node_masks["g_i_i"][1] == 1]).flatten(1)
            item_users_attention = torch.exp(self.graph_i_u_attention_layer[i](torch.concat(
                [fusion_item_embedding, item_embedding_from_interest_diffusion], 1))) + 0.7
            item_items_attention = torch.exp(self.graph_i_i_attention_layer[i](torch.concat(
                [fusion_item_embedding, item_embedding_from_citation_diffusion], 1))) + 0.3
            item_users_attention = item_users_attention / \
                (item_users_attention + item_items_attention)
            item_items_attention = 1 - item_users_attention

            fusion_user_embedding = 1/2 * fusion_user_embedding + 1/2 * \
                user_embedding_from_interest_diffusion*user_items_attention
            + 1/2 * user_embedding_from_influence_diffusion*user_users_attention
            # fusion_user_embedding = user_embedding_from_interest_diffusion + \
            # user_embedding_from_influence_diffusion

            fusion_item_embedding = 1/2 * fusion_item_embedding + 1/2 * \
                item_embedding_from_interest_diffusion*item_users_attention
            + 1/2 * item_embedding_from_citation_diffusion*item_items_attention
            # fusion_item_embedding = item_embedding_from_interest_diffusion\
            # + item_embedding_from_citation_diffusion

            user_embeddings.append(fusion_user_embedding)
            item_embeddings.append(fusion_item_embedding)

        # prediction layer

        user_final_embedding = torch.cat(user_embeddings, 1)
        item_final_embedding = torch.cat(item_embeddings, 1)

        # prediction = self.predict(user_final_embedding, item_final_embedding)
        h = {'author': user_final_embedding, 'paper': item_final_embedding}
        with torch.no_grad():
            self.final_user_embedding[user_idx] = user_final_embedding.detach()
            self.final_item_embedding[item_idx] = item_final_embedding.detach()

        pos_score = self.prediction(positive_graph, h, user_idx, item_idx)
        neg_score = self.prediction(negative_graph, h, user_idx, item_idx)
        weights = self.user_embedding.weight[user_idx], self.item_embedding.weight[item_idx]
        return pos_score, neg_score, weights

    def node_embed(self, node_type, node_idx):
        if node_type == 'author':
            return self.final_user_embedding[node_idx]
        elif node_type == 'paper':
            return self.final_item_embedding[node_idx]
        else:
            raise ValueError("node type must be author or paper")

    def save_model(self, path):
        torch.save({"model": self.state_dict()}, path)
        print(f"Model saved at {model_name}.")
        # save embeddings
        embeding_file_name = os.path.splitext(
            path)[0] + '_embedding' + os.path.splitext(path)[1]
        torch.save({"user_embedding": self.final_user_embedding,
                    "item_embedding": self.final_item_embedding}, embeding_file_name)
        print(f"Embeddings saved at {embeding_file_name}.")
        config_file_name = os.path.splitext(path)[0] + '_hyper_params.json'
        json.dump(self.config, open(config_file_name, 'w'))
        print(f"Config file saved at {config_file_name}.")

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model"])
        print(f"Model loaded from {path}.")
        # load embeddings
        embedding_file_name = os.path.splitext(
            path)[0] + '_embedding' + os.path.splitext(path)[1]
        checkpoint = torch.load(embedding_file_name)
        self.final_user_embedding = checkpoint["user_embedding"]
        self.final_item_embedding = checkpoint["item_embedding"]
        print(f"Embeddings loaded from {embedding_file_name}.")


def compute_loss(pos_score, neg_score, weights, lambda_, etype=('author', 'ref', 'paper')):
    n_edges = pos_score[etype].shape[0]
    labels = torch.cat([torch.ones(n_edges), torch.zeros(n_edges)]).to(device)
    preds = torch.cat([pos_score[etype], neg_score[etype].view(
        n_edges, -1).mean(-1, keepdim=True)]).to(device)
    reg_loss = lambda_ * (weights[0].norm(2).pow(2) + weights[1].norm(2).pow(2))
    # reg_loss = lambda_ * torch.cat(weights, dim=0).norm(2).pow(2)
    # return F.binary_cross_entropy_with_logits(preds.squeeze(1), labels)
    # return (1 - F.logsigmoid(pos_score[etype] - neg_score[etype].view(n_edges, -1))).mean() + reg_loss
    return (1 - F.logsigmoid(pos_score[etype].view(n_edges, -1))).mean() + \
        (1 - F.logsigmoid(-neg_score[etype].view(n_edges, -1))).mean() + reg_loss
    # return (1 - F.logsigmoid(pos_score[etype].view(n_edges, -1))).mean() + \
    # (1 - F.logsigmoid(-neg_score[etype].view(n_edges, -1))).mean()


def compute_loss_with_labels(node_embeddings):
    train_arr = torch.tensor(train_refs_with_labels.values)
    predict = torch.sum(
        node_embeddings['author'][train_arr[:, 0]]*node_embeddings['paper'][train_arr[:, 1]], 1)
    lbl_true = torch.tensor(
        train_refs_with_labels.label.to_numpy(), dtype=torch.float32, device=device)
    return F.binary_cross_entropy_with_logits(predict, lbl_true.to(device))


def train(model: MyModel, loader, opt, lambda_=1e-6, decay_rate=1000, max_epoch=100, use_mean_threshold=False):
    model.train()
    for epoch in range(max_epoch):
        with tqdm(loader) as tq:
            for i, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(tq):
                blocks = [b.to(device) for b in blocks]
                input_features = blocks[0].srcdata['features']
                positive_graph = positive_graph.to(device)
                negative_graph = negative_graph.to(device)
                # import pdb;pdb.set_trace()
                pos_score, neg_score, weights = model(
                    positive_graph, negative_graph, blocks, input_features)
                # score = torch.cat([pos_score[rel_list[0]], neg_score[rel_list[0]]])
                # label = torch.cat([torch.ones_like(pos_score[rel_list[0]]), torch.zeros_like(neg_score[rel_list[0]])])
                # loss = F.binary_cross_entropy_with_logits(score, label)
                loss = compute_loss(
                    pos_score, neg_score, weights, lambda_=lambda_*np.exp(-epoch/decay_rate))
                # loss = compute_loss_with_labels({'author':model.final_user_embedding, 'paper':model.final_item_embedding})
                opt.zero_grad()
                loss.backward()
                opt.step()
                tq.set_postfix({"loss": "{:.4f}".format(
                    loss.item()), "Epoch": epoch+1}, refresh=False)
        eval(model, use_mean_threshold=use_mean_threshold)


def eval(model: MyModel, train_test=False, validation=True, prediction=False, 
         load_from_model_name: str = None, sav_fig=False, fig_name: str = None, 
         save_fig_path: str = None, save_result_path: str = None, use_mean_threshold=False):
    
    model.eval()

    if load_from_model_name is not None:
        model.load_model(model_name)

    with torch.no_grad():
        user_nids = hetero_graph.nodes('author').to(device)
        item_nids = hetero_graph.nodes('paper').to(device)
        # item_nids = torch.LongTensor(model.local_to_global_nid['paper']).to(device)
        node_embeddings = {}
        node_embeddings['author'] = model.node_embed(
            'author', user_nids).detach().cpu()
        node_embeddings['paper'] = model.node_embed(
            'paper', item_nids).detach().cpu()
        # print(node_embeddings['author'].shape, node_embeddings['paper'].shape)

    # Use Cosine Similarity to measure the probability of recommandation

    def cos_sim(a, b):
        cos_sim = np.sum(a * b, axis=1) / (norm(a, axis=1) * norm(b, axis=1))
        return cos_sim

    train_test = train_test or (prediction and use_mean_threshold)
    if train_test:
        train_arr = np.array(train_refs_with_labels.values)
        res = cos_sim(np.array(node_embeddings['author'][train_arr[:, 0]]), np.array(
            node_embeddings['paper'][train_arr[:, 1]]))
        lbl_true = train_refs_with_labels.label.to_numpy()
        lbl_true = lbl_true.flatten()
        threshold = 0  # 0.5
        train_mean_threshold = res.mean()
        if use_mean_threshold: threshold = train_mean_threshold
        lbl_pred = np.array(res)
        lbl_pred[lbl_pred >= threshold] = 1
        lbl_pred[lbl_pred < threshold] = 0
        # Test scores
        print("F1-Score on train: {:.3f}".format(f1_score(lbl_true, lbl_pred)))

    test_arr = np.array(test_refs.values)
    # import pdb;pdb.set_trace()
    res = cos_sim(np.array(node_embeddings['author'][test_arr[:, 0]]), np.array(
        node_embeddings['paper'][test_arr[:, 1]]))

    # Generate predict labels
    lbl_true = test_refs.label.to_numpy()
    lbl_true = lbl_true.flatten()

    # ROC curve
    if sav_fig:
        precision, recall, _ = precision_recall_curve(lbl_true, np.array(res))
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot()
        if fig_name is not None:
            plt.title(fig_name)
        if save_fig_path is not None:
            figpath = save_fig_path
        else:
            dir, filename = os.path.split(model_name)
            basename = os.path.splitext(filename)[0]
            figpath = os.path.join("figs", "ROC_" + basename +".png")
        plt.savefig(figpath)
        print(f"ROC graph saved at {figpath}.")
        plt.close()

    if validation:
        # import pdb;pdb.set_trace()
        lbl_pred = np.array(res)
        threshold = 0  # 0.5
        if use_mean_threshold: threshold = res.mean()
        lbl_pred[lbl_pred >= threshold] = 1
        lbl_pred[lbl_pred < threshold] = 0
        # Test scores
        print("F1-Score on valid: {:.3f}".format(f1_score(lbl_true, lbl_pred)))

    # Output your prediction
    if prediction:
        print("Generating final prediction ...")
        test_arr = np.array(refs_to_pred)
        res = cos_sim(np.array(node_embeddings['author'][test_arr[:, 0]]), np.array(
            node_embeddings['paper'][test_arr[:, 1]]))
        if use_mean_threshold: threshold = train_mean_threshold
        res[res >= threshold] = 1
        res[res < threshold] = 0
        data = []
        for index, p in enumerate(list(res)):
            tp = [index, str(int(p))]
            data.append(tp)

        df = pd.DataFrame(data, columns=["Index", "Predicted"], dtype=object)
        if save_result_path is None:
            save_result_path = 'submission/Submission_1.csv'
        df.to_csv(save_result_path, index=False)
        print(f"Final prediction saved at {save_result_path}.")


def parse_args():
    """
    Argument parser for flexible experiment control.
    """
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--batch_size", "-b", type=int, default=131072, help="batch size" #32768
    )
    arg_parser.add_argument(
        "--lr", "-r", type=float, default=5e-3, help="learning rate"
    )
    arg_parser.add_argument(
        "--num_epochs", "-e", type=int, default=40, help="num of epochs"
    )
    arg_parser.add_argument("--seed", "-s", type=int,
                            default=26, help="random seed")

    arg_parser.add_argument("--layers_num", "-l",
                            type=int, default=2, help="layer num")

    arg_parser.add_argument(
        "--negative_sample_num", "-n", type=int, default=30, help="negative sample num"
    )
    arg_parser.add_argument("--emb_size", "-d", type=int,
                            default=64, help="emb size")
    arg_parser.add_argument(
        "--hidden_size", "-c", type=int, default=32, help="hidden size"
    )
    arg_parser.add_argument("--out_feats", "-o", type=int,
                            default=64, help="out feats")

    arg_parser.add_argument(
        "--eval", "-v", action="store_true", help="whether evaluate model"
    )
    arg_parser.add_argument(
        "--eval_model_path",
        "-m",
        type=str,
        default="",
        help="evaluate model path",
    )
    arg_parser.add_argument(
        "--reg_lambda", type=float, default=1e-6, help="coefficient of the weight penalty"
    )
    arg_parser.add_argument(
        "--decay_rate", type=int, default=1000, help="the decay rate of lambda. eg. lambda_=lambda*np.exp(-epoch/decay_rate)"
    )
    arg_parser.add_argument("--gcn_layers_num",
                            type=int, default=3, help="gcn layer num")
    arg_parser.add_argument("--gat_layers_num",
                            type=int, default=2, help="gat layer num")
    arg_parser.add_argument("--use_mean_threshold",
                            type=bool, 
                            default=True, 
                            help="whether to use the mean value of cosine similarity as the threshold to predict or not")
    
    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    layers_num = args.layers_num
    negative_sample_num = args.negative_sample_num
    seed = args.seed
    dgl.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=layers_num)
    sampler = dgl.dataloading.NeighborSampler([4]*layers_num)
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler,
        exclude='reverse_types', reverse_etypes={'ref': 'beref', 'beref': 'ref'},
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(negative_sample_num),
    )
    train_eid_dict = {
        etype: hetero_graph.edges(etype=etype, form="eid")
        for etype in hetero_graph.etypes
    }  # The returned result is a 1D tensor EID, representing the IDs of all edges.
    loader = dgl.dataloading.DataLoader(
        hetero_graph,
        train_eid_dict,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    num_nodes = {ntype: hetero_graph.num_nodes(ntype) for ntype in hetero_graph.ntypes}
    assert (
        node_features.keys() == num_nodes.keys()
        and node_features["author"].shape[1] == node_features["paper"].shape[1]
    )
    dict = {
        "in_feats": node_features["author"].shape[1],
        "emb_size": args.emb_size,
        "hidden_size": args.hidden_size,
        "out_feats": args.out_feats,
        "etypes": hetero_graph.etypes,
        "num_nodes": num_nodes,
        "num_layers": layers_num,
        'gcn_layers': 3,
        'gat_layers': 2
    }

    model = MyModel(**dict).to(device)
    if not args.eval:
        lr = args.lr
        max_epoch = args.num_epochs
        lambda_ = args.reg_lambda
        decay_rate = args.decay_rate
        print("lr:", lr)
        opt = optim.Adam(model.parameters(), lr=lr)

        train(model, loader, opt, max_epoch=max_epoch, 
              lambda_=lambda_, decay_rate=decay_rate, use_mean_threshold=args.use_mean_threshold)
        model_name = f"models/model_lr_{lr}_{max_epoch}.pth"

        # print(model.final_user_embedding)
        # print(model.final_item_embedding)
        model.save_model(model_name)

    else:
        model_name = args.eval_model_path
    eval(model, load_from_model_name=model_name, sav_fig=True, prediction=True, use_mean_threshold=args.use_mean_threshold)

    # layers_num = 2
    # negative_sample_num = 5
    # dgl.seed(26)
    # # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(
    # # num_layers=layers_num)
    # sampler = dgl.dataloading.NeighborSampler([4, 4])
    # sampler = dgl.dataloading.as_edge_prediction_sampler(
    #     sampler, exclude='reverse_types', reverse_etypes={'ref': 'beref', 'beref': 'ref'},
    #     negative_sampler=dgl.dataloading.negative_sampler.Uniform(negative_sample_num))
    # train_eid_dict = {
    #     etype: hetero_graph.edges(etype=etype, form='eid')
    #     for etype in hetero_graph.etypes}  # The returned result is a 1D tensor EID, representing the IDs of all edges.
    # loader = dgl.dataloading.DataLoader(
    #     hetero_graph, train_eid_dict, sampler,
    #     batch_size=131072,  # 32768
    #     shuffle=True,
    #     drop_last=False,
    #     num_workers=0,

    # )

    # num_nodes = {ntype: hetero_graph.num_nodes(
    #     ntype) for ntype in hetero_graph.ntypes}
    # assert node_features.keys() == num_nodes.keys(
    # ) and node_features['author'].shape[1] == node_features['paper'].shape[1]
    # dict = {'in_feats': node_features['author'].shape[1],
    #         'emb_size': 64,
    #         'hidden_size': 32,
    #         'out_feats': 64,
    #         'etypes': hetero_graph.etypes,
    #         'num_nodes': num_nodes,
    #         'num_layers': layers_num,
    #         'gcn_layers': 3,
    #         'gat_layers': 2
    #         }

    # lr = 0.003
    # max_epoch = 40
    # lambda_ = 1e-6
    # print("lr:", lr)
    # model = MyModel(**dict).to(device)
    # opt = optim.Adam(model.parameters(), lr=lr)

    # model_name = f"models/model_lr_{lr}_{max_epoch}.pth"
    # # model.load_model(model_name)
    # train(model, loader, opt, max_epoch=max_epoch, lambda_=lambda_)

    # # print(model.final_user_embedding)
    # # print(model.final_item_embedding)
    # model.save_model(model_name)

    # eval(model, load_from_model_name=model_name, sav_fig=True, prediction=True)
