import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch_geometric.nn import MessagePassing, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_mean_pool
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
# from conv_base import GNN_node_Virtualnode
from GOOD.networks.models.MolEncoders import AtomEncoder, BondEncoder
#from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from gcn_conv import GCNConv
import numpy as np
class GINEncoder(torch.nn.Module):
    def __init__(self, num_layer, in_dim, emb_dim):
        super(GINEncoder, self).__init__()
        self.num_layer = num_layer
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.dropout_rate = 0.5
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(num_layer - 1)])
        self.batch_norm1 = nn.BatchNorm1d(emb_dim)
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(emb_dim) for _ in range(num_layer - 1)])
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropouts = nn.ModuleList(
            [nn.Dropout(self.dropout_rate) for _ in range(num_layer - 1)])
        self.conv1 = GINConv(
            nn.Sequential(nn.Linear(in_dim, 2 * emb_dim),
                          nn.BatchNorm1d(2 * emb_dim), nn.ReLU(),
                          nn.Linear(2 * emb_dim, emb_dim)))
        self.convs = nn.ModuleList([
            GINConv(
                nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim),
                              nn.BatchNorm1d(2 * emb_dim), nn.ReLU(),
                              nn.Linear(2 * emb_dim, emb_dim)))
            for _ in range(num_layer - 1)
        ])

    def forward(self, batched_data):

        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        post_conv = self.dropout1(
            self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
        return post_conv


class Causal(nn.Module):
    def __init__(self, hidden_in, hidden_out, hidden, num_layer, cls_layer=2):
        super(Causal, self).__init__()

        self.num_classes = 2  # 11个类别
        self.global_pool = global_add_pool

        # GIN编码器用于节点嵌入
        self.gnn_node = GINEncoder(
            num_layer=num_layer,
            in_dim=hidden_in,
            emb_dim=hidden
        )

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)

        # 使用 LayerNorm 替代 BatchNorm1d
        self.norm = nn.LayerNorm(hidden)

        # 定义 GCNConv，不使用 edge_norm
        self.objects_convs = GCNConv(hidden, hidden)

        # 对象 MLP，使用 LayerNorm 和 ReLU
        self.object_mlp = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, self.num_classes)  # 输出7个类别
        )

    def forward(self, causal):
        """前向传播：使用因果特征进行预测"""
        return self.objects_readout_layer(causal)

    def forward_causal(self, batched_data, return_attn=False):
        """生成因果特征，可选返回注意力权重"""
        if 'batch' not in batched_data:
            print("Warning: 'batch' key is missing in batched_data. Using default batch tensor.")
            batch = torch.zeros(batched_data['seq_feat'].size(0), dtype=torch.long).to(
                batched_data['seq_feat'].device)  # 默认全零 batch
        else:
            batch = batched_data['batch']

        # 从 batched_data 中提取其他信息
        x, edge_index = batched_data['x'], batched_data['edge_index']
        row, col = edge_index

        # 确保数据为 float32 类型
        x = x.to(torch.float32)  # 将输入数据转换为 float32 类型
        batch = batch.to(torch.long)  # 确保 batch 的数据类型为 long

        # 通过GIN编码器获得节点嵌入
        x = self.gnn_node(batched_data)

        # 计算边和节点的注意力权重
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight = edge_att[:, 1]

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        node_weight = node_att[:, 1]

        # 生成因果特征
        causal = node_weight.view(-1, 1) * x
        causal = self.norm(causal)
        causal = F.relu(self.objects_convs(causal, edge_index, edge_weight))

        # 对因果特征进行全局池化
        causal = self.global_pool(causal, batch)

        if return_attn:
            return causal, node_weight, edge_weight
        else:
            return causal

    def objects_readout_layer(self, x):
        """对象嵌入的读取层"""
        return self.object_mlp(x)

    def eval_forward(self, batched_data, return_attn=False):
        """用于评估的前向传播，可选返回注意力权重"""
        x, edge_index = batched_data.x, batched_data.edge_index

        # 处理batch为None的情况（单个图）
        if hasattr(batched_data, 'batch') and batched_data.batch is not None:
            batch = batched_data.batch.to(torch.long)
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)  # 创建全0的batch
        row, col = edge_index

        # 确保数据为 float32 类型（与训练保持一致）
        x = x.to(torch.float32)  # 修正为 float32 以保持一致性
        batch = batch.to(torch.long)  # 确保 batch 的数据类型为 long

        # 通过GIN编码器获得节点嵌入
        x = self.gnn_node(batched_data)

        # 计算边和节点的注意力权重
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight = edge_att[:, 1]

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        node_weight = node_att[:, 1]

        # 生成因果特征，并确保传入 LayerNorm 的是张量
        causal = node_weight.view(-1, 1) * x
        causal = self.norm(causal)
        causal = F.relu(self.objects_convs(causal, edge_index, edge_weight))

        # 对因果特征进行全局池化并输出预测结果
        causal = self.global_pool(causal, batch)
        pred = self.objects_readout_layer(causal)

        if return_attn:
            return pred, node_weight, edge_weight
        else:
            return pred

    def get_node_importance(self, data):
        """获取节点重要性分数（基于注意力权重）"""
        self.eval()
        with torch.no_grad():
            _, node_weight, _ = self.eval_forward(data, return_attn=True)
        return node_weight

    def get_edge_importance(self, data):
        """获取边重要性分数（基于注意力权重）"""
        self.eval()
        with torch.no_grad():
            _, _, edge_weight = self.eval_forward(data, return_attn=True)
        return edge_weight

    def get_causal_subgraph(self, data, node_threshold=0.5, edge_threshold=0.5):
        """
        获取因果子图的节点和边索引

        参数:
        - data: 输入图数据
        - node_threshold: 节点重要性阈值
        - edge_threshold: 边重要性阈值

        返回:
        - important_nodes: 重要节点索引
        - important_edges: 重要边索引
        """
        self.eval()
        with torch.no_grad():
            # 获取节点和边的重要性
            node_weight = self.get_node_importance(data)
            edge_weight = self.get_edge_importance(data)

            # 筛选重要节点和边
            important_nodes = (node_weight > node_threshold).nonzero(as_tuple=True)[0]
            important_edges = []

            # 遍历所有边，筛选两端节点都重要且边本身重要的边
            for edge_idx, (u, v) in enumerate(data.edge_index.t().cpu().numpy()):
                if u in important_nodes and v in important_nodes and edge_weight[edge_idx] > edge_threshold:
                    important_edges.append(edge_idx)

            return important_nodes, torch.tensor(important_edges, device=data.edge_index.device)