import torch
import numpy as np
import pandas as pd
from scipy import sparse as sp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, subgraph, remove_self_loops, add_self_loops
import os

def read_data(args, dataset):
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    
    split_idx = dataset.get_idx_split()
    n = dataset.graph['num_nodes']
    # infer the number of classes for non one-hot and one-hot labels
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']
    print(f"num nodes {n} | num edges {edge_index.size(1)} | num classes {c} | num node feats {d}")
    
    adjs = []
    adj, _ = remove_self_loops(edge_index)
    adj, _ = add_self_loops(adj, num_nodes=n)
    adjs.append(adj)
    for i in range(1): # edge_index of high order adjacency
        adj = adj_mul(adj, adj, n)
        adjs.append(adj)
    dataset.graph['adjs'] = [adj.to(torch.int).to(torch.int64) for adj in adjs]

    adj_loss_inter, _ = remove_self_loops(dataset.edge[:, 0:dataset.n_infer])
    adj_loss_intra2, _ = remove_self_loops(dataset.edge[:, dataset.n_infer:])
    
    return dataset, x, n, c, d, split_idx, adjs, adj_loss_inter, adj_loss_intra2
    
def adj_mul(adj_i, adj, N): 
    adj_i_sp = torch.sparse_coo_tensor(adj_i, torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_j = torch.sparse.mm(adj_i_sp, adj_sp)
    adj_j = adj_j.coalesce().indices()
    return adj_j
    
def hard_loss(sub_edge_inter, z2, lamda1, alpha=0.7):
    """
    跨模态一致性增强损失：
    - alpha：控制MSE和余弦相似度的权重（默认0.7）
    """
    node1_inter = sub_edge_inter[0].long()
    node2_inter = sub_edge_inter[1].long()

    feature1_inter = z2[:, node1_inter, :]
    feature2_inter = z2[:, node2_inter, :]

    # --- ① MSE loss ---
    mse_loss_inter = F.mse_loss(feature1_inter, feature2_inter)

    # --- ② Cosine similarity loss (方向一致性) ---
    cos_sim = F.cosine_similarity(feature1_inter, feature2_inter, dim=-1)
    cos_loss = 1 - torch.mean(cos_sim)  # 越相似越小

    # --- ③ 融合两种约束 ---
    total_loss = alpha * mse_loss_inter + (1 - alpha) * cos_loss

    # --- ④ 阈值修正 ---
    total_loss = torch.clamp(total_loss - lamda1, min=0)
    return total_loss


def query_graph_loss(p, sub_edge_intra2, lamda2, temperature=0.5, beta=0.6):
    """
    模态内一致性增强损失：
    - 引入soft一致性约束（KL散度）
    - beta控制“hard match”与“soft分布对齐”的平衡
    - temperature控制logit平滑度
    """
    # softmax平滑
    probs = F.softmax(p / temperature, dim=1)
    node1_intra = sub_edge_intra2[0].long()
    node2_intra = sub_edge_intra2[1].long()

    p1 = probs[node1_intra]
    p2 = probs[node2_intra]

    # --- ① Soft分布一致性 (KL散度) ---
    kl_loss = F.kl_div(p1.log(), p2, reduction='batchmean') + F.kl_div(p2.log(), p1, reduction='batchmean')

    # --- ② Hard匹配部分（原逻辑强化版） ---
    values, indices = torch.max(probs, dim=1)
    node1_values = values[node1_intra]
    node2_values = values[node2_intra]
    node1_indices = indices[node1_intra]
    node2_indices = indices[node2_intra]

    match_mask = (node1_indices == node2_indices).float()
    conf = node1_values * node2_values
    a = conf * match_mask
    hard_loss_term = 1 - torch.mean(a)

    # --- ③ 混合两种约束 ---
    loss_mix = beta * hard_loss_term + (1 - beta) * kl_loss
    loss_mix = torch.clamp(loss_mix - lamda2, min=0)
    return loss_mix

def move_query_to_reference(model, x, adjs, atac_idx, train_atac_idx, label_train, device, k = 0.95):
    
    out, link_loss_, z1, z2 = model(x, adjs)
    p = F.softmax(out, dim=1)
                
    selected_indices = torch.nonzero(torch.max(p, axis = 1)[0][atac_idx] > k).squeeze()
    if (selected_indices.numel()):
        if atac_idx[selected_indices].dim() == 0:
            query_selected = atac_idx[selected_indices].unsqueeze(0)
        else:
            query_selected = atac_idx[selected_indices]
        train_atac_idx = torch.cat((train_atac_idx, query_selected), dim=0)
        label_pre = torch.max(p, axis=1)[1][query_selected].to(device)
        label_train[query_selected] = label_pre
        mask = torch.ones_like(atac_idx, dtype=torch.bool)
        mask[selected_indices] = 0
        atac_idx = atac_idx[mask]    
    return atac_idx, train_atac_idx, label_train, p
    
def sub_graph(args, idx_i, adj_loss_inter, adj_loss_intra2, num_nodes, adjs, device):
    adjs_i = []
    sub_edge_inter, _, inter_mask_i = subgraph(idx_i, adj_loss_inter, num_nodes=num_nodes, relabel_nodes=True, return_edge_mask=True)
    sub_edge_intra2, _, intra2_mask_i = subgraph(idx_i, adj_loss_intra2, num_nodes=num_nodes, relabel_nodes=True, return_edge_mask=True)
    edge_index_i, _, edge_mask_i = subgraph(idx_i, adjs[0], num_nodes=num_nodes, relabel_nodes=True, return_edge_mask=True)
    adjs_i.append(edge_index_i.to(device))
    for k in range(1):
        edge_index_i, _ = subgraph(idx_i, adjs[k+1], num_nodes=num_nodes, relabel_nodes=True)
        adjs_i.append(edge_index_i.to(device))
    
    return sub_edge_inter, sub_edge_intra2, adjs_i
    

def save_model(args, dataset, model, x, adjs, epoch):
    resultname = args.data_dir + 'results/'
    modelname = args.data_dir + 'model/'
            
    if not os.path.exists(resultname):
        os.makedirs(resultname)
    if not os.path.exists(modelname):
        os.makedirs(modelname)
            
    filename = args.data_dir + f'results/{args.dataset}.csv'
            
    out, link_loss_, z1, z2 = model(x, adjs)
    p = F.softmax(out, dim=1)
    query_acc = sum(dataset.label.squeeze().numpy()[dataset.n_data1:] == p.argmax(dim=-1, keepdim=True).detach().squeeze().numpy()[dataset.n_data1:])/len(dataset.label.squeeze().numpy()[dataset.n_data1:]);
    
    torch.save(z2, resultname + 'embedding.pt')
    torch.save(p, resultname + 'out.pt')
    torch.save(dataset.label, resultname + 'label.pt')
    torch.save(dataset.num_celltype, resultname + 'num_celltype.pt')
    torch.save(dataset.metadata, resultname + 'metadata.pt')
    torch.save(args, resultname + 'args.pt')
    torch.save(model.state_dict(), modelname + f'{args.dataset}.pkl')
    with open(f"{filename}", 'w') as write_obj:
        write_obj.write(f"{args.dataset}," +
                        f"epoch: {epoch:02d}," +
                        f"Query: {100 * query_acc:.2f}%")
    return query_acc

def DGTA_output(args):
    resultname = args.data_dir + 'results/'

    # Label
    label = torch.load(resultname + 'label.pt', map_location=torch.device('cpu')) # celltype
    label = label.squeeze().numpy()

    # Confidence score for predicting cell type
    out = torch.load(resultname + 'out.pt', map_location=torch.device('cpu')) # pre
    pre = out.argmax(dim=-1, keepdim=True).detach().squeeze().numpy()
    pro = out.max(dim=-1, keepdim=True)[0].detach().squeeze().numpy()

    metadata = torch.load(resultname + 'metadata.pt', map_location=torch.device('cpu')) # tech
    metadata = metadata.squeeze().numpy() 

    num_celltype = torch.load(resultname + 'num_celltype.pt', map_location=torch.device('cpu'))

    celltype = [num_celltype[i] for i in label] # cell type
    pre = [num_celltype[i] for i in pre] # predicting cell type
    metadata = ["scATAC-seq" if i == 2 else "scRNA-seq" for i in metadata] # tech
    output = {   'celltype' : celltype,
                 'prediction' : pre,
                 'omic' : metadata,
                 'Confidence score' : pro}
    output = pd.DataFrame(output)
    return output
