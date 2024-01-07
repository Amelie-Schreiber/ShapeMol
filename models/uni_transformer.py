import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_sum
from torch_geometric.nn import radius_graph, knn_graph
from utils.covalent_graph import connect_covalent_graph
import time
import copy
from models.common import GaussianSmearing, MLP, outer_product
from models.shape_vn_layers import VNStdFeature, VNLinearLeakyReLU
import pdb

EPS = 1e-6

class BaseX2HAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, shape_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, shape_mode='attention', topo_emb_type='topo_layer', out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.out_fc = out_fc
        self.shape_mode = shape_mode
        self.topo_emb_type = topo_emb_type
        
        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        if 'attention' in self.shape_mode:
            kv_input_dim += shape_dim
                
        if 'topo_layer' in topo_emb_type:
            kv_input_dim += hidden_dim

        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        
        if 'attention' not in self.shape_mode:
            self.node_output = MLP(2 * hidden_dim + shape_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)
        else:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, invar_ligand_shape, topo_out, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None and self.edge_feat_dim > 0:
            kv_input = torch.cat([edge_feat, kv_input], -1)
        
        if 'topo_layer' in self.topo_emb_type and topo_out is not None:
            topo_hj = topo_out[dst]
            kv_input = torch.cat([kv_input, topo_hj], -1)

        if 'attention' in self.shape_mode:
            invar_ligand_shape = invar_ligand_shape[dst]
            kv_input = torch.cat([kv_input, invar_ligand_shape], -1)
        
        # compute k
        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute v
        v = self.hv_func(kv_input)
        v = v * e_w.view(-1, 1)
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute q
        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)
        
        # compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0)  # [num_edges, n_heads]

        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        output = output.view(-1, self.output_dim)
        output = torch.cat([output, h], -1)

        if 'attention' not in self.shape_mode:
            output = torch.cat([output, invar_ligand_shape], -1)
        
        output = self.node_output(output)
        output = output + h
        
        return output


class BaseH2XAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, shape_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, shape_mode='attention_residue', topo_emb_type='topo_layer'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.shape_mode = shape_mode
        self.topo_emb_type = topo_emb_type
        
        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        if 'attention' in shape_mode:
            kv_input_dim += shape_dim
        
        if 'topo_layer' in topo_emb_type:
            kv_input_dim += hidden_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        self.shape_linear = VNLinearLeakyReLU(self.n_heads + shape_dim + 1, self.n_heads, dim=4)

    def forward(self, h, x, rel_x, r_feat, edge_feat, edge_index, invar_ligand_shape, ligand_shape_emb, topo_out, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]
        
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None and self.edge_feat_dim > 0:
            kv_input = torch.cat([edge_feat, kv_input], -1)
        
        if 'topo_layer' in self.topo_emb_type and topo_out is not None:
            topo_hj = topo_out[dst]
            kv_input = torch.cat([kv_input, topo_hj], -1)
        
        if 'attention' in self.shape_mode:
            invar_ligand_shape = invar_ligand_shape[dst]
            kv_input = torch.cat([kv_input, invar_ligand_shape], -1)
        
        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.xv_func(kv_input)
        v = v * e_w.view(-1, 1)
        
        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)   # (xi - xj) [n_edges, n_heads, 3]
        
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # Compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0)  # (E, heads)

        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        output = scatter_sum(m, dst, dim=0, dim_size=N)
        
        if self.shape_mode == 'attention_residue':
            tmp_output = torch.cat((x.unsqueeze(1), output, ligand_shape_emb), dim=1)  # (E, heads + shape, 3)
            res_output = self.shape_linear(tmp_output).mean(dim=1)
            output = output.mean(dim=1) + res_output
        elif self.shape_mode == 'attention':
            output = torch.cat((x.unsqueeze(1), output, ligand_shape_emb), dim=1)  # (E, heads + shape, 3)
            output = self.shape_linear(tmp_output).mean(dim=1)
        else:
            raise ValueError("unexpected shape mode %s" % (self.shape_mode))
        return output


class EquivariantShapeEmbLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layer = VNLinearLeakyReLU(input_dim, output_dim, dim=4)

    def forward(self, shape_h):
        batch_size = shape_h.size(0)
        equiv_shape_h = self.hidden_layer(shape_h)
        return equiv_shape_h

class InvariantShapeEmbLayer(nn.Module):
    def __init__(self, input_dim, output_dim, act_fn='relu', norm=True):
        super().__init__()
        self.hidden_layer = MLP(input_dim, output_dim, input_dim, norm=norm, act_fn=act_fn)

    def forward(self, shape_h):
        batch_size = shape_h.size(0)
        shape_mean = shape_h.mean(dim=1)
        shape_mean_norm = (shape_mean * shape_mean).sum(-1, keepdim=True)
        shape_mean_norm = shape_mean / (shape_mean_norm + EPS)
        
        invar_shape_emb = torch.einsum('bij,bj->bi', shape_h, shape_mean_norm)
        invar_shape_emb = self.hidden_layer(invar_shape_emb)
        return invar_shape_emb

class BaseTopoLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        
        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim
        
        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        
        xq_input_dim = input_dim
        
        self.xq_func = MLP(xq_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        
        self.topo_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]
        
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None and self.edge_feat_dim > 0:
            kv_input = torch.cat([edge_feat, kv_input], -1)
        
        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.xv_func(kv_input)

        v = v * e_w.view(-1, 1)
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)
        
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)
        # Compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0)  # (E, heads)
        
        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3)
        
        output = output.view(output.shape[0], -1)
        output = torch.cat([output, h], -1)
        output = self.topo_output(output)
        return output  # [num_nodes, hidden_dim]

class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, shape_dim, act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, num_topo=1, r_min=0., r_max=10., num_node_types=8,
                 r_feat_mode='basic', topo_emb_type='topo_layer',
                 x2h_out_fc=True, sync_twoup=False, num_shape=15):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_topo = num_topo
        # self.r2_min = r_min ** 2 if r_min >= 0 else -(r_min ** 2)
        # self.r2_max = r_max ** 2
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.r_feat_mode = r_feat_mode  # ['origin', 'basic', 'sparse']
        
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        self.num_shape = num_shape
        self.shape_dim = shape_dim
        self.topo_emb_type = topo_emb_type
        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)
        
        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, shape_dim, n_heads, edge_feat_dim,
                                # r_feat_dim=num_r_gaussian,
                                r_feat_dim=num_r_gaussian * max(edge_feat_dim, 1),
                                act_fn=act_fn, norm=norm,
                                topo_emb_type=topo_emb_type, out_fc=self.x2h_out_fc)
            )

        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, shape_dim, n_heads, edge_feat_dim,
                                # r_feat_dim=num_r_gaussian,
                                r_feat_dim=num_r_gaussian * max(edge_feat_dim, 1),
                                act_fn=act_fn, norm=norm, topo_emb_type=topo_emb_type)
            )

    def forward(self, h, x, edge_attr, edge_index, invar_ligand_shape, ligand_shape_emb, topo_out, e_w=None):
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr  # shape: [#edges_in_batch, #bond_types]
        else:
            edge_feat = None

        if self.topo_emb_type == 'topo_layer':
            e_w, covalent_e_w = e_w
            edge_attr, covalent_edge_attr = edge_attr
            edge_index, covalent_edge_index = edge_index
        
        src, dst = edge_index
        rel_x = x[dst] - x[src]
        # dist = torch.sum(rel_x ** 2, -1, keepdim=True)
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        h_in = h
        
        # 4 separate distance embedding for p-p, p-l, l-p, l-l
        #t1 = time.time()
        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            output = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, invar_ligand_shape, topo_out, e_w=e_w)
            #h_out = self._pred_node_output(output, h_in, ligand_shape, i)
            h_in = output
        x2h_out = h_in

        new_h = h if self.sync_twoup else x2h_out
        #t3 = time.time()
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            
            delta_x = self.h2x_layers[i](new_h, x, rel_x, dist_feat, edge_feat, edge_index, invar_ligand_shape, ligand_shape_emb, topo_out, e_w=e_w)
            #delta_x = self._pred_pos_output(output, new_h, topo_out, ligand_shape, i)

            x = x + delta_x  #* mask_ligand[:, None]  # only ligand positions will be updated
            rel_x = x[dst] - x[src]
            # dist = torch.sum(rel_x ** 2, -1, keepdim=True)
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)
        
        #t4 = time.time()
        #print(f'x2h_layers: {t2 - t1}, topo_layers: {t3 - t2}, h2x_layers: {t4 - t3}')
        return x2h_out, x


class UniTransformerO2TwoUpdateGeneral(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, shape_dim, shape_latent_dim, n_heads=1, k=32,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='radius', shape_coeff=0.25, ew_net_type='global', topo_emb_type='topo_layer', 
                 r_feat_mode='basic', num_topo=8, num_init_x2h=1, num_init_h2x=0, num_x2h=1, 
                 num_h2x=1, r_max=10., x2h_out_fc=True, atom_enc_mode='add_aromatic', shape_type='pointAE_shape', sync_twoup=False):
        super().__init__()
        # Build the network
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types
        # radius graph / knn graph
        self.cutoff_mode = cutoff_mode  # [radius, none]
        self.k = k
        self.ew_net_type = ew_net_type  # [r, m, none]
        self.topo_emb_type = topo_emb_type # [topo_layer, topo_attr]
        self.r_feat_mode = r_feat_mode  # [basic, sparse]
        self.atom_enc_mode = atom_enc_mode
        self.shape_type = shape_type

        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_topo =num_topo
        self.num_init_x2h = num_init_x2h
        self.num_init_h2x = num_init_h2x
        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)
        if self.ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)

        self.shape_dim = shape_dim
        self.shape_latent_dim = shape_latent_dim
        self.shape_coeff = shape_coeff
        
        self.base_block = self._build_share_blocks()

        if self.topo_emb_type == "topo_layer":
            self.topo_layers = nn.ModuleList()
            for i in range(self.num_topo):
                self.topo_layers.append(
                    BaseTopoLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                # r_feat_dim=num_r_gaussian,
                                r_feat_dim=num_r_gaussian * max(edge_feat_dim, 1),
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type)
                )
        
        if self.shape_type == 'pointAE_shape':
            self.invariant_shape_layer = InvariantShapeEmbLayer(shape_dim, shape_latent_dim)
            self.equivariant_shape_layer = EquivariantShapeEmbLayer(shape_dim, shape_latent_dim // 3)

    def __repr__(self):
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'r_feat_mode={self.r_feat_mode}, \n' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    def _build_init_h_layer(self):
        layer = AttentionLayerO2TwoUpdateNodeGeneral(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, self.shape_dim, self.shape_latent_dim, act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            r_feat_mode=self.r_feat_mode, topo_emb_type=self.topo_emb_type,
            x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        )
        return layer

    def _build_share_blocks(self):
        # Equivariant layers
        base_block = []
        for l_idx in range(self.num_layers):
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, self.shape_dim, act_fn=self.act_fn, norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                r_feat_mode=self.r_feat_mode, topo_emb_type=self.topo_emb_type,
                x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)

    @staticmethod
    def _build_edge_type(edge_index, covalent_index=None):
        edge_type = torch.zeros(len(edge_index[0])).to(edge_index[0])
        
        if covalent_index is not None:
            edge_type[covalent_index] = 1
            edge_type = F.one_hot(edge_type, num_classes=2)
        else:
            edge_type = F.one_hot(edge_type, num_classes=1)
        return edge_type

    def _find_covalent_indices(self, edge_index, covalent_edge_index):
        tensor_edge_index = edge_index.transpose(1, 0)
        covalent_edge_index = covalent_edge_index.transpose(1, 0)
        
        _, idx, counts = torch.cat([tensor_edge_index, covalent_edge_index], dim=0).unique(dim=0, return_inverse=True, return_counts=True)
        mask = torch.isin(idx, torch.where(counts.gt(1))[0])
        mask1 = mask[:tensor_edge_index.shape[0]]
        covalent_indices = torch.arange(len(mask1))[mask1]
        return covalent_indices

    def _connect_graph(self, ligand_pos, ligand_v, batch):
        edge_index = self._connect_edge(ligand_pos, ligand_v, batch)
        
        if self.cutoff_mode == 'knn' and self.topo_emb_type == 'topo_attr':
            covalent_edge_index = self._connect_edge(ligand_pos, ligand_v, batch, cutoff_mode='cov_radius')
            covalent_index = self._find_covalent_indices(edge_index, covalent_edge_index)
            edge_type = self._build_edge_type(edge_index, covalent_index=covalent_index)
        elif self.cutoff_mode == 'cov_radius':
            # for cov_radius option, both knn graph and covalent radius graph need to be constructed and learned.
            edge_type = self._build_edge_type(edge_index, covalent_index=None)
            covalent_edge_index = self._connect_edge(ligand_pos, ligand_v, batch, cutoff_mode='cov_radius')
            covalent_edge_type = self._build_edge_type(covalent_edge_index, covalent_index=None)
            edge_index = (edge_index, covalent_edge_index)
            edge_type = (edge_type, covalent_edge_type)
        elif self.topo_emb_type == 'topo_attr':
            raise ValueError(f"Not supported edge feature: {self.topo_emb_type} for cutoff mode {self.cutoff_mode}")
        else:
            edge_type = self._build_edge_type(edge_index, covalent_index=None)
        return edge_index, edge_type

    def _connect_edge(self, ligand_pos, ligand_v, batch, cutoff_mode='knn'):
        if cutoff_mode == 'knn':
            edge_index = knn_graph(ligand_pos, k=self.k, batch=batch, flow='source_to_target')
        elif cutoff_mode == 'cov_radius':
            edge_index = connect_covalent_graph(ligand_pos, ligand_v, atom_mode=self.atom_enc_mode)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index

    def _pred_ew(self, x, edge_index):
        src, dst = edge_index
        dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
        dist_feat = self.distance_expansion(dist)
        logits = self.edge_pred_layer(dist_feat)
        e_w = torch.sigmoid(logits)
        return e_w

    def forward(self, v, h, x, batch, ligand_shape, return_all=None):

        # todo: not sure whether the init embedding layer would be helpful
        # full_src, full_dst = edge_index
        # h, _ = self.init_h_emb_layer(h, x)

        all_x = [x]
        all_h = [h]

        batch_size = torch.max(batch).item() + 1
        
        invar_ligand_shape_emb = self.invariant_shape_layer(ligand_shape)
        invar_ligand_shape_emb = torch.index_select(invar_ligand_shape_emb, 0, batch)
        
        ligand_shape_emb = torch.index_select(ligand_shape, 0, batch)
        for b_idx in range(self.num_blocks):
            edge_index, edge_type = self._connect_graph(x, v, batch)
            
            if self.ew_net_type == 'global':
                if self.topo_emb_type == 'topo_layer': # need to calculate edge weight for both the topology graph and the knn graph
                    knn_e_w = self._pred_ew(x, edge_index[0])
                    covalent_e_w = self._pred_ew(x, edge_index[1])
                    e_w = (knn_e_w, covalent_e_w)
                else:
                    e_w = self._pred_ew(x, edge_index)
            else:
                e_w = None
            
            if self.topo_emb_type == 'topo_layer':
                covalent_edge_index = edge_index[1]
                covalent_edge_attr = edge_type[1]
                cov_src, cov_dst = covalent_edge_index
                cov_rel_x = x[cov_dst] - x[cov_src]
                cov_dist = torch.norm(cov_rel_x, p=2, dim=-1, keepdim=True)
                cov_dist_feat = self.distance_expansion(cov_dist)

                h_in = h
                for i in range(self.num_topo):
                    out = self.topo_layers[i](h_in, cov_dist_feat, covalent_edge_attr, covalent_edge_index, e_w=covalent_e_w)
                    h_in = out
                topo_out = h_in
            else:
                topo_out = None


            for l_idx, layer in enumerate(self.base_block):
                h, x = layer(h, x, edge_type, edge_index, invar_ligand_shape_emb, ligand_shape_emb, topo_out, e_w=e_w)
            
            #print(f'connect edge time: {t2 - t1}, edge type compute time: {t3 - t2}, forward time: {t4 - t3}')
            all_x.append(x)
            all_h.append(h)

        # edge_index = self._connect_edge(x, mask_ligand, batch)

        outputs = {'x': x, 'h': h}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        return outputs
