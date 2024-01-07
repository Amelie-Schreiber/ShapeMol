import os
import time
import math
import random
import numpy as np
import pickle
import pdb
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from utils import *

from models.shape_vn_layers import *

EPS = 1e-6

class DecoderInner(nn.Module):
    ''' Decoder class.

    It does not perform any form of normalization.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, z_dim=128, hidden_size=128, layer_num=4, loss_type='occupancy', leaky=False, acts='all'):
        super().__init__()
        self.z_dim = z_dim
        self.layer_num = layer_num
        self.acts = acts
        if self.acts not in ['all', 'inp', 'first_rn', 'inp_first_rn']:
            #self.acts = 'all'
            raise ValueError('Please provide "acts" equal to one of the following: "all", "inp", "first_rn", "inp_first_rn"')

        # Submodules
        if z_dim > 0:
            self.z_in = VNLinear(z_dim, z_dim)

        self.fc_in = nn.Linear(z_dim*2+1, hidden_size)

        self.blocks = []
        for i in range(self.layer_num):
            self.blocks.append(
			    ResnetBlockFC(hidden_size)
			)
        
        self.fc_out = nn.Linear(hidden_size, 1)
        self.loss_type = loss_type

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
	
    def to(self, device):
        if self.z_dim > 0: self.z_in = self.z_in.to(device)
        self.fc_in = self.fc_in.to(device)
        for i in range(len(self.blocks)): self.blocks[i] = self.blocks[i].to(device)
        self.fc_out = self.fc_out.to(device)
        return self
    
    def forward(self, p, z, c=None, **kwargs):
        batch_size, T, D = p.size()
        acts = []
        acts_inp = []
        acts_first_rn = []
        acts_inp_first_rn = []

        net = (p * p).sum(2, keepdim=True)

        if self.z_dim != 0:
            z = z.view(batch_size, -1, D).contiguous()
            net_z = torch.einsum('bmi,bni->bmn', p, z)
            z_dir = self.z_in(z)
            z_inv = (z * z_dir).sum(-1).unsqueeze(1).repeat(1, T, 1)
            net = torch.cat([net, net_z, z_inv], dim=2)

        acts.append(net)
        acts_inp.append(net)
        acts_inp_first_rn.append(net)

        net = self.fc_in(net)
        acts.append(net)
        # acts_inp.append(net)
        # acts_inp_first_rn.append(net)

        for i in range(self.layer_num):
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        if self.loss_type == 'occupancy':
            out = F.sigmoid(out)
            
        return out

class PointCloud_AE(nn.Module):
	def __init__(self, config):
		super(PointCloud_AE, self).__init__()
		if config.encoder == 'VN_DGCNN':
			self.encoder = VN_DGCNN_Encoder(config.hidden_dim, config.latent_dim, config.layer_num, config.num_k)
		elif config.encoder == 'VN_Resnet':
			self.encoder = VN_Resnet_Encoder(config.hidden_dim, config.latent_dim, config.layer_num, config.num_k)

		self.generator = DecoderInner(config.point_dim, config.latent_dim, config.hidden_dim, config.layer_num, config.loss_type)
		self.loss_type = config.loss_type
		
	def to(self, device):
		self.encoder = self.encoder.to(device)
		self.generator = self.generator.to(device)
		return self
	
	def forward(self, inputs, z_vector, point_coord, is_training=False):
		if is_training:
			z_vector = self.encoder(inputs)
			net_out = self.generator(point_coord, z_vector)
		else:
			if inputs is not None:
				z_vector = self.encoder(inputs)
			if z_vector is not None and point_coord is not None:
				net_out = self.generator(point_coord, z_vector)
			else:
				net_out = None
		return z_vector, net_out
	
	def get_val_loss(self, point_clouds, sample_points, sample_values):
		point_clouds = point_clouds.unsqueeze(1)
		z_vectors, net_out = self.forward(point_clouds, None, sample_points, is_training=False)
		loss = torch.mean((net_out-sample_values)**2)
		
		pred_values = (net_out > 0.5).long()
		acc = torch.sum(pred_values == sample_values) / (pred_values.size(0) * pred_values.size(1))
		occ_idxs = torch.where(sample_values == 1)
		rec = torch.sum(pred_values[occ_idxs[0], occ_idxs[1]] == \
		  sample_values[occ_idxs[0], occ_idxs[1]]) / occ_idxs[0].size(0)
		return loss, acc, rec

	def get_train_loss(self, point_clouds, sample_points, sample_values):
		point_clouds = point_clouds.unsqueeze(1)
		_, net_out = self.forward(point_clouds, None, sample_points, is_training=True)
		loss = torch.mean((net_out-sample_values)**2)
		return loss
	

class VN_Resnet_Encoder(nn.Module):
	def __init__(self, hidden_dim, latent_dim, layer_num, num_k):
		super(VN_Resnet_Encoder, self).__init__()
		self.layer_num = layer_num
		self.num_k = num_k
		self.conv_pos = VNLinearLeakyReLU(3, hidden_dim, negative_slope=0.2, share_nonlinearity=False, use_batchnorm=False)
		self.fc_pos = VNLinear(hidden_dim, 2*hidden_dim)
        
		self.blocks = []

		for i in range(layer_num):
			self.blocks.append(
				VNResnetBlockFC(2*hidden_dim, hidden_dim)
			)

		self.pool = mean_pool
        
		self.fc_c = VNLinear(hidden_dim, latent_dim)
		
		self.actvn_c = VNLeakyReLU(hidden_dim, negative_slope=0.2, share_nonlinearity=False)
	
	def to(self, device):
		self.conv_pos = self.conv_pos.to(device)
		self.fc_pos = self.fc_pos.to(device)
		for i in range(len(self.blocks)): self.blocks[i] = self.blocks[i].to(device)
		self.fc_c = self.fc_c.to(device)
		self.actvn_c = self.actvn_c.to(device)
		return self

	def forward(self, input):
		"""
		input size (batch, N, )
		"""
		batch_size = input.size(0)
		input = input.transpose(2, 3)
		
		feat = get_graph_feature_cross(input, k=self.num_k, if_cross=True)
		hidden = self.conv_pos(feat)
		hidden = self.pool(hidden, dim=-1)
		hidden = self.fc_pos(hidden)
		
		for i in range(self.layer_num):
			hidden = self.blocks[i](hidden)
			pooled_hidden = self.pool(hidden, dim=-1, keepdim=True).expand(hidden.size())
			if i < self.layer_num-1:
				hidden = torch.cat((hidden, pooled_hidden), dim=1)
			else:
				hidden = pooled_hidden
		
		hidden = self.pool(hidden, dim=-1)
		latent = self.fc_c(self.actvn_c(hidden))
		return latent
	

class VN_DGCNN_Encoder(nn.Module):
	def __init__(self, hidden_dim, latent_dim, layer_num, num_k):
		super(VN_DGCNN_Encoder, self).__init__()
		self.layer_num = layer_num
		self.num_k = num_k
		self.blocks = []
		self.conv_pos = VNLinearLeakyReLU(2, hidden_dim)
		final_input_dim = 0
		for i in range(layer_num):
			self.blocks.append(
				VNLinearLeakyReLU(2*hidden_dim, hidden_dim)
			)
			final_input_dim += hidden_dim

		self.pool = mean_pool
        
		self.conv_c = VNLinearLeakyReLU(final_input_dim, latent_dim, dim=4, share_nonlinearity=True)

	def to(self, device):
		self.conv_pos = self.conv_pos.to(device)
		for i in range(len(self.blocks)): self.blocks[i] = self.blocks[i].to(device)
		self.conv_c = self.conv_c.to(device)
		return self

	def forward(self, input):
		"""
		input size (batch, N, )
		"""
		batch_size = input.size(0)
		try:
			input = input.transpose(2, 3)
		except:
			pdb.set_trace()
		
		feat = get_graph_feature_cross(input, k=self.num_k)
		hidden = self.conv_pos(feat)
		hidden = self.pool(hidden, dim=-1)
		hiddens = []
		
		for i in range(self.layer_num):
			hidden_feat = get_graph_feature_cross(hidden, k=self.num_k)
			hidden = self.blocks[i](hidden_feat)
			hidden = self.pool(hidden)
			hiddens.append(hidden)
		
		final_input_vecs = torch.cat(hiddens, dim=1)
		latent_vecs = self.conv_c(final_input_vecs)
		latent = latent_vecs.mean(dim=-1, keepdim=False)
		return latent