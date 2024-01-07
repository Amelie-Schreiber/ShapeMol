import os
import time
import math
import random
import numpy as np
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from utils import *

class generator(nn.Module):
	def __init__(self, z_dim, point_dim, gf_dim):
		super(generator, self).__init__()
		self.z_dim = z_dim
		self.point_dim = point_dim
		self.gf_dim = gf_dim
		self.linear_1 = nn.Linear(self.z_dim+self.point_dim, self.gf_dim*8, bias=True)
		self.linear_2 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_3 = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
		self.linear_4 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
		self.linear_5 = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
		self.linear_6 = nn.Linear(self.gf_dim*1, 1, bias=True)
		nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_2.bias,0)
		nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_3.bias,0)
		nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_4.bias,0)
		nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_5.bias,0)
		nn.init.normal_(self.linear_6.weight, mean=1e-5, std=0.02)
		nn.init.constant_(self.linear_6.bias,0)

	def forward(self, points, z, is_training=False):
		zs = z.view(-1,1,self.z_dim).repeat(1,points.size()[1],1)

		pointz = torch.cat([points,zs],2)

		l1 = self.linear_1(pointz)
		l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

		l2 = self.linear_2(l1)
		l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

		l3 = self.linear_3(l2)
		l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

		l4 = self.linear_4(l3)
		l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

		l5 = self.linear_5(l4)
		l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

		l6 = self.linear_6(l5)

		l6 = torch.max(torch.min(l6, l6*0.01+0.99), l6*0.01)
		
		return l6.squeeze(-1)

class encoder(nn.Module):
	def __init__(self, ef_dim, z_dim):
		super(encoder, self).__init__()
		self.ef_dim = ef_dim
		self.z_dim = z_dim
		self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=False)
		self.in_1 = nn.InstanceNorm3d(self.ef_dim)
		self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, 4, stride=2, padding=1, bias=False)
		self.in_2 = nn.InstanceNorm3d(self.ef_dim*2)
		self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=False)
		self.in_3 = nn.InstanceNorm3d(self.ef_dim*4)
		self.conv_4 = nn.Conv3d(self.ef_dim*4, self.z_dim, 4, stride=2, padding=0, bias=True)
		nn.init.xavier_uniform_(self.conv_1.weight)
		nn.init.xavier_uniform_(self.conv_2.weight)
		nn.init.xavier_uniform_(self.conv_3.weight)
		nn.init.xavier_uniform_(self.conv_4.weight)
		nn.init.constant_(self.conv_4.bias,0)

	def forward(self, inputs, is_training=False):
		d_1 = self.in_1(self.conv_1(inputs))
		d_1 = F.leaky_relu(d_1, negative_slope=0.02, inplace=True)

		d_2 = self.in_2(self.conv_2(d_1))
		d_2 = F.leaky_relu(d_2, negative_slope=0.02, inplace=True)
		
		d_3 = self.in_3(self.conv_3(d_2))
		d_3 = F.leaky_relu(d_3, negative_slope=0.02, inplace=True)

		d_4 = self.conv_4(d_3)
		d_4 = d_4.view(-1, self.z_dim)
		d_4 = torch.sigmoid(d_4)

		return d_4


class IM_AE(nn.Module):
	"""
	voxel-based autoencoder trained to predict whether each voxel is occupied or not
	"""
	def __init__(self, config):
		super(IM_AE, self).__init__()
		self.ef_dim = config.ef_dim
		self.gf_dim = config.gf_dim
		self.z_dim = config.z_dim
		self.point_dim = config.point_dim
		self.encoder = encoder(self.ef_dim, self.z_dim)
		self.generator = generator(self.z_dim, self.point_dim, self.gf_dim)

	def forward(self, inputs, z_vector, point_coord, is_training=False):
		if is_training:
			z_vector = self.encoder(inputs, is_training=is_training)
			net_out = self.generator(point_coord, z_vector, is_training=is_training)
		else:
			if inputs is not None:
				z_vector = self.encoder(inputs, is_training=is_training)
			if z_vector is not None and point_coord is not None:
				net_out = self.generator(point_coord, z_vector, is_training=is_training)
			else:
				net_out = None

		return z_vector, net_out
	
	def get_val_loss(self, shape_voxels, shape_points, shape_values, max_dist=11, recon_num=5):
		shape_voxels = shape_voxels.unsqueeze(1)
		z_vectors, net_out = self.forward(shape_voxels, None, shape_points, is_training=False)
		loss = torch.mean((net_out-shape_values)**2)
		
		acc, num = 0, 0
		voxel_grid_num = shape_voxels.shape[2] * shape_voxels.shape[3] * shape_voxels.shape[4]
		for i in range(min(recon_num, z_vectors.shape[0])):
			z_vector = z_vectors[i, :].unsqueeze(0)
			shape_voxel = shape_voxels[i, :, :, :]
			reconstruct_voxel = self.get_reconstruct_voxel(z_vector, shape_voxel, max_dist)
			
			acc += (1 - torch.sum(torch.abs(reconstruct_voxel - shape_voxel)) / voxel_grid_num)
			num += 1
		return loss, acc / num, 0

	def get_train_loss(self, shape_voxels, shape_points, shape_values):
		shape_voxels = shape_voxels.unsqueeze(1)
		_, net_out = self.forward(shape_voxels, None, shape_points, is_training=True)
		loss = torch.mean((net_out-shape_values)**2)
		return loss

	def get_reconstruct_voxel(self, z_vector, shape_voxel, max_dist, is_training=False):
		z_vector = self.encoder(shape_voxel, is_training=is_training)
		point_coords = torch.zeros(shape_voxel.shape[1:] + (3, ))
        
		for i in range(shape_voxel.shape[1]):
			for j in range(shape_voxel.shape[2]):
				for k in range(shape_voxel.shape[3]):
					point_coords[i, j, k, :] = torch.tednsor([i, j, k])
		
		point_coords = point_coords.view(-1, 3).unsqueeze(0)
		if torch.cuda.is_available(): point_coords = point_coords.to('cuda')
		net_out = self.generator(point_coords, z_vector, is_training=is_training)
		reconstruct_voxel = (net_out > 0.5).long().view(shape_voxel.shape)
		return reconstruct_voxel