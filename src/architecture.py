import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import numpy as np



class NeuralNet(nn.Module):
	def __init__(self, input_size, output_size, hidden_size, num_mini_batches):
		super(NeuralNet, self).__init__()

		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		# self.num_images = num_images
		# self.batch_size = batch_size
		self.num_mini_batches = num_mini_batches

		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, hidden_size)
		self.fc4 = nn.Linear(hidden_size, hidden_size)
		self.fc5 = nn.Linear(hidden_size, hidden_size)
		self.output_layer = nn.Linear(hidden_size, output_size)

	def forward(self, data):
		data = data.view(-1, self.input_size)
		self.output_1st_hidden = F.relu(self.fc1(data))
		self.output_2nd_hidden = F.relu(self.fc2(self.output_1st_hidden))
		self.output_3rd_hidden = F.relu(self.fc3(self.output_2nd_hidden))
		self.output_4th_hidden = F.relu(self.fc4(self.output_3rd_hidden))
		self.output_final_layer = F.relu(self.fc5(self.output_4th_hidden))
		self.softmax_output = F.log_softmax(self.output_layer(self.output_final_layer), dim = 1)
		return self.softmax_output

class VGGModule(nn.Module):
	def __init__(self, vgg16, output_size, num_mini_batches):
		super(VGGModule,self).__init__()
		self.num_mini_batches = num_mini_batches
		self.output_size = output_size
		self.net = vgg16
		self.last_layer = nn.Linear(self.net.classifier[-1].out_features, self.output_size)

	def forward(self, data):
		self.output_final_layer = self.net(data)
		# print('Passed Thru VGG')
		self.softmax_output =  F.log_softmax(self.last_layer(self.output_final_layer), dim = 1)
		return self.softmax_output

def get_arch(architecture, trainloader):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Device: ", device)
	print("Architecture: ", architecture)
	
	try:
		try:
			num_images, w, h, num_channels = trainloader.dataset.data.shape
		except:
			num_channels = 1
			num_images, w, h = trainloader.dataset.data.shape

	except:
		num_images = trainloader.dataset.__len__()
		num_channels,w,h = trainloader.dataset.__getitem__(0)[0].shape
	
	input_size = w*h*num_channels
	try:
		output_size = len(np.unique(trainloader.dataset.targets))
	except:
		output_size = len(np.unique(trainloader.dataset.dataset.targets))

	batch_size = trainloader.batch_size

	num_mini_batches = num_images/batch_size

	print('Input Size', '# classes', '#training images', 'batch size', '#mini batches')
	print(input_size, output_size, num_images, batch_size, num_mini_batches)


	if architecture == 'fcn':
		net = NeuralNet(input_size, output_size, 1024, num_mini_batches).to(device)

		return net
	
	elif architecture == 'vgg':
		vgg16 = torchvision.models.vgg16(pretrained = True)
		# Un-Freeze last 3 fc layers
		for name, child in vgg16.named_children():
			for name2, params in child.named_parameters():
		#         print(name, name2)
		#         if ((name == 'features') & (int(name2.split('.')[0]) > 24)) or (name == 'classifier'):
				if (name == 'classifier') & (int(name2.split('.')[0]) > 0):
					params.requires_grad = True
				else:
					params.requires_grad = False
					
		vgg_model = VGGModule(vgg16, output_size, num_mini_batches).to(device)

		print('Trainable Parameters', sum(p.numel() for p in vgg_model.parameters() if p.requires_grad))
		
		return vgg_model
	elif architecture == 'resnet':
		resnet = RESNET()
		return resnet