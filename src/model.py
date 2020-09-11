import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import numpy as np

from getdata import get_data
from train import lets_train
from test import get_accuracy
from architecture import get_arch
from serve import serve_json
from tensor_board import create_graph_viz, create_3d_projector
import os
from datetime import datetime

def CVtrain(dataset, task, architecture, epochs, batch_size, optimizer, rate, viz, num, PATH):

	trainloader, testloader = get_data(dataset, batch_size, num)

	# define architecture

	logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
	print('log file for Tensor Board at ', logdir)

	# print(net)
	if PATH is None:
		net = get_arch(architecture, trainloader)

		trained_net = lets_train(trainloader, testloader, task, net, epochs, optimizer, rate, logdir)
	else:
	# PATH = './data/abc.pth'
	# torch.save(trained_net, PATH)
		# print(PATH)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		trained_net = torch.load(PATH).to(device)

	# accuracy = lets_test(testloader, trained_net)
	create_graph_viz(trained_net, trainloader, logdir)
	create_3d_projector(testloader, num, trained_net, logdir, cm = True)

	json_output = serve_json(testloader, trained_net, viz, num)

	return json_output