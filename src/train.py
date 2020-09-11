import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import numpy as np
from tensor_board import create_scalars, create_multiple_scalars
from test import get_accuracy, calculate_loss

def lets_train(trainloader, testloader, task, net, epochs, opt, rate, logdir):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	if task == 'classification':
		criterion = nn.CrossEntropyLoss().to(device)
	else:
		criterion = None
	

	if opt == 'SGD':
		optimizer = optim.SGD(net.parameters(), lr=rate, momentum=0.9)
	elif opt == 'adam':
		optimizer = optim.Adam(net.parameters(), lr=rate)


	br = int(0.25* net.num_mini_batches)
	print('Training...')

	for epoch in range(epochs):  # loop over the dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data

			inputs = inputs.to(device)
			labels = labels.to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % br == br-1:    
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / br))
				running_loss = 0.0
		if True:
			# create_scalars('Train/loss', loss.item(), epoch, logdir)
			# create_scalars('Test/loss', calculate_loss(testloader, net, criterion, 'Test'), epoch, logdir)
			# create_scalars('Train/accuracy', get_accuracy(trainloader, net, 'Train'), epoch, logdir)
			# create_scalars('Test/accuracy', get_accuracy(testloader, net, 'Test'), epoch, logdir)
			create_multiple_scalars('Loss', {'Train': loss.item(),
												'Test': calculate_loss(testloader, net, criterion, 'Test')}, epoch, logdir)
			create_multiple_scalars('Accuracy', {'Train': get_accuracy(trainloader, net, 'Train'),
												'Test': get_accuracy(testloader, net, 'Test')}, epoch, logdir)
	print('Finished Training!')

	return net