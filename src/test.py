import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_accuracy(loader, net, tag):
	correct = 0
	total = 0
	with torch.no_grad():
		for data in loader:
			images, labels = data
			images = images.to(device)
			outputs = net(images)
			_, predicted = torch.max(outputs.to('cpu').data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	accuracy = 100 * (correct/total)
	print(tag,'Accuracy ', accuracy)
	return accuracy

def calculate_loss(loader, net, criterion, tag):
	total_loss = 0
	with torch.no_grad():
		for data in loader:
			images, labels = data
			images = images.to(device)
			labels = labels.to(device)
			outputs = net(images)
			loss = criterion(outputs, labels)
			total_loss += loss

	# try:
	# 	num_images, _, _, _ = loader.dataset.data.shape
	# except:
	# 	num_images, _, _ = loader.dataset.data.shape


	num_images = loader.dataset.__len__()
	avg_loss = (total_loss/num_images)* loader.batch_size
	
	print(tag, 'Loss ',avg_loss.item())
	return avg_loss.item()