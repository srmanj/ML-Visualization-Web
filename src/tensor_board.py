import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import sklearn.metrics
from viz import plot_confusion_matrix, plot_to_image
# import tensorflow as tf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_graph_viz(net, trainloader, logdir):
	writer = SummaryWriter(logdir)
	dataiter = iter(trainloader)
	images, labels = dataiter.next()

	writer.add_graph(net, images.to(device))
	writer.close()

def create_scalars(some_tag, loss, iteration, logdir):
	writer = SummaryWriter(logdir)
	writer.add_scalar(some_tag, loss, iteration)
	writer.flush()

def create_multiple_scalars(some_tag, some_dic, iteration, logdir):
	writer = SummaryWriter(logdir)
	writer.add_scalars(some_tag, some_dic, iteration)
	writer.flush()

# def select_n_random(data, labels, n):
# 	'''
# 	Selects n random datapoints and their corresponding labels from a dataset
# 	'''
# 	# assert len(data) == len(labels)

# 	perm = torch.randperm(len(data)).numpy()
# 	labels = np.asarray(labels)
# 	# print(type(data), type(labels))
# 	return data[perm][:n], labels[perm][:n]

def create_image_confusion_matrix(logdir, labels, predictions, class_names):
	writer = SummaryWriter(logdir)

	cm = sklearn.metrics.confusion_matrix(labels, predictions)
	# Log the confusion matrix as an image summary.
	figure = plot_confusion_matrix(cm, class_names=class_names)
	cm_image = plot_to_image(figure)

	writer.add_image("Confusion Matrix", cm_image)
	writer.close()

def create_3d_projector(loader, num, net, logdir, cm):
	
	try:
		classes = list(loader.dataset.class_to_idx.keys())
	except:
		classes = list(loader.dataset.dataset.class_to_idx.keys())

	with torch.no_grad():
		for data in loader:
			images, labels = data
			images = images.to(device)
			outputs = net(images)
			_, predictions = torch.max(outputs.to('cpu').data, 1)
			class_labels = [classes[lab] for lab in labels]
			features = net.output_final_layer.to('cpu')
			break

	images = images.to('cpu')
	# print(images.shape)
	try:
		_, _, _, _ = images.shape
		# images_new = images.permute(0,3,1,2)
		images_new = images
	except:
		images_new = images.unsqueeze(1)
	if cm == True:
		create_image_confusion_matrix(logdir, labels.numpy(), predictions.numpy(), classes)

	writer = SummaryWriter(logdir)
	writer.add_embedding(features,
						metadata=class_labels,
						label_img=images_new)
	writer.close()





