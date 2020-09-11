import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib, matplotlib.pyplot as plt
import io
import itertools
import PIL.Image
# matplotlib.use('agg')
plt.switch_backend('Agg')

def plot_confusion_matrix(cm, class_names):

	figure = plt.figure(figsize=(8, 8))
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title("Confusion matrix")
	plt.colorbar()
	tick_marks = np.arange(len(class_names))
	plt.xticks(tick_marks, class_names, rotation=45)
	plt.yticks(tick_marks, class_names)
	cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
	threshold = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		color = "white" if cm[i, j] > threshold else "black"
		plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	return figure

def plot_to_image(figure):
	"""Converts the matplotlib plot specified by 'figure' to a PNG image and
	returns it. The supplied figure is closed and inaccessible after this call."""
	# Save the plot to a PNG in memory.
	buf = io.BytesIO()
	plt.savefig(buf, format='png')
	# Closing the figure prevents it from being displayed directly inside
	# the notebook.
	plt.close(figure)
	buf.seek(0)
	image = PIL.Image.open(buf)
	image = transforms.ToTensor()(image)
	return image