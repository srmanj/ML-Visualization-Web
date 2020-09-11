import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import numpy as np


def get_data(dataset, batch_size, num):
	
	if dataset == 'cifar10':
		transform = transforms.Compose(
	    [transforms.ToTensor(),
	     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
		                                        download=True, transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
		                                          shuffle=True, num_workers=2)

		testset = torchvision.datasets.CIFAR10(root='./data', train=False,
		                                       download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=num,
		                                         shuffle=True, num_workers=2)



	elif dataset == 'mnist':
		transform = transforms.Compose(
	    [transforms.ToTensor(),
	     transforms.Normalize((0.5,), (0.5,))])

		trainset = torchvision.datasets.MNIST(root='./data', train=True,
		                                        download=True, transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
		                                          shuffle=True, num_workers=2)

		testset = torchvision.datasets.MNIST(root='./data', train=False,
		                                       download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=num,
		                                         shuffle=True, num_workers=2)


	elif dataset == 'fmnist':
		transform = transforms.Compose(
	    [transforms.ToTensor(),
	     transforms.Normalize((0.5,), (0.5,))])

		trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
		                                        download=True, transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
		                                          shuffle=True, num_workers=2)

		testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
		                                       download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=num,
		                                         shuffle=True, num_workers=2)

	else:
		dataset = torchvision.datasets.ImageFolder(root=dataset+'/', 
                                           transform=transforms.Compose([transforms.Resize((64,64)),
                                           								 transforms.ToTensor(),
                                           								 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
		num_images = len(dataset.imgs)
		train_images = int(0.8*num_images)
		test_images = num_images-train_images

		trainset, testset = torch.utils.data.random_split(dataset, [train_images, test_images])
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
		testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                         shuffle=False, num_workers=2)
	return trainloader, testloader