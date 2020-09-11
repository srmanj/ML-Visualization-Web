import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import numpy as np

import json
from torchvision.utils import save_image

from sklearn.manifold import TSNE

def serve_json(testloader, net, viz, num_images):

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	try:
		classes = list(testloader.dataset.class_to_idx.keys())
	except:
		classes = list(testloader.dataset.dataset.class_to_idx.keys())
	print('Classes', classes)

	tsne = TSNE(n_components = 3)


	correct = 0
	total = 0
	i = 1
	myListObj = []

	print('Generating JSON...')

	# num_images = len(testloader.dataset.targets)
	with torch.no_grad():
	    for data in testloader:
	        images, labels = data
        	images = images.to(device)
        	labels = labels
	        outputs = net(images)
	        _, predictions = torch.max(outputs.to('cpu').data, 1)
	        reduced_dimensions = tsne.fit_transform(net.output_final_layer.to('cpu'))

	        ground_truth = [classes[label] for label in labels]
	        predicted_labels = [classes[prediction] for prediction in predictions]
	        # j = 0
	        for img in images:
	            
	            fname = './data/sample/'+ str(i) + '.png'
	            # save_image(img, fname)
	            myDictObj = {"url": fname, 
	                         "ground_truth": ground_truth[i-1], 
	                         "predicted_label": predicted_labels[i-1],
	                         "x": str(reduced_dimensions[i-1][0]),
	                         "y": str(reduced_dimensions[i-1][1]),
	                         "z": str(reduced_dimensions[i-1][2])}
	            myListObj.append(myDictObj)
	            i += 1
	            # j +=1

	        # print(str(i-1) + ' out of ' + str(num_images))
	        # if i > num_images:
	        # 	break
	        
	        total += labels.size(0)
	        correct += (predictions == labels).sum().item()
	        break
	# print(correct/total)
	# with open('sample_new.json', 'w') as f:
	#     json.dump(myListObj, f, indent=3)
	print('Accuracy: ', (correct/total)*100)
	print('Done!')
	return  json.dumps(myListObj, indent = 3)