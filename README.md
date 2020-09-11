# An interactive web tool to visualize ML performance on images
## Introduction
The objective is to build an easy to use interactive tool that lets users upload their data and play with in-built deep learning algorithms (like VGG/Resnet), parameters or even use pre-trained models to visualize the impact of these machine learning models on different image datasets.

A user can play with existing popular datasets (like MNIST/CIFAR10/Fashion MNIST) or upload their dataset. They can choose the computer vision task of their choice. He/She can also decide to train the model from scratch by tweaking hyperparameters or use a saved model to visualize the results.

In regard to the tech-stack, the tool uses HTML/CSS/JQuery for the UI, PyTorch for deep learning, Flask to handle backend API requests and TensorBoard  to render the visualizations. The tool also has GPU support and can be easily hosted on a VM like Google Compute Engine running Telsa K80.

Read this [writeup](Visualizing%20the%20impact%20of%20training%20images%20on%20ML%20performance.pdf) for more info
## How-To
Open Terminal (Linux/Mac) or WSL (Windows). Make sure python version is 3.0+, virtualenv/venv and git is installed.

**Step 1: Clone the repository**<br>
``` git clone https://github.iu.edu/ML2Viz/ml-visualization-web.git```

**Step 2: Run Script**<br>
```cd ml-visualization-web```<br>
```sh start-app.sh```

**Step 3: Paste url in browser**<br>
Copy and paste http://localhost:5000 on google chrome

## Future Work
The most exciting aspect of this project is the endless possibilities in terms of functionality which could be added to the tool. We could extend the project to create visualizations for different computer vision tasks like object detection, image segmentation etc. We could add support for modern and state of the art architectures. And lastly, we could collaborate with different researchers around the world who are working with interpretable AI to add new and informative visualizations as part of the tool.

## FAQ and Known Issues

 1. Currently the tool supports Image Classification
 2. If a user uploads a custom dataset, he/she needs to ensure that the folder hierarchy is such that the images are inside sub-directories for different image categories. The tool wrangles through the different sub-directories and creates a train/test set with an 80% and 20% split. The images are resized to 64x64 resolution and pixel intensities for each channel are standardized. The tool can support both RGB and grayscale images.
 3. Commands inside the `start-app.sh` script might need modification if operating system is Windows/Linux. For Linux replace `source env/bin/activate` with `. env/bin/activate`

## Contact
Please reach out to srmanj@iu.edu for questions and feedback.
