Convolutional Neural Networks (CNNs) are a specialized class of artificial neural networks designed primarily for image processing and computer vision tasks. CNNs exploit the spatial structure of data, making them highly effective for tasks such as image classification, object detection, and pattern recognition.

A CNN consists of several types of layers arranged sequentially with blocks - Convolutional layers, Pooling layers, and Fully connected layers - for extracting features from input data and perform classification.

![cnn](https://github.com/user-attachments/assets/67e8162c-0e53-48ac-a720-35a3ab6103fd)

The convolutional layer is the core component of a CNN which applies a set of learnable filters (kernels) to the input image. Each filter slides across the image and performs element-wise multiplication followed by summation, producing a feature map. 

![cnn2](https://github.com/user-attachments/assets/b1cbc5fa-c10b-41dc-b211-3ab65989f82a)

An activation function, most commonly ReLU (Rectified Linear Unit), is applied after convolution.

Pooling layers reduce the spatial dimensions of feature maps while retaining important information. Max pooling is commonly used method. It selects the maximum value within a window.
