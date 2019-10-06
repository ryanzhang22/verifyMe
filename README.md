# verifyMe

VerifyMe is a signature verification tool that uses an artificial neural network
to classify a signature as genuine or a forgery. The user begins by uploading a 
set of training images (usually a combination of both genuine and forged signatures).
The neural network then trains itself using forward and backward propagation, obtaining
a set of weights and biases. The user then uploads the image that they'd like to test
into the system, and a prediction on whether or not the signature is genuine is obtained.

It is important that the user uploads a variety of images so that the neural network 
has ample training data to make accurate predictions. The more images that are uploaded
and classified, the better the model will perform. However, due to the limited number 
of training examples and inherently imperfect nature of the neural network, images
are sometimes misclassified. The best way to avoid this is, as always, to use more
training images.

This project was one of my first dives into neural networks and was done mainly for fun.
I will come back to this project and update with the knowledge I have now.
