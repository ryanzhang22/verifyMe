import numpy as np
from skimage.morphology import thin, skeletonize_3d
from skimage import color, io
from skimage.transform import resize
from skimage.filters import threshold_otsu, gaussian
from skimage.util import invert
import skimage
import os
from skimage.measure import regionprops
import sys
from scipy import ndimage
from scipy.stats import skew

class training(object):
	def __init__(self, strings, Y):
		#Initialize X, Y, and receive passed in parameters
		filePaths = strings
		X = np.zeros((9, len(strings)))
		self.lamb = .4
		Y = np.transpose(Y)
		images = []
		#Start run process
		self.runNN(Y, filePaths, X, images)

	def preprocess(self, filePath, images, X, m, imgRatio):
		#Obtain file path
		filename = os.path.join(skimage.data_dir, filePath[0])

		#Transform filepath to image
		imgTemp = io.imread(filename)

		#Perform gaussian denoising on image to reduce random variations
		imgTemp = gaussian(imgTemp, 1)

		#Transform image to black and white
		imgTemp = color.rgb2gray(imgTemp)

		#Invert colors of image
		imgTemp = invert(imgTemp)

		#Calculate the threshold value with the otsu method
		meanBright = threshold_otsu(imgTemp)

		#Perform thresholding on image
		img = imgTemp > meanBright

		#Trim image to signature size
		for i in range(img.shape[0]):
			if 1 not in img[0]:
				img = img[1:,:]
		for i in range(img.shape[0]):
			if 1 not in img[img.shape[0]-1]:
				img = img[:img.shape[0]-1,:]
		column = True
		while column:
			for i in range(img.shape[0]):
				if img[i][0] == 1:
					column = False
			img = img[:, 1:]
		column = True
		while column:
			for i in range(img.shape[0]):
				if img[i][img.shape[1] - 1] == 1:
					column = False
			img = img[:, :img.shape[1] - 1]
			
		#Obtain image ratio
		imgRatio.append(img.shape[1]/img.shape[0])
		#Reshape image to 200x300
		resizeImg = resize(img, (200, 300))
		# perform skeletonization
		finalImg = skeletonize_3d(resizeImg)
		images.append(finalImg)
		del filePath[0]
		if (len(filePath) == 0):
			#If filePath array exhausted, move on the feature extraction
			return self.extractFeatures(images, 0, X, m, imgRatio)
		#If filePath array not exhausted, recursively run
		return self.preprocess(filePath, images, X, m, imgRatio)

	def extractFeatures(self, image, count, X, m, imgRatio):
		#Extract feature set
		extractImg = image[0]
		X[0, count] = np.sum(extractImg)
		extractions = regionprops(extractImg)
		for i in extractions:
			X[1, count] = i.centroid[0]
			X[2, count] = i.centroid[1]
			X[3, count] = i.eccentricity
			X[4, count] = i.major_axis_length
			X[5, count] = i.minor_axis_length
		X[6, count] = extractImg.mean()
		X[7, count] = imgRatio[count]
		del image[0]
		count += 1
		if(count == m):
			return X
		return self.extractFeatures(image, count, X, m, imgRatio)

	def sigmoid(self, z):
		#Sigmoid function
		return 1/(1+np.exp(-z))

	def featureScaling(self, X):
		#Perform mean normalization
		self.tempMax = np.zeros(9)
		self.tempMean = np.zeros(9)
		self.tempMin = np.zeros(9)
		for i in range(8):
			self.tempMean[i] = np.average(X[i])
			self.tempMax[i] = np.max(X[i])
			self.tempMin[i] = np.min(X[i])
		for i in range(8):
			X[i] = (X[i] - self.tempMean[i])/(self.tempMax[i] - self.tempMin[i])
		return X

	def initialize(self, sizeH, sizeX, sizeY):
		#Initialize weights and biases
		W1 = np.random.randn(sizeH, sizeX)
		B1 = np.zeros((sizeH, 1))
		W2 = np.random.randn(sizeY, sizeH)
		B2 = np.zeros((sizeY, 1))
		return W1, B1, W2, B2

	def forwardProp(self, X, B1, W1, B2, W2):
		#Perform forwardprop
		Z1 = np.dot(W1, X) + B1
		A1 = np.tanh(Z1)
		Z2 = np.dot(W2, A1) + B2
		A2 = self.sigmoid(Z2)
		return Z1, A1, Z2, A2

	def calcCost(self, m, A2, Y, W1, W2):
		#Calculate the cost with regularization
		regularize = self.lamb * (1/(2*m)) * (np.sum(np.square(W1)) + np.sum(np.square((W2))))
		cost = -(1/m) * np.sum((np.log(A2) * Y) + ((1-Y) * np.log(1-A2))) + regularize
		return cost

	def backProp(self, X, A2, Y, A1, W2, W1):
		#Perform backprop with regularization
		m = X.shape[1]
		dZ2 = A2 - Y
		dW2 = (1/m) * np.dot(dZ2, np.transpose(A1)) + (1/m) * (self.lamb * W2)
		dB2 = (1/m) * np.sum(dZ2)
		dZ1 = np.dot(np.transpose(W2), dZ2) * (1 - np.power(A1, 2))
		dW1 = (1/m) * np.dot(dZ1, np.transpose(X)) + (1/m) * (self.lamb * W1)
		dB1 = (1/m) * np.sum(dZ1)
		return dZ2, dW2, dB2, dZ1, dW1, dB1

	def gradDesc(self, dW1, dB1, dW2, dB2, W1, W2, B1, B2):
		#Perform gradient descent
		alpha = .1
		W1 = W1 - alpha * dW1
		B1 = B1 - alpha * dB1
		W2 = W2 - alpha * dW2
		B2 = B2 - alpha * dB2
		return W1, B1, W2, B2

	def runNN(self, Y, filePath, X, images):
		#Initialize neural network to X size input, 1 hidden layers with 4 nodes, 1 output node
		count = 0
		sizeX = X.shape[0]
		sizeH = 4 
		sizeY = Y.shape[0]
		m = len(filePath)
		#Set X training parameters to the extracted features from processed images
		X = self.preprocess(filePath, images, X, m, [])
		#Set initial weights and biases
		W1, B1, W2, B2 = self.initialize(sizeH, sizeX, sizeY)
		#Scale features with mean normalization
		X = self.featureScaling(X)
		#Set each node's inputs and outputs, compute output with random variables (forward prop)
		Z1, A1, Z2, A2 = self.forwardProp(X, B1, W1, B2, W2)
		#Initialize cost
		cost = .1
		while count < 15000 and cost > .005:
			#Compute forward propagation again
			Z1, A1, Z2, A2 = self.forwardProp(X, B1, W1, B2, W2)
			#Calculate the cost
			cost = self.calcCost(m, A2, Y, W1, W2)
			#Perform backpropagation to determine partial derivatives
			dZ2, dW2, dB2, dZ1, dW1, dB1 = self.backProp(X, A2, Y, A1, W2, W1)
			#Update parameters with backprop determined updates
			W1, B1, W2, B2 = self.gradDesc(dW1, dB1, dW2, dB2, W1, W2, B1, B2)
			count+=1
		output = A2
		self.W1 = W1
		self.W2 = W2
		self.B1 = B1
		self.B2 = B2
		return cost

	def returnPred(self, filePath, first):
		#Return final predication
		self.testSlice = 0
		X = np.zeros((9, 1))
		if first:
			self.tempMean = [self.tempMean]
			self.tempMean = np.transpose(self.tempMean)
			self.tempMax = [self.tempMax]
			self.tempMax = np.transpose(self.tempMax)
			self.tempMin = [self.tempMin]
			self.tempMin = np.transpose(self.tempMin)
		X = self.preprocess(filePath, [], X, 1, [])
		X[0:8] = (X[0:8] - (self.tempMean[0:8]))/(((self.tempMax[0:8])) - (self.tempMin[0:8]))
		Z1, A1, Z2, A2 = self.forwardProp(X, self.B1, self.W1, self.B2, self.W2)
		return A2