import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import math
from training import training
import numpy as np

"""
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

Requires python3, tkinter, skimage, and scipy
"""

class verifyMe(tk.Frame):
	def __init__(self, GUI):
		#GUI customization
		self.GUI = GUI
		self.GUI.title("Verify Me!")
		self.GUI.geometry("650x550")

		#Variable initialization for use later
		self.selectedImages = []
		self.ci = 0
		self.countTemp = 0
		self.activate = True

		#Methods to run
		self.initButtons()
		self.initHelpText()
		self.begin()

	def initButtons(self):
		#This method initializes the button images we will use throughout the application

		#Initialize images and append to label array
		self.buttonImages = []
		self.buttonImages.append(ImageTk.PhotoImage(Image.open("ButtonImgs/trainNetworkImg.jpg")))
		self.buttonImages.append(ImageTk.PhotoImage(Image.open("ButtonImgs/trainNetworkImgHover.jpg")))
		self.buttonImages.append(ImageTk.PhotoImage(Image.open("ButtonImgs/uploadButtonWR.jpg")))
		self.buttonImages.append(ImageTk.PhotoImage(Image.open("ButtonImgs/uploadButtonNR.jpg")))
		self.buttonImages.append(ImageTk.PhotoImage(Image.open("ButtonImgs/uploadButtonWB.jpg")))
		self.buttonImages.append(ImageTk.PhotoImage(Image.open("ButtonImgs/markButtonNB.jpg")))
		self.buttonImages.append(ImageTk.PhotoImage(Image.open("ButtonImgs/markButtonWB.jpg")))
		self.buttonImages.append(ImageTk.PhotoImage(Image.open("ButtonImgs/ArrowLeft.jpg")))
		self.buttonImages.append(ImageTk.PhotoImage(Image.open("ButtonImgs/ArrowRight.jpg")))
		self.buttonImages.append(ImageTk.PhotoImage(Image.open("ButtonImgs/forgeryNB.jpg")))
		self.buttonImages.append(ImageTk.PhotoImage(Image.open("ButtonImgs/genuineNB.jpg")))
		self.buttonImages.append(ImageTk.PhotoImage(Image.open("ButtonImgs/forgeryWB.jpg")))
		self.buttonImages.append(ImageTk.PhotoImage(Image.open("ButtonImgs/genuineWB.jpg")))
		self.buttonImages.append(ImageTk.PhotoImage(Image.open("ButtonImgs/finishNB.jpg")))
		self.buttonImages.append(ImageTk.PhotoImage(Image.open("ButtonImgs/finishWB.jpg")))
		self.buttonImages.append(ImageTk.PhotoImage(Image.open("ButtonImgs/VerifyNB.jpg")))
		self.buttonImages.append(ImageTk.PhotoImage(Image.open("ButtonImgs/VerifyWB.jpg")))
		self.GUIButtons = []
		for i in range(len(self.buttonImages)):
			self.GUIButtons.append(tk.Label(self.GUI, image = self.buttonImages[i]))

	def initHelpText(self):
		#Initialize instruction text 
		self.helpText = []
		self.helpText.append(tk.Label(self.GUI, font=("CALIBRI 18 bold"), text = "Welcome to VerifyMe! \n\nClick the button below to upload images for training."))
		self.helpText.append(tk.Label(self.GUI, font=("CALIBRI 16"), text = "Note: You should upload at least 3 verified signatures \nand 3 forgeries (the more the better) to optimize accuracy."))
		self.helpTextV = tk.StringVar()
		self.helpTextV.set("You have uploaded 0 images")
		self.helpText.append(tk.Label(self.GUI, font=("CALIBRI 16 bold"), textvariable = self.helpTextV))
		self.helpText.append(tk.Label(self.GUI, font=("CALIBRI 18 bold"), text = "Click the arrows to view and classify your images" ))
		self.helpText.append(tk.Label(self.GUI, font= ("CALIBRI 18 bold"), text = "Please Wait..." ))
		self.helpText.append(tk.Label(self.GUI, font= ("CALIBRI 18 bold"), text = "Click the button to upload an image to classify" ))
		self.helpTextV2 = tk.StringVar()
		self.helpText.append(tk.Label(self.GUI, font=("CALIBRI 20 bold"), textvariable = self.helpTextV2))

	def begin(self):
		#Place and bind commands to buttons
		self.helpText[0].place(x = 83, y = 50)
		self.helpText[1].place(x = 80, y = 300)
		self.helpText[2].place(x = 200, y = 370)
		self.GUIButtons[2].place(x = 200, y = 160)
		self.GUIButtons[2].bind("<Enter>", self.enterB2)
		self.GUIButtons[3].place(x = 200, y = 160)
		self.GUIButtons[3].bind("<Enter>", self.enterB2)
		self.GUIButtons[4].place(x = 200, y = 160)
		self.GUIButtons[2].lift()
		self.GUIButtons[4].bind("<Leave>", self.exitB2)
		self.GUIButtons[4].bind("<Button-1>", self.uploadImages)
		self.GUIButtons[5].place(x = 200, y = 410)
		self.GUIButtons[5].bind("<Enter>", self.enterB3)
		self.GUIButtons[6].place(x = 200, y = 410)
		self.GUIButtons[6].bind("<Leave>", self.exitB3)
		self.GUIButtons[6].bind("<Button-1>", self.markingImages)
		self.GUIButtons[5].lift()

	def enterB2(self, event):
		if self.activate:
			self.GUIButtons[2].lower()
			self.GUIButtons[2].lower()
			self.GUIButtons[3].lower()
		self.activate = True

	def exitB2(self, event):
		self.GUIButtons[3].lift()
		self.GUIButtons[3].lift()

	def enterB3(self, event):
		self.GUIButtons[6].lift()

	def exitB3(self, event):
		self.GUIButtons[5].lift()	

	def uploadImages(self, event):
		self.GUIButtons[4].lower()
		self.GUIButtons[3].lift()
		self.GUIButtons[3].lift()
		self.activate = False

		#Open dialog to upload files
		self.tempString = filedialog.askopenfilenames(initialdir = "/", title = "Choose Images for Training",
			filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
		self.tempString = list(self.tempString)
		for i in self.tempString:
			self.selectedImages.append((i))
		self.helpTextV.set("You have uploaded %s images" % len(self.selectedImages))

	def markingImages(self, event):
		if (len(self.selectedImages) > 1):
			#Reset GUI
			for i in self.GUIButtons:
				i.place_forget()
			for i in self.helpText:
				i.place_forget()
			self.GUI.geometry("550x550")

			#Initialize arrays
			self.unprocessedImages = []
			self.dispExImages = []
			self.tempImageH = []
			self.tempStringH = []

			for i in self.selectedImages:
				self.unprocessedImages.append(ImageTk.PhotoImage(Image.open(i)))
				self.tempStringH.append(Image.open(i))
			for i in range(len(self.tempStringH)):
				self.imgWidth = self.unprocessedImages[i].width()
				self.imgHeight = self.unprocessedImages[i].height()
				#Reshape image for display on GUI using sigmoid function
				self.tempImageH.append(self.tempStringH[i].resize(((int(290/(1 + math.exp(-.001 * (self.imgWidth-1000))))+120), (int(290/(1 + math.exp(-.001 * (self.imgHeight-1000))))+120))))
			for i in self.tempImageH:
				self.dispExImages.append(ImageTk.PhotoImage((i)))
			#Place and bind buttons and labels
			self.dispImages = tk.Label(self.GUI, image = self.dispExImages[self.ci])
			self.dispImages.configure(image = self.dispExImages[self.ci])
			self.dispImages.place(x = 275, y = 200, anchor=tk.CENTER)
			self.GUIButtons[7].place(x = 50, y = 200, anchor=tk.CENTER)
			self.GUIButtons[8].place(x = 500, y = 200, anchor=tk.CENTER)
			self.helpText[3].place(x = 275, y = 30, anchor=tk.CENTER)
			self.GUIButtons[9].place(x = 20, y = 400)
			self.GUIButtons[10].place(x = 285, y = 400)
			self.GUIButtons[11].place(x = 20, y = 400)
			self.GUIButtons[12].place(x = 285, y = 400)
			self.GUIButtons[7].bind("<Button-1>", self.imgLeft)
			self.GUIButtons[8].bind("<Button-1>", self.imgRight)
			self.GUIButtons[9].bind("<Enter>", self.enterB4)
			self.GUIButtons[10].bind("<Enter>", self.enterB5)
			self.GUIButtons[11].bind("<Leave>", self.exitB4)
			self.GUIButtons[11].bind("<Button-1>", self.markF)
			self.GUIButtons[12].bind("<Leave>", self.exitB5)
			self.GUIButtons[12].bind("<Button-1>", self.markG)
			self.GUIButtons[13].bind("<Enter>", self.enterB6)
			self.GUIButtons[14].bind("<Button-1>", self.preprocessing)
			self.GUIButtons[14].bind("<Leave>", self.exitB6)
			self.GUIButtons[10].lift()
			self.GUIButtons[9].lift()
			self.mark = []
			self.reliftF = []
			self.reliftG = []
			for i in range(len(self.dispExImages)):
				self.mark.append(2)
				self.reliftF.append(True)
				self.reliftG.append(True)

	def imgLeft(self, event):
		#Scroll left
		self.GUIButtons[9].lift()
		self.GUIButtons[10].lift()
		if self.ci > 0:
			self.ci-=1
		else:
			self.ci = len(self.dispExImages) - 1
		self.dispImages.configure(image = self.dispExImages[self.ci])
		if self.mark[self.ci] == 1:
			self.GUIButtons[10].lift()
			self.GUIButtons[11].lift()
		if self.mark[self.ci] == 0:
			self.GUIButtons[9].lift()
			self.GUIButtons[12].lift()

	def imgRight(self, event):
		#Scroll right
		self.GUIButtons[9].lift()
		self.GUIButtons[10].lift()
		if self.ci < len(self.dispExImages) - 1:
			self.ci+=1
		else:
			self.ci = 0
		self.dispImages.configure(image = self.dispExImages[self.ci])
		if self.mark[self.ci] == 1:
			self.GUIButtons[10].lift()
			self.GUIButtons[11].lift()
		if self.mark[self.ci] == 0:
			self.GUIButtons[9].lift()
			self.GUIButtons[12].lift()

	def enterB4(self, event):
		self.GUIButtons[9].lower()

	def enterB5(self, event):
		self.GUIButtons[10].lower()

	def exitB4(self, event):
		if self.reliftF[self.ci]:
			self.GUIButtons[9].lift()

	def exitB5(self, event):
		if self.reliftG[self.ci]: 
			self.GUIButtons[10].lift()

	def markF(self, event):
		#Mark image as a forgery
		self.GUIButtons[11].lift()
		self.mark[self.ci] = 1
		self.reliftF[self.ci] = False
		if not self.reliftG[self.ci]:
			self.GUIButtons[12].lower()
			self.reliftG[self.ci] = True
		if 2 not in self.mark:
			self.GUIButtons[13].place(x = 380, y = 10)
			self.GUIButtons[14].place(x = 380, y = 10)
			self.GUIButtons[13].lift()
			self.helpText[3].place_forget()

	def markG(self, event):
		#Mark image as genuine
		self.GUIButtons[12].lift()
		self.mark[self.ci] = 0
		self.reliftG[self.ci] = False
		if not self.reliftF[self.ci]:
			self.GUIButtons[11].lower()
			self.reliftF[self.ci] = True
		if 2 not in self.mark:
			self.GUIButtons[13].place(x = 380, y = 10)
			self.GUIButtons[14].place(x = 380, y = 10)
			self.GUIButtons[13].lift()
			self.helpText[3].place_forget()

	def enterB6(self, event):
		self.GUIButtons[14].lift()

	def exitB6(self, event):
		self.GUIButtons[13].lift()

	def preprocessing(self, event):
		self.processedImgs = []
		self.tempMark = np.zeros((len(self.mark), 1))
		for i in (range(8)):
			self.GUIButtons[i+7].place_forget()
		self.dispImages.place_forget()
		for i in range(len(self.mark)):
			self.tempMark[i][0] = self.mark[i]
		#Send image to training to train neural network
		self.t = training(self.selectedImages, self.tempMark)
		self.helpText[5].place(x = 270, y = 170, anchor=tk.CENTER)
		self.GUIButtons[15].place(x = 270, y = 275, anchor=tk.CENTER)
		self.GUIButtons[16].place(x = 270, y = 275, anchor=tk.CENTER)
		self.GUIButtons[15].bind("<Enter>", self.enterB7)
		self.GUIButtons[16].bind("<Button-1>", self.getTestImg)
		self.GUIButtons[16].bind("<Leave>", self.exitB7)
		self.GUIButtons[15].lift()

	def enterB7(self, event):
		self.GUIButtons[16].lift()

	def exitB7(self, event):
		self.GUIButtons[15].lift()

	def getTestImg(self, event):
		#Open file dialog for test image
		self.testString = ""
		self.GUIButtons[15].lift()
		self.testString = filedialog.askopenfilename(initialdir = "/", title = "Choose Images for Training",
			filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
		self.testing()

	def testing(self):
		#Run testing and display on screen
		if (self.countTemp == 0):
			self.reDo = True
		else:
			self.reDo = False
		self.prediction = self.t.returnPred([self.testString], self.reDo)
		self.helpText[6].place(x = 25, y = 360)
		if self.prediction > .5:
			self.helpTextV2.set("There is a %s%% chance that \nthis signature is a forgery" %str(self.prediction[0][0] * 100))
		else:
			self.helpTextV2.set("There is a %s%% chance that \nthis signature is genuine" %str((1 - self.prediction[0][0]) * 100))
		self.countTemp+=1

if __name__ == '__main__':
	main = tk.Tk()
	mainScreen = verifyMe(main)
	main.mainloop()