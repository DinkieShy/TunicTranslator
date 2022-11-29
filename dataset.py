import os
from torchvision.io import read_image
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import random
import math
from torch import as_tensor, float32, int64
from torchvision.transforms import Normalize

class ocrDataset():
	def __init__(self, dir, transform=None):
		assert os.path.exists(dir), "Dataset created with invalid path"
		self.transform = transform
		self.imagePaths = []
		self.labels = []

		subDirs = os.listdir(dir)
		for i in range(len(subDirs)):
			currentLabel = subDirs[i]
			currentDir = os.listdir(os.path.join(dir, currentLabel))
			for ii in range(len(currentDir)):
				self.imagePaths.append(os.path.join(dir, currentLabel, currentDir[ii]))
				self.labels.append(i)

	def __len__(self):
		return len(self.imagePaths)

	def __getitem__(self, index):
		image = read_image(self.imagePaths[index])
		label = self.labels[index]
		if self.transform:
			image = self.transform(image)

		return image, label

class syntheticTextPageSet():
	def __init__(self, dir, amountToMake, transform=None, size=(1280, 720)):
		assert os.path.exists(dir), "Dataset created with invalid path"
		self.transform = transform
		self.characters = []
		self.characterImages = {}
		self.dir = dir
		subDirs = os.listdir(dir)
		for i in subDirs:
			self.characterImages[i] = []
			self.characters.append(i)
			for ii in os.listdir(os.path.join(dir, i)):
				self.characterImages[i].append(ii)
		self.amount = amountToMake
		self.pageSize = size

	def __len__(self):
		return self.amount

	def __getitem__(self, index):
		generatedImage = Image.new('RGB', self.pageSize, color='white')
		# targets = [] # <- YOLO

		targets = {}
		boxes = []
		labels = []

		for sentences in range(random.randint(2, 4)):
			currentPosition = (random.randrange(10, self.pageSize[0]-200), random.randrange(10, self.pageSize[1]-200))
			scale = 0.75 + random.random()*0.5
			for words in range(random.randint(2, 4)):
				currentPosition = (currentPosition[0]+random.randint(10, 15), currentPosition[1])
				for characters in range(random.randint(2, 6)):
					if currentPosition[0] > generatedImage.width or currentPosition[1] > generatedImage.height:
						continue;
					nextCharacter = self.characters[random.randint(0, len(self.characters)-1)]
					characterImage = Image.open(os.path.join(self.dir, nextCharacter, self.characterImages[nextCharacter][random.randint(0, len(self.characterImages[nextCharacter])-1)]))
					characterImage.resize((math.floor(characterImage.width*scale), math.floor(characterImage.height*scale)))
					generatedImage.paste(characterImage, currentPosition)

					# --- YOLO label
					# relativeWidth, relativeHeight = characterImage.width/self.pageSize[0], characterImage.height/self.pageSize[1]
					# label = [self.characters.indexof(nextCharacter), (currentPosition[0]/self.pageSize[0])+relativeWidth/2, (currentPosition[1]/self.pageSize[1])+relativeHeight/2, relativeWidth, relativeHeight]
					# targets.append(label)
					
					# --- FasterRCNN label
					x1, y1, x2, y2 = currentPosition[0], currentPosition[1], currentPosition[0]+characterImage.width, currentPosition[1]+characterImage.height
					boxes.append([x1, y1, x2, y2])
					labels.append(self.characters.index(nextCharacter))

					currentPosition = (currentPosition[0]+characterImage.width+random.randint(0, 5), currentPosition[1])

		targets["boxes"] = as_tensor(boxes, dtype=float32)
		targets["labels"] = as_tensor(labels, dtype=int64)

		# return generatedImage
		imageTensor = pil_to_tensor(generatedImage).to(float32)

		mean, std = imageTensor.mean(), imageTensor.std()
		normaliseTransform = Normalize(mean, std)
		imageTensor = normaliseTransform(imageTensor)

		if self.transform:
			imageTensor = self.transform(imageTensor)

		return imageTensor, targets


def main():
	dataset = syntheticTextPageSet(os.path.join(os.getcwd(), "data", "training_data"), 5)

	image, labels = dataset[0]
	print(image)
	print(labels)

if __name__ == "__main__":
	main()