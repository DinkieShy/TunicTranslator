import torch
from torch import nn 
import os
from torchvision.io import read_image
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import Resize

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

class Model(nn.Module):
	def __init__(self, numClasses):
		super(Model, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(1, 32, 5),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(32, 16, 5)
		)
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(10816, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, numClasses),
		)

	def forward(self, x):
		x = self.conv(x)
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits

def train(dataLoader, device, model, lossFunction, optimiser):
	model.train()
	count, totalLoss = 0, 0
	for batch, (x, y) in (progressBar := tqdm(enumerate(dataLoader))):
		count += 1
		x = x.to(device).to(torch.float32)
		y = y.to(device)

		pred = model(x)
		loss = lossFunction(pred, y)
		totalLoss += loss.item()

		optimiser.zero_grad()
		loss.backward()
		optimiser.step()
		progressBar.set_postfix({"Loss": totalLoss/count})

def test(dataLoader, model, lossFunction, device):
	model.eval()
	testLoss, correct = 0, 0
	with torch.no_grad():
		for (x, y) in dataLoader:
			x = x.to(device).to(torch.float32)
			y = y.to(device)
			pred = model(x)
			testLoss += lossFunction(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()

	testLoss /= len(dataLoader)
	correct /= len(dataLoader.dataset)

	print("Test Error: \n  Correct rate: ", str(100*correct), "%\n  Average Loss: ", str(testLoss), sep="")


resizeTransform = Resize((64, 64))

trainingData = ocrDataset(os.path.join(os.getcwd(), "data", "training_data"), transform=resizeTransform)
testingData = ocrDataset(os.path.join(os.getcwd(), "data", "testing_data"), transform=resizeTransform)

trainLoader = DataLoader(trainingData, batch_size=16, shuffle=True)
testLoader = DataLoader(testingData, batch_size=16, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)
model = Model(numClasses=36).to(device)

lossFunction = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=1e-5)

for epoch in range(50):
	print("\nEpoch ", str(epoch), "\n", sep="")
	train(trainLoader, device, model, lossFunction, optimiser)
	test(testLoader, model, lossFunction, device)