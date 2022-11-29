import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FasterRCNN
from model import Model
from dataset import ocrDataset, syntheticTextPageSet
from torch.optim.lr_scheduler import ExponentialLR

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



resizeTransform = transforms.Resize((48, 48))
trainingAugment = nn.Sequential(
	transforms.RandomInvert(0.5),
	transforms.ColorJitter(hue=0.5, brightness=0.2),
	transforms.RandomRotation(45),
	resizeTransform
)

# trainingData = ocrDataset(os.path.join(os.getcwd(), "data", "training_data"), transform=trainingAugment)
# testingData = ocrDataset(os.path.join(os.getcwd(), "data", "testing_data"), transform=resizeTransform)

trainingData = syntheticTextPageSet(os.path.join(os.getcwd(), "data", "training_data"), transform=trainingAugment)
testingData = syntheticTextPageSet(os.path.join(os.getcwd(), "data", "testing_data"), transform=resizeTransform)

batchSize = 128

trainLoader = DataLoader(trainingData, batch_size=batchSize, shuffle=True)
testLoader = DataLoader(testingData, batch_size=batchSize, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)
# model = Model(numClasses=36).to(device)
model = FasterRCNN.fasterrcnn_resnet50_fpn(num_classes = 36)

lossFunction = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=1e-3)
scheduler = ExponentialLR(optimiser, 0.99)

for epoch in range(50):
	print("\nEpoch ", str(epoch), "\n", sep="")
	train(trainLoader, device, model, lossFunction, optimiser)
	test(testLoader, model, lossFunction, device)