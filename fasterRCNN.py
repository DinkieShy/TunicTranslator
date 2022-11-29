import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import syntheticTextPageSet
from torch.optim.lr_scheduler import ExponentialLR
from numpy import mean

def train(dataLoader, device, model, optimiser):
	count, totalLoss = 0, 0
	trainingLossList = []
	for batch, (images, targets) in (progressBar := tqdm(enumerate(dataLoader), total=len(dataLoader))):
		count += 1
		images = list(image.to(device) for image in images)
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

		lossDict = model(images, targets)

		losses = sum(loss for loss in lossDict.values())
		lossValue = losses.item()
		totalLoss += lossValue
		trainingLossList.append(lossValue)

		optimiser.zero_grad()
		losses.backward()
		optimiser.step()
		progressBar.set_postfix({"Loss": totalLoss/count})

	return trainingLossList

def test(dataLoader, model, device):
	testLoss, count = 0, 0
	testingLossList = []
	with torch.no_grad():
		for batch, (images, targets) in (progressBar := tqdm(enumerate(dataLoader), total=len(dataLoader))):
			count += 1
			images = list(image.to(device) for image in images)
			targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

			lossDict = model(images, targets)

			losses = sum(loss for loss in lossDict.values())
			lossValue = losses.item()
			testLoss += lossValue
			testingLossList.append(lossValue)
			progressBar.set_postfix({"Loss": testLoss/count})
	return testingLossList

	# testLoss /= len(dataLoader)
	# correct /= len(dataLoader.dataset)

	# print("Test Error: \n  Correct rate: ", str(100*correct), "%\n  Average Loss: ", str(testLoss), sep="")

resizeTransform = transforms.Resize((1280, 720))
trainingAugment = nn.Sequential(
	transforms.RandomInvert(0.5),
	transforms.ColorJitter(hue=0.5, brightness=0.2),
	transforms.RandomRotation(45),
	resizeTransform
)

trainingData = syntheticTextPageSet(os.path.join(os.getcwd(), "data", "training_data"), 1000, transform=trainingAugment)
testingData = syntheticTextPageSet(os.path.join(os.getcwd(), "data", "testing_data"), 200)

batchSize = 2

trainLoader = DataLoader(trainingData, batch_size=batchSize, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
testLoader = DataLoader(testingData, batch_size=batchSize, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)
model = fasterrcnn_resnet50_fpn(weights=None)
inputFeatures = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(inputFeatures, 36)
model.to(device)
model.train()

lossFunction = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=1e-3)
scheduler = ExponentialLR(optimiser, 0.99)

for epoch in range(50):
	print("\nEpoch ", str(epoch), sep="")
	trainingLossList = train(trainLoader, device, model, optimiser)
	testingLossList = test(testLoader, model, device)

	print(f"Epoch #{epoch} train loss: {mean(trainingLossList):.3f}")   
	print(f"Epoch #{epoch} test loss: {mean(testingLossList):.3f}") 