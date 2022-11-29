from torch import nn

class Model(nn.Module):
	def __init__(self, numClasses):
		super(Model, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(1, 16, 3),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(16, 4, 6),
			nn.MaxPool2d(2, 2)
		)
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(324, 100),
			nn.ReLU(),
			nn.Linear(100, 200),
			nn.ReLU(),
			nn.Linear(200, numClasses),
		)

	def forward(self, x):
		x = self.conv(x)
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits