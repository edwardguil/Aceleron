from python.Layers import *
from python.Losses import *

X = [
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 4.0, 2.5],
    [3.2, 4.2, 2.5, 8.1],
]

y = [
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 1, 0]
    ]

y2 = [3, 1, 2]

layer1 = Dense(4, 16)
layer2 = ReLU()
layer3 = Dense(16, 4)
layer4 = Softmax()
loss = CategoricalCrossentropy()


out = layer1.forward(X)
out = layer2.forward(out)
out = layer3.forward(out)
out = layer4.forward(out)
loss = loss.calculateLoss(y, out)
print("Loss 1:", "\n", loss, "\n")
yo = SparseCategoricalCrossentropy()

print("Loss 2:", "\n", yo.calculateLoss(y2, out), "\n")