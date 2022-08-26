from Layers import *
from Losses import *
from Metrics import *

'''
W = [[-0.05078057],
 [ 0.64329567],
 [ 1.20661244],
 [ 0.67322229]]

[[-0.05078057, -0.45289629],
 [ 0.64329567,  0.69494955],
 [ 1.20661244,  0.107874  ],
 [ 0.67322229,  0.65965736]]
'''
X = [
    [1.0, 2.0, 3.0],
    [-4.0, -5.0, -6.0],
    [7.0, 8.0, 9.0],
]

y_true = [
    [0, 1],
    [1, 0],
    [0, 1]
    ]

y_true = np.array(y_true)
X = np.array(X)

layer1 = Dense(3, 3)
layer2 = ReLU()
layer3 = Dense(3, 2)
layer4 = SoftmaxCrossEntropy()

print(X)
out = layer1.forward(X)
print(out)
out2 = layer2.forward(out)
print(out2)
out3 = layer3.forward(out2)
print(out3)
out4 = layer4.forward(out3, y_true)
loss = layer4.loss
print(out4)
print(loss)
acc = Accuracy()
print(acc.calculate(y_true, out4))

print("Start of backprop relu and dense")
back4 = layer2.backward(out2)
print(back4)
back3 = layer1.backward(back4)
print(back3)
print(layer1.dbiases)
print(layer1.dweights)

print("START OF TEST")

softmax_outputs = np.array([[0.7, 0.1, 0.2],
[0.1, 0.5, 0.4],
[0.02, 0.9, 0.08]])
class_targets = np.array([[0, 1],
                        [1, 0],
                        [0, 1]])
print(layer4.backward(softmax_outputs, class_targets))

print("Start of backprop full")

back4 = layer4.backward(out4, y_true)
print(back4)
back3 = layer3.backward(back4)
print(back3)
back3 = layer2.backward(back3)
print(back3)
back1 = layer1.backward(back3)
print(back1)