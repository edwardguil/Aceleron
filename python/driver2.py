from Layers import *
from Losses import *
from Metrics import *
from Optimizers import *
from data.data import *

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
# X = [
#     [1.0, 2.0, 3.0],
#     [-4.0, -5.0, -6.0],
#     [7.0, 8.0, 9.0],
# ]

# y_true = [
#     [0, 1],
#     [1, 0],
#     [0, 1]
#     ]

x_train = np.array(x_train_raw_1000)
y_train = np.array(y_train_raw_1000)


x_test = np.array(x_test_raw_1000)
y_test = np.array(y_test_raw_1000)

layer1 = Dense(2, 16)
layer2 = ReLU()
layer3 = Dense(16, 2)
layer4 = SoftmaxCrossEntropy()
sgd = SGD(learning_rate = 1, decay=0.001)

for epoch in range(1001):
    out = layer1.forward(x_train)
    out2 = layer2.forward(out)
    out3 = layer3.forward(out2)
    out4 = layer4.forward(out3, y_train)
    loss = layer4.loss
    acc = Accuracy().calculate(y_train, out4)

    back4 = layer4.backward(out4, y_train)
    back3 = layer3.backward(back4)
    back3 = layer2.backward(back3)
    back1 = layer1.backward(back3)

    sgd.pre_update()
    sgd.update(layer3)
    sgd.update(layer1)
    sgd.post_update()

    if not epoch % 100:
        
        out = layer1.forward(x_test)
        out2 = layer2.forward(out)
        out3 = layer3.forward(out2)
        out4 = layer4.forward(out3, y_test)
        lossTest = layer4.loss
        accTest = Accuracy().calculate(y_test, out4)
        
        print(f'epoch: {epoch}, ' +
        f'acc: {acc:.3f}, ' +
        f'loss: {loss:.3f}, ' +
        f'acc_test: {accTest:.3f}, ' +
        f'loss_test: {lossTest:.3f}, ' +
        f'lr: {sgd.current_learning_rate:.3f}')
git commit -m "Python: Added SGD optimizer, added hardcoded data file, updated layers, added new driver for new data."