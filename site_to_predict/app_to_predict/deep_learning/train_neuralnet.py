import sys, os
import numpy as np
import pickle
from .dataset.mnist import load_mnist
from .two_layer_net import TwoLayerNet
from django.conf import settings

def neural():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    print("iter_per_epoch",iter_per_epoch)

    save_file = "./network_10000.pkl"

    if os.path.exists(save_file):
        print("Load")
        with open(save_file, "rb") as f:
            network = pickle.load(f)
    else:
        for i in range(iters_num):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]
            
            # grad = network.numerical_gradient(x_batch, t_batch)
            grad = network.gradient(x_batch, t_batch)
            
            for key in ('W1', 'b1', 'W2', 'b2'):
                network.params[key] -= learning_rate * grad[key]
            
            loss = network.loss(x_batch, t_batch)
            train_loss_list.append(loss)
            
            if i % iter_per_epoch == 0:
                train_acc = network.accuracy(x_train, t_train)
                test_acc = network.accuracy(x_test, t_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

        print("Creating pickle file ...")
        with open(save_file, 'wb') as f:
            pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)
        print("Done!")

    from PIL import Image
    img = np.array(Image.open(settings.IMAGE_URL).convert("L"))
    img = img.reshape(1, 784)
    img = img.astype('float32')
    img = img / 255

    y = network.predict(img)
    print(y)
    p= np.argmax(y)
    print(y[0][p] * 100,"%")
    print("[",p,"]")
    return p
