import numpy as np
import matplotlib.pyplot as plt

import csv

csv_file = "train_epoch_lr_0.0003_imgsize_256_patchsize_4.csv"
file = open(csv_file)
csvreader = csv.reader(file)

epochs = []
train_loss = []
train_acc = []
val_loss = []
val_acc = []

for row in csvreader:
    epochs.append(int(row[0].replace('Epoch ', '')))
    train_loss.append(float(row[1].replace(" Train loss ", "")))
    train_acc.append(float(row[2].replace(" Train accuracy ", "")))
    val_loss.append(float(row[3].replace(" Valid loss ", "")))
    val_acc.append(float(row[4].replace(" ","").replace("Validaccuracy", "")))

print(epochs)
print(train_loss)
print(train_acc)
print(val_loss)
print(val_acc)

epochs = [i for i in range(1, 21)]
train_loss = [0.077, 0.039, 0.032, 0.023, 0.029, 0.030, 0.016, 0.016, 0.018, 0.013, 0.010, 0.010, 0.009, 0.008, 0.006, 0.009, 0.005, 0.011, 0.005, 0.005]
train_acc = [0.973, 0.988, 0.990, 0.991, 0.990, 0.991, 0.993, 0.992, 0.993, 0.993, 0.995, 0.996, 0.997, 0.997, 0.998, 0.998, 0.997, 0.998, 0.999, 0.998]
val_loss = [0.269, 1.375, 0.187, 0.272, 0.301, 0.161, 0.184, 0.184, 0.135, 0.156, 0.219, 0.139, 0.211, 0.116, 0.453, 0.273, 0.248, 0.286, 0.218, 0.355]
val_acc = [0.886, 0.884, 0.939, 0.935, 0.930, 0.905, 0.913, 0.926, 0.915, 0.927, 0.905, 0.922, 0.930, 0.939, 0.896, 0.913, 0.918, 0.908, 0.923, 0.917]

print(len(train_loss))
print(len(train_acc))

print(len(val_loss))
print(len(val_acc))

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)

plt.plot(epochs, train_loss, label="Train loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.legend()
plt.title("Loss")
plt.xticks([i for i in range(1, 21)])
plt.yticks([0.1*i for i in range(1, 16)])

plt.subplot(1, 2, 2)

plt.plot(epochs, train_acc, label="Train accuracy")
plt.plot(epochs, val_acc, label="Validation accuracy")
plt.legend()
plt.title("Accuracy")
plt.xticks([i for i in range(1, 21)])
plt.yticks([0.9 + 0.01*i for i in range(0, 11)])

plt.show()


# plt.subplot(2, 1, 2)
# # plt.imshow([])
# plt.title("Loss")