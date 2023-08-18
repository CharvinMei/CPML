import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model_Resnet50 import CPML as model
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize(512),
            transforms.RandomResizedCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(512),
               transforms.CenterCrop(448),
               transforms.ToTensor(),
               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


    train_dataset = datasets.ImageFolder(root="./dataset/CUB_200_2011/dataset/train",
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    bird_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in bird_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices_CUB200.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root="./dataset/CUB_200_2011/dataset/test",
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images fot validation.".format(train_num,
                                                                           val_num))

    net = model(num_classes=200)
    net = net.to(device)

    loss_function = nn.CrossEntropyLoss().to(device)
    loss_kl = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.SGD(net.parameters(), lr=0.002, weight_decay=0.00005, momentum=0.9)

    epochs = 128
    best_acc = 0.0
    save_path = './result/CUB200/best_model_resnet50_CPML.pth'
    train_steps = len(train_loader)
    val_accuracy_list = []
    train_accuracy_list = []
    epochs_list = []
    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        # train
        net.train()
        if (epoch == 20):   #20,0.001
            optimizer = optim.SGD(net.parameters(), lr=0.001, weight_decay=0.00005, momentum=0.9)
        elif (epoch == 30): #30,0.0005
            optimizer = optim.SGD(net.parameters(), lr=0.0005, weight_decay=0.00005, momentum=0.9)
        elif (epoch == 50): #50,0.0001
            optimizer = optim.SGD(net.parameters(), lr=0.0001, weight_decay=0.00005, momentum=0.9)
        elif (epoch == 90): #50,0.0001
            optimizer = optim.SGD(net.parameters(), lr=0.00005, weight_decay=0.00005, momentum=0.9)

        train_bar = tqdm(train_loader)
        train_acc = 0.0
        train_loss = 0.0
        train_steps = 0
        for step, data in enumerate(train_bar):
            train_steps += 1
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            x, logits_r = net(images, flag="train")
            loss = loss_function(x, labels) + loss_function(logits_r, labels) + loss_kl(F.log_softmax(x, dim=1),
                                F.softmax(logits_r, dim=1)) + loss_kl(F.log_softmax(logits_r, dim=1), F.softmax(x, dim=1))
            train_predict = torch.max(x.data, dim=1)[1]
            train_acc += torch.eq(train_predict, labels.to(device)).sum().item()
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # validate
        net.eval()
        val_acc = 0.0
        # train_acc = 0.0
        val_loss = 0.0
        # train_loss = 0.0
        with torch.no_grad():

            val_bar = tqdm(validate_loader)
            val_steps = 0
            for val_data in val_bar:
                val_steps += 1
                val_images, val_labels = val_data
                val_outputs, _ = net(val_images.to(device), flag="val")
                tmp_val_loss = loss_function(val_outputs, val_labels.to(device))
                val_predict = torch.max(val_outputs, dim=1)[1]
                val_acc += torch.eq(val_predict, val_labels.to(device)).sum().item()
                val_loss += tmp_val_loss.item()
                val_bar.desc = "valid in val_dataset epoch[{}/{}]".format(epoch + 1, epochs)

        train_accurate = train_acc / train_num
        val_accurate = val_acc / val_num

        if (val_accurate > best_acc):
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f train_acc: %.3f val_loss:%.3f val_acc: %.3f'
              % (epoch + 1, train_loss / train_steps, train_accurate, val_loss / val_steps, val_accurate))

        # 构造各个参数的列表，准备画图
        val_accuracy_list.append(val_accurate)
        train_accuracy_list.append(train_accurate)
        train_loss_list.append(train_loss / train_num)
        val_loss_list.append(val_loss / val_num)
        epochs_list.append(epoch + 1)


        # train_acc && val_loss
        plt.figure()
        plt.plot(epochs_list, val_accuracy_list, color="red", label="val_acc")
        plt.plot(epochs_list, train_accuracy_list, color="green", label="train_acc")
        plt.xlabel("epochs")
        plt.ylabel("Acc")
        plt.title('ResNet50 in CUB200')
        plt.xticks([i for i in range(0, len(epochs_list), 20)])
        acc_gap = [i * 0.2 for i in range(0, min(int(len(epochs_list) / 2 + 1), 6))]
        acc_gap.append(max(val_accuracy_list))
        acc_gap.append(max(train_accuracy_list))
        plt.yticks(acc_gap)
        plt.grid()
        plt.legend()
        plt.savefig("./result/CUB200/Acc_resnet50_CPML.jpg")

        # train_loss && val_loss
        plt.figure()
        plt.plot(epochs_list, train_loss_list, color="red", label="train_loss")
        plt.plot(epochs_list, val_loss_list, color="green", label="val_loss")
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.title('ResNet50 in CUB200')
        plt.xticks([i for i in range(0, len(epochs_list), 20)])
        plt.grid()
        plt.legend()
        plt.savefig("./result/CUB200/Loss_resnet50_CPML.jpg")

    print('Finished Training')
    print("the best val_accuracy is : {}".format(best_acc))


if __name__ == '__main__':
    main()
