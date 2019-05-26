import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

from network import Network

import time

EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 1e-1
WEIGHT_DECAY = 1e-4

def main():
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    net = Network(3, 128, 10, 10).cuda()

    ACE = nn.CrossEntropyLoss().cuda()
    opt = optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=.9, nesterov=True)

    for epoch in range(1, EPOCHS + 1):
        print('[Epoch %d]' % epoch)
        
        train_loss = 0
        train_correct, train_total = 0, 0

        start_point = time.time()

        for inputs, labels in train_loader:
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            opt.zero_grad()

            preds = net(inputs)
            
            loss = ACE(preds, labels)
            loss.backward()

            opt.step()

            train_loss += loss.item()

            train_correct += (preds.argmax(dim=1) == labels).sum().item()
            train_total += len(preds)

        print('train-acc : %.4f%% train-loss : %.5f' % (100 * train_correct / train_total, train_loss / len(train_loader)))
        print('elapsed time: %ds' % (time.time() - start_point))

        test_loss = 0
        test_correct, test_total = 0, 0

        for inputs, labels in test_loader:
            with torch.no_grad():
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

                preds = net(inputs)

                test_loss += ACE(preds, labels).item()

                test_correct += (preds.argmax(dim=1) == labels).sum().item()
                test_total += len(preds)

        print('test-acc : %.4f%% test-loss : %.5f' % (100 * test_correct / test_total, test_loss / len(test_loader)))
        
        torch.save(net.state_dict(), './checkpoint/checkpoint-%04d.bin' % epoch)

if __name__ == '__main__':
    main()
