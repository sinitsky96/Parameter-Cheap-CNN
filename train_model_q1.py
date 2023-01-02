import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.criteria = nn.CrossEntropyLoss()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=32, out_channels=36, kernel_size=3, padding=1),
            nn.BatchNorm2d(36, momentum=0.01),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=36, out_channels=95, kernel_size=2, padding=1),
            nn.BatchNorm2d(95, momentum=0.01),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.PReLU(),
            nn.Linear(1520, 10),
        )

    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x

    def test(self, data):
        """
        test the network
        :param data: test data loader
        :return: accuracy
        """
        correct = 0
        total = 0
        for images, labels in data:
            outputs = self(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = self.criteria(outputs, labels)
        return 100 * correct / total, loss

    def load(self, path):
        """
        load the network
        :param path: path to load the network
        """
        self.load_state_dict(torch.load(path))

    def save(self, path):
        """
            save the network
            :param path: path to save the network
        """
        torch.save(self.state_dict(), path)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model_q1():
    # Hyper Parameters
    num_epochs = 50
    batch_size = 200
    learning_rate = 0.01
    weight_decay = 1e-4
    grad_clip = 0.1

    #  Image Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.2434, 0.2615)),
    ])

    # CIFAR-10 Dataset
    train_dataset = dsets.CIFAR10(root='./data/',
                                  train=True,
                                  transform=transform,
                                  download=True)

    test_dataset = dsets.CIFAR10(root='./data/',
                                 train=False,
                                 transform=transform,
                                 download=True)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    cnn = CNN()
    if torch.cuda.is_available():
        cnn = cnn.cuda()


    # Loss and Optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), learning_rate, weight_decay=weight_decay)

    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, epochs=num_epochs,
                                                steps_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss()
    # print number of parameters
    print('number of parameters: ', sum(param.numel() for param in cnn.parameters()))
    max = 80
    train_error_list = []
    test_error_list = []
    train_loss_list = []
    test_loss_list = []
    for epoch in range(num_epochs):
        # Training phase
        print(f"Epoch number : {epoch + 1}")
        cnn.train()
        lrs = []
        for i, (images, labels) in enumerate(train_loader):
            # Forward + Backward + Optimize
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = cnn(images)
            train_loss = criterion(outputs, labels)
            train_loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(cnn.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        train_loss_list.append(train_loss.data.to('cpu'))


        # Evaluation phase
        cnn.eval()
        with torch.no_grad():

            # train error
            correct = 0
            total = 0
            for images, labels in train_loader:
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = cnn(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('Accuracy of the model on the train images: %d %%' % (100 * correct / total))
            train_error_list.append((100 - (100 * correct / total)).to('cpu'))

            # test error and loss
            correct = 0
            total = 0
            for images, labels in test_loader:
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = cnn(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))
            test_error_list.append((100 - (100 * correct / total)).to('cpu'))
            test_loss_list.append(criterion(outputs, labels).data.to('cpu'))
            if ((100 * correct / total) >= max):
                max = (100 * correct / total)
                cnn.save('model_q1.pkl')

    # plot the error and loss
    plt.figure(1)
    plt.plot(train_error_list, label='train error')
    plt.plot(test_error_list, label='test error')
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.legend()
    plt.figure(2)
    plt.plot(train_loss_list, label='train loss')
    plt.plot(test_loss_list, label='test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    print('min train error: ', min(test_error_list).item())


if __name__ == '__main__':
    train_model_q1()
