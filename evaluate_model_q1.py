import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from train_model_q1 import CNN

# Hyper Parameters
batch_size = 200


# Image Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.2434, 0.2615)),
])

# CIFAR-10 Dataset (test set)
test_dataset = dsets.CIFAR10(root='./data/',
                             train=False,
                             transform=transform,
                             download=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
def evaluate_model_q1():

     # Load the model cuda
    cnn = CNN()
    cnn.load_state_dict(torch.load('model_q1.pkl'))
    if torch.cuda.is_available():
        cnn = cnn.cuda()

    # Test the Model
    cnn.eval()
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

    print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    evaluate_model_q1()