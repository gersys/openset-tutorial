'''Train CIFAR10 with PyTorch.'''

# Cifar-10 dataset을 closed-set으로 학습을 시키고 SVHN test dataset을 openset으로 Test하는 코드입니다.
# SVHN 데이터셋은 검색해보시면 어떠한 데이터셋인지 나올 겁니다.



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Data set을 loading하는 부분입니다.
# 기존과 차이점은 openset data를 load하는 부분이 추가되었습니다.

# -----------------------------------------------------------------------------------

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

openset = torchvision.datasets.SVHN(
    './data/open/SVHN', split='test',download=True, transform=transform_test)
openloader = torch.utils.data.DataLoader(
    openset, batch_size=100, shuffle=False, num_workers=2
)

# --------------------------------------------------------------------------------

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 학습 Model을 정의하는 부분입니다. Resnet18을 사용하겠습니다.

print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


# 저장된 모델을 load하는 부분입니다.
# ----------------------------------------------------------------------------------
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
# ----------------------------------------------------------------------------------



# loss function 및 optimizaer, learning rate scheduler를 정의하는 부분입니다.
# -------------------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
# --------------------------------------------------------------------------------------




# openset 탐지 능력을 검증하는 코드들 입니다.
# ------------------------------------------------------------------------------
def evaluate_openset(networks, dataloader_on, dataloader_off, **options):

    # closed-set test-data에서 softmax-max값을 추출하여 저장합니다.
    d_scores_on = get_openset_scores(dataloader_on, networks, **options)

    # open-set test-data에서 softmax-max값을 추출하여 저장합니다.
    d_scores_off = get_openset_scores(dataloader_off, networks, **options)


    # closed-set을 클래스 '0' open-set을 클래스 '1'로 지정하여 label을 생성합니다.
    y_true = np.array([0] * len(d_scores_on) + [1] * len(d_scores_off))

    # 각 레이블당 confidence (softmax-max값)을 할당하여 저장합니다.
    y_score = np.concatenate([d_scores_on, d_scores_off])

    # 생성한 label값과 이에 해당하는 confidence값을 이용하여 AUROC값을 추출합니다.
    auc_score = roc_auc_score(y_true, y_score)

    return auc_score


def get_openset_scores(dataloader, networks, dataloader_train=None, **options):

    #위 코드에서 사용되는 함수로 softmax의 max값을 추출하는 함수입니다.
    openset_scores = openset_softmax_confidence(dataloader, networks)
    return openset_scores



def openset_softmax_confidence(dataloader, netC):

    # softmax의 max값을 추출하여 저장하는 부분입니다.

    # 먼저 값을 저장할 list를 선언합니다.
    openset_scores = []

    #dataloader를 통해서 data를 받으면서 softmax값을 추출하고 이의 max값을 저장해줍니다.
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
            preds = F.softmax(netC(images), dim=1)
            openset_scores.extend(preds.max(dim=1)[0].data.cpu().numpy())

    # 마지막에 '-'를 붙여서 return하는 이유는 다음과 같습니다.
    # 위에서 closed-set을 '0' 클래스, open-set을 '1' 클래스로 정의하였습니다.
    # 이때 confidence값이 작으면 '0' 클래스, 크면 '1' 클래스로 지정되도록 현재 AUROC 계산함수는 인식합니다.
    # 그러나 softmax-max output값은 closed-set ('0')이 큰 값을 가지고 open-set ('1')이 작은 값을 가집니다.
    # 때문에 AUROC 함수가 인식하는 결과에 맞게 -를 붙여서 closed-set('0')이 작은 값, open-set ('1')은 큰값이 되도록 합니다.
    # 이 부분은 헷갈리시면 말씀해주세요.

    return -np.array(openset_scores)








# Training 하는 함수입니다.
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# test 하는 함수입니다.
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


if __name__=='__main__':
    #실제 코드 실행하는 부분입니다.
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch) #train 함수 호출
        test(epoch)  #test 함수 호출

        # 앞서 보았던 evaludate_openset함수를 실행하고 output인 auroc값을 출력
        # 이때 입력으로는 network, closed-testloader, open-testloader를 줌.
        print("AUROC : {:.2f} ".format(evaluate_openset(net,testloader,openloader)))
        scheduler.step()
