import torch
import torch.optim as optim
from models import *
import os
from network_p import Net1
from dataset import ParamData1
from dataset import ParamData2
from torch.utils.data import DataLoader, Dataset
from a20220919 import SlimmableConv2d



if __name__ == "__main__":

    device = torch.device('cuda:0')
    model_p = SlimmableConv2d([64, 192, 384, 256], [24, 12, 18, 18], (3, 3))

    model_p.to(device)
    train_p = ParamData1('trainset')

    trainloader_p = DataLoader(train_p, batch_size=1, shuffle=False)
    test_p = ParamData2('testset')

    testloader_p = DataLoader(test_p, batch_size=1, shuffle=False)
    criterion_p = nn.MSELoss()
    optimizer_p = optim.SGD(model_p.parameters(), lr=0.2, momentum=0.9)
    for epoch in range(20):
        i = 0
        for i, data in enumerate(trainloader_p, 0):
            def get_alexparams(x):
                model = AlexNet()
                model.load_state_dict(torch.load(os.path.join("train/model{}_alexnet.pth".format(x)), map_location='cuda:0'))
                params = list([])
                for p in model.parameters():
                    params.append(p)
                gth = params[0]
                other = params[5:]
                return gth, other
            inputpara_p = data
            gth_p, otherparam_p = get_alexparams(i+1)
            gth_p = gth_p.to(device)
            outt = torch.zeros(1728, 1).to(device)
            optimizer_p.zero_grad()
            j = 0
            loss_p = 0
            for j in range(len(inputpara_p)-2):
                a = inputpara_p[j].squeeze(0).to(device)
                x = model_p(a)
                outputs_1 = x.reshape(64, 3, 3, 3)
                loss_1 = criterion_p(outputs_1, gth_p)
                loss_p = loss_p + loss_1

            loss_p.backward()
            optimizer_p.step()
        print("epoch:", epoch, loss_p.item())

        with torch.no_grad():
            for i, data in enumerate(testloader_p, 0):
                def get_alexparams(x):
                    model = AlexNet()
                    model.load_state_dict(
                        torch.load(os.path.join("test/model{}_alexnet.pth".format(x)), map_location='cuda:0'))
                    params = list([])
                    for p in model.parameters():
                        params.append(p)
                    gth = params[0]
                    other = params[5:]
                    return gth, other

                inputpara_p = data
                gth_p, otherparam_p = get_alexparams(i + 1)
                gth_p = gth_p.to(device)

                outt = torch.zeros(1728, 1).to(device)
                j = 0
                loss_p = 0
                for j in range(len(inputpara_p)-2):
                    a = inputpara_p[j].squeeze(0).to(device)
                    x = model_p(a).reshape(64, 3, 3, 3)
                    loss_1 = criterion_p(x, gth_p)
                    loss_p = loss_p + loss_1
                print(i+1, loss_p.item())
                # torch.save(x, 'first.pth')

