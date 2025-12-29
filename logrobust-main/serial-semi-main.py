from data_loader import AliyunDataLoader
from model import robustlog
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
classes=2
batch_size = 64
learning_rate = 0.001
epoch = 10
device = torch.device('cuda:0')
out_dir = "0916"
output = f"output/{out_dir}_{epoch}e_{learning_rate}lr_{batch_size}bs"
save_dir = os.path.join(output,"model_save")
if not os.path.exists(output):
    os.mkdir(output)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
IS_TRAIN = True

def load_datasets():
    train_db = AliyunDataLoader(mode='train')
    test_db = AliyunDataLoader(mode='test')
    train_loader = DataLoader(train_db,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_db,batch_size=batch_size)
    return train_loader,test_loader


def cal_f1(label_list, pred_list):
    # label_arr = np.array(label_list)
    # pred_arr = np.array(pred_list)
    # 异常检测
    print("异常检测结果:")
    print(classification_report(label_list,pred_list))
    return classification_report(label_list,pred_list)


def main():
    train_loader,test_loader = load_datasets()

    model = robustlog(300,10,2,device=device).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criteon = nn.CrossEntropyLoss().to(device)

    f = open(os.path.join(output,"log.txt"),'w')

    if IS_TRAIN:
        # save model parameter
        torch.save(model.state_dict(),os.path.join(save_dir,f"model_param.pkl"))

        best_f1_score = 0
        for e in range(epoch):
            print("*"*5,e+1,"*"*5)

            #train
            model.train()
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                # print(data.shape)
                logits = model(data)

                loss = criteon(logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print('Train Epoch: [{}/{}]\tLoss: {:.6f}'.format(
                e+1, epoch, total_loss))  
            
            #valid
            model.eval()
            test_loss = 0
            y_true = []
            y_pred = []
            for data, target in test_loader:
                y_true.extend(target)
                data, target = data.to(device), target.to(device)
                logits = model(data)
                test_loss += criteon(logits, target).item()

                pred = logits.data.topk(1)[1].flatten().cpu()
                y_pred.extend(pred)

            # test_loss /= len(test_loader.dataset)

            # F1_Score = f1_score(y_true, y_pred)
            F1_Score = accuracy_score(y_true, y_pred)

            print('\nVALID set: Average loss: {:.4f},F1-score:{}\n'.format(
                test_loss,round(F1_Score.item(),4)))  

            f.write('\nVALID set: Average loss: {:.4f},F1-score:{}\n'.format(
                test_loss,round(F1_Score.item(),4)))
            
            if F1_Score>best_f1_score:
                best_f1_score = F1_Score
                torch.save(model.state_dict(),os.path.join(save_dir,f"model_param.pkl"))

    model.load_state_dict(torch.load(os.path.join(save_dir,f"model_param.pkl")))
    pred_list = []
    label_list = []
    model.eval()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        logits = model(data)
        pred = logits.data.topk(1)[1].flatten()
        pred_list.extend(list(pred.cpu()))
        label_list.extend(list(target.cpu()))

    res = cal_f1(label_list, pred_list)

    f.write(res)
    f.close()

if __name__=='__main__':
    main()