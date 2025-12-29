from data_loader import AliyunDataLoaderForFed,CMCCDataLoaderForFed
from model import robustlog
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import copy
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import pathlib
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--classes', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--global_epochs', type=int, default=10)
parser.add_argument('--local_epochs', type=int, default=10)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--embed_dim', type=int, default=768)
parser.add_argument('--world_size', type=int, default=6)
parser.add_argument('--out_dir', type=str, default='0917')
parser.add_argument('--datatype', type=str, default='aliyun')
parser.add_argument('--is_train', type=bool, default=True)
# parser.add_argument('--is_semi', type=int, default=0)

args = parser.parse_args()

seed = args.seed
classes = args.classes
batch_size = args.batch_size
learning_rate = args.learning_rate
global_epochs = args.global_epochs
local_epochs = args.local_epochs
device = torch.device('cuda:{}'.format(args.gpu_id))
embed_dim = args.embed_dim
world_size = args.world_size
out_path = args.out_dir
datatype = args.datatype
is_train = args.is_train
# is_semi = True if args.is_semi==1 else False
is_semi = False

out_dir = f"output/fedlog_{datatype}_logs_{out_path}_{global_epochs}e_{local_epochs}locEpoch_{learning_rate}lr_{batch_size}bs_{world_size}ws_{is_semi}semis"
save_dir = os.path.join(out_dir, "model_save")

def torch_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def load_datasets_aliyun(rank,world_size):
    train_db = AliyunDataLoaderForFed(mode='train',semi=False,rank=rank,world_size=world_size)
    test_db = AliyunDataLoaderForFed(mode='test',semi=False,rank=rank,world_size=world_size)
    print(rank,len(train_db),len(test_db))
    train_loader = DataLoader(train_db,batch_size=batch_size,shuffle=True,drop_last=True) 
    test_loader = DataLoader(test_db,batch_size=batch_size)
    val_loader = test_loader
    return train_loader,val_loader,test_loader
def load_datasets_cmcc(rank,world_size):
    train_db = CMCCDataLoaderForFed(mode='train',semi=False,rank=rank,world_size=world_size)
    test_db = CMCCDataLoaderForFed(mode='test',semi=False,rank=rank,world_size=world_size)
    train_loader = DataLoader(train_db,batch_size=batch_size,shuffle=True,drop_last=True) 
    test_loader = DataLoader(test_db,batch_size=batch_size)
    val_loader = test_loader
    return train_loader,val_loader,test_loader

# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

    
def cal_f1(label_list, pred_list,fw=None):
    label_arr = np.array(label_list)
    pred_arr = np.array(pred_list)
    # 异常检测
    ad_label = np.where(label_arr>0,1,0)
    ad_pred = np.where(pred_arr>0,1,0)
    print("异常检测结果:")
    print(classification_report(ad_label,ad_pred))
    if fw:
        fw.write("异常检测结果:\n")
        fw.write(classification_report(ad_label,ad_pred))
        fw.write('\n\n')

if __name__=='__main__':
    torch_seed(seed)
    print(out_dir,'\n',save_dir)
    if datatype == 'aliyun':
        loader_list = [load_datasets_aliyun(i,world_size) for i in range(1,world_size)]
    else:
        loader_list = [load_datasets_cmcc(i,world_size) for i in range(1,world_size)]

    client_list = [robustlog(300,10,2,device=device).to(device) for i in range(1,world_size)]

    # optimizer_client_list = [optim.Adam(client_list[i].parameters(), lr=learning_rate) for i in range(world_size-1)]
    # optimizer_server_list = [optim.Adam(server_list[i].parameters(), lr = learning_rate) for i in range(world_size-1)]
    best_score_list = [0 for _ in range(world_size-1)]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    fw_list = [open("./{}/log{}.txt".format(out_dir, i+1),'w') for i in range(0,world_size-1)]
    criteon = nn.CrossEntropyLoss().to(device)

    # save model parameter
    for i,model in enumerate(client_list):
        torch.save(model.state_dict(),os.path.join(save_dir,f"model_param{i}.pkl"))
    
    for it in range(global_epochs):
        print('*'*5,it,'*'*5)
        #train
        for i in range(world_size-1):
            client_model = client_list[i]
            optimizer_client = optim.SGD(client_list[i].parameters(), lr=learning_rate)

            client_model.train()
    
            train_loader = loader_list[i][0]
            for e in range(local_epochs):
                print('server',i ,'train epoch:', e)
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
            
                    target[target!=0] = 1
                    # print(data.dtype,inputs.dtype)
                    # logits = model(data)

                    logits = client_model(data)
                    loss = criteon(logits, target)

                    optimizer_client.zero_grad()

                    loss.backward()

                    optimizer_client.step()

        client_model = FedAvg(list(map(lambda x:x.state_dict(),client_list)))

        for i in range(world_size-1):
            client_list[i].load_state_dict(client_model)
                
        for i in range(world_size-1):
            client_model = client_list[i]

            client_model.eval()

            val_loader = loader_list[i][1]
            #valid
            test_loss = 0
            y_true = []
            y_pred = []
            for data, target in val_loader:
                target[target!=0] = 1
                y_true.extend(target)
                data, target = data.to(device), target.to(device)
         
                logits = client_model(data)
                # logits = model(data)
                test_loss += criteon(logits, target).item()

                pred = logits.data.topk(1)[1].flatten().cpu()
                y_pred.extend(pred)

            test_loss /= len(val_loader.dataset)

            # F1_Score = f1_score(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)

            print('\n{},VALID set: Average loss: {:.4f},score:{}\n'.format(
                i,test_loss,round(acc.item(),4)))
                
            fw_list[i].write('\n{},VALID set: Average loss: {:.4f},score:{}\n'.format(
                it,test_loss,round(acc.item(),4)))
            
            if acc>best_score_list[i]:
                best_score_list[i] = acc
                torch.save(client_model.state_dict(),os.path.join(save_dir,f"model_param{i}.pkl"))

    
    for i in range(world_size-1):
        print("local server",i)
        client_model.load_state_dict(torch.load(os.path.join(save_dir,f"model_param{i}.pkl")))
        client_model.eval()
        test_loader = loader_list[i][2]
        pred_list = []
        label_list = []

        for data, target in test_loader:
            target[target!=0] = 1
            data, target = data.to(device), target.to(device)

            logits = client_model(data)
            pred = logits.data.topk(1)[1].flatten()
            pred_list.extend(list(pred.cpu()))
            label_list.extend(list(target.cpu()))
        cal_f1(label_list, pred_list,fw_list[i])

    for f in fw_list:
        f.close()