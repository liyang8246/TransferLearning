from torch_directml import torch
from Model import Model
from Dataloader import GetData
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch import optim


device = "privateuseone"
lr     = 0.005
epoch  = 30
bs     = 300
tbs    = 256
model = Model().to(device)
loss_fn = CrossEntropyLoss().to(device)
opt = optim.SGD(model.parameters(),lr=lr)
train_dataset = GetData('data/pre_train',tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
]))
train_loader = DataLoader(train_dataset,batch_size=bs,shuffle=True)
# test_loader = DataLoader(train_dataset,batch_size=tbs,shuffle=True)

model.train()
for current_epoch in range(epoch):
    print('\ncurrent_epoch:' + str(current_epoch+1))
    for idx,items in enumerate(train_loader):
        labels,imgs = items[0].to(device),items[1].to(device)
        print(imgs.shape,imgs)
        opt.zero_grad()
        pre_labels = model(imgs)
        loss = loss_fn(pre_labels,labels)
        loss.backward()
        opt.step()
        print(pre_labels)
        print('\r','-'*idx,'loss: {}'.format(loss.item()),end='')

    # for idx,items in enumerate(test_loader):
    #     labels,imgs = items[0].to(device),items[1].to(device)
    #     pre_labels = model(imgs)
    #     print(pre_labels)
    #     pre_labels = torch.argmax(pre_labels,dim=-1)
    #     true_num = torch.sum(pre_labels == labels)
    #     print('    acc:',true_num.item()/tbs)
    #     break
torch.save(model,'./model.pt')