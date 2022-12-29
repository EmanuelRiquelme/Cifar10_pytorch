import torch
from torchvision import datasets,transforms
import os
from torch.utils.data import DataLoader
from model import Model
import torch.optim as optim
import torch.nn as nn
from tqdm import trange

transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((64,64))
            ])

transform_label = transforms.Compose([
            lambda y: torch.tensor(y)
            ])

train_data = datasets.CIFAR10(root = os.getcwd(),train = True,
                                transform = transform_img,target_transform = transform_label)
batch_size = 64

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

def load_model(model):
    model.load_state_dict(torch.load(f'{os.getcwd()}/cifar_model.pt'))

def save_model(model):
    torch.save(model.state_dict(), f'{os.getcwd()}/cifar_model.pt')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

cifar_model = Model().to(device)
#load_model(cifar_model) 
opt = optim.Adam(cifar_model.parameters(),lr=0.005)
loss_fn = nn.CrossEntropyLoss()

epochs = 200

def train_pipeline(train_data = train_dataloader,model = cifar_model,
                    loss_fn = loss_fn,opt = opt,epochs = epochs,device = device):
    for epoch in  (t := trange(epochs)):
        it = iter(train_data)
        loss_temp = []
        for _ in range(len(train_data)):
            input,target = next(it)
            input,target = input.to(device),target.to(device)
            opt.zero_grad()
            output = model(input)
            loss = loss_fn(output,target)
            loss_temp.append(loss.item())
            loss.backward()
            opt.step()

        save_model(model)
        t.set_description("loss: %.2f" % (sum(loss_temp)/len(train_data)))

if __name__ == '__main__':
    train_pipeline()
    save_model(cifar_model)
