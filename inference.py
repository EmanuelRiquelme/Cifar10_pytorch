import torch
from torchvision import transforms
import os
from model import Model
from PIL import Image

transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize((64,64)),
            ])

transform_label = transforms.Compose([
            lambda y: torch.tensor(y)
            ])

labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
cifar_model = Model().to(device)

def prediction(img_name,model,transformation = transform_img,labels = labels,device = device):
    img = Image.open(f'image/{img_name}.jpg')
    img = transformation(img).to(device).unsqueeze(0)
    pred = torch.argmax(model(img),-1)
    return labels[pred]

def load_model(model):
    model.load_state_dict(torch.load(f'{os.getcwd()}/cifar_model.pt'))

if __name__ == '__main__':
    load_model(cifar_model)
    cifar_model.eval()
    print(prediction('bugatti',cifar_model))
