import torch
import torchvision
import torchvision.transforms as transforms
import glob
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from vgg_pytorch import VGG
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import cv2


Training_Data_Path='' #modifiy to Covid Data folder

train_transforms = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CovidDataset(Dataset):
    def __init__(self, transforms = None):
        self.transforms=transforms
        self.imgs_path = Training_Data_Path
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        for class_name in ['COVID19','NORMAL']:
            for img_path in glob.glob(self.imgs_path + class_name + "/*.jpg"):
                self.data.append([img_path, class_name])
        self.class_map = {"COVID19" : 0, "NORMAL": 1}
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        class_id = self.class_map[class_name]
        img_tensor = self.transforms(img)
        class_id = torch.tensor(class_id)
        return img_tensor, class_id



trainset = CovidDataset(transforms=train_transforms)
trainloader = DataLoader(trainset, batch_size=8,shuffle=True, num_workers=4)


model = VGG.from_name("vgg16")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

for epoch in range(4): 
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:   
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 9:.3f}')
            running_loss = 0.0

torch.save(model, 'Trained_Covid_Model.pth')
