import torch 
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.network = models.resnet50(pretrained=True) 

        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
    
data_dir = '/Users/travis/Garbify/data/Garbage classification/Garbage classification'
classes = os.listdir(data_dir)
model = ResNet(num_classes=len(classes))
model.load_state_dict(torch.load('model.pth'))
model.eval()

def predict_image(filepath):

    transform = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor()])
    img = Image.open(filepath)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        yb = model(img)
        _, preds = torch.max(yb, dim=1)
    return classes[preds[0].item()].capitalize()
















