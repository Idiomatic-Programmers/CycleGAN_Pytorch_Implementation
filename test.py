import torch
from models.modules import Generator
import os
import torchvision.transforms as transforms
from PIL import Image

config = {
    'channels': 3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'load_state': 'latest',
    'checkpoint_path': 'checkpoints',
    'test_image': 'datasets/testA/0.jpg'
}


G = Generator(config['channels']).to(config['device'])
G.load_state_dict(torch.load(os.path.join(config['checkpoint_path'], f'G_{config["load_state"]}.pth')))

to_tensor = transforms.Compose([
    transforms.ToTensor()
])

im = Image.open(config['test_image'])
im_tensor = to_tensor(im)
im_tensor = im_tensor.unsqueeze(0)
