import numpy as np
import torch
import torch.utils
from PIL import Image
from torch.autograd import Variable
from model import Finetunemodel
import torchvision.transforms as transforms
import time

def lowlight(img_path):
    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform = transforms.Compose(transform_list)
    im = Image.open(img_path).convert('RGB')
    img_norm = transform(im).numpy()
    img_norm = np.transpose(img_norm, (1, 2, 0))
    low = np.asarray(img_norm, dtype=np.float32)
    low = np.transpose(low[:, :, :], (2, 0, 1))
    image = torch.from_numpy(low)
    image = image.unsqueeze(0)
    input = Variable(image, volatile=True)

    model = Finetunemodel('./weights/medium.pt')
    model.eval()
    start = time.time()
    model(input)
    end_time = (time.time() - start)
    print(end_time)
    return end_time

if __name__ == '__main__':
    with torch.no_grad():
        
        filePath = '/home/linhhima/low_light_enhancement/Zero-DCE++/data/test_data/real/11_0_.png'
        sum_time = 0
        for i in range(0, 110):
            inference_time = lowlight(filePath)
            if i < 10:
                continue
            sum_time = sum_time + inference_time
        print("Perfomance SCI: ", sum_time/100)