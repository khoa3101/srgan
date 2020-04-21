import glob
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

def compute_cropsize(size, up_factor):
    return size - (size % up_factor)

class TrainData(Dataset):
    def __init__(self, path, size, up_factor):
        super(TrainData, self).__init__()

        self.paths = glob.glob(path + '/*')
        self.size = compute_cropsize(size, up_factor)
        self.up_factor = up_factor

        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(self.size),
            transforms.ToTensor()
        ])
        self.lr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.size // up_factor, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])


    def __len__(self):
        return len(self.paths)


    def __getitem__(self, index):
        img = Image.open(self.paths[index])
        hr = self.hr_transform(img)
        lr = self.lr_transform(hr)

        return lr, hr


class ValData(Dataset):
    def __init__(self, path, up_factor):
        super(ValData, self).__init__()

        self.paths = glob.glob(path + '/*')
        self.up_factor = up_factor

    
    def __len__(self):
        return len(self.paths)


    def __getitem__(self, index):
        img = Image.open(self.paths[index])
        w, h = img.size
        size = compute_cropsize(min(h, w), self.up_factor)

        lr_scale = transforms.Resize(size // self.up_factor, interpolation=Image.BICUBIC)
        hr_scale = transforms.Resize(size, interpolation=Image.BICUBIC)

        hr = transforms.CenterCrop(size)(img)
        lr = lr_scale(hr)
        hr_lr = hr_scale(lr)

        lr = transforms.ToTensor()(lr)
        hr = transforms.ToTensor()(hr)
        hr_lr = transforms.ToTensor()(hr_lr)

        return lr, hr, hr_lr
