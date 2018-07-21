import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch
import torchvision.models as models
model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}
import cPickle as pkl
from PIL import Image
import os
import os.path
import numpy as np
import pdb
# from myutils import *

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(imdb):
    #TODO: classes: list of classes
    #TODO: class_to_idx: dictionary with keys=classes and values=class index

    classes = list(imdb._classes)
    class_to_idx = imdb._class_to_ind

    return classes, class_to_idx


def make_dataset(imdb, class_to_idx):
    #TODO: return list of (image path, list(+ve class indices)) tuples
    #You will be using this in IMDBDataset
    image_index = imdb._load_image_set_index()
    
    images=[]
    for idx in image_index:
        img_path = imdb.image_path_from_index(idx)
        annotation = imdb._load_pascal_annotation(idx)
        images.append((img_path, annotation['gt_classes']))
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 20, kernel_size=1),
            )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x




class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetRobust, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 20, kernel_size=1),
            nn.Dropout2d(0.3, inplace=True),
            )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)*0.7
        return x


def localizer_alexnet(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(**kwargs)
    # Pre-trained on imageNet
    if pretrained:
        print("Pre-trained True")
        if os.path.exists('pretrained_alexnet.pkl'):
            pret_net = pkl.load(open('pretrained_alexnet.pkl','r'))
        else:
            pret_net = model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
            pkl.dump(pret_net, open('pretrained_alexnet.pkl','wb'), pkl.HIGHEST_PROTOCOL)
        # model.features = models.alexnet(pretrained=True).features
        own_state = model.state_dict()
        for name, param in pret_net.items():
            if name not in own_state or 'features' not in name:
                continue
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
                print('Copied {}'.format(name))
            except:
                print('Did not find {}'.format(name))
                continue
    # Xavier initialization
    else:
        print("Pre-trained False")
        for i in range(len(model.features)):
            if len(model.features[i].state_dict()):
                nn.init.xavier_normal(model.features[i].weight.data)
    
    # In any case, last 3 conv layers are xavier initialized
    for i in range(len(model.classifier)):
        if len(model.classifier[i].state_dict()):
            nn.init.xavier_normal(model.classifier[i].weight.data)
    return model

def localizer_alexnet_robust(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(**kwargs)
    if pretrained:
        print("Pre-trained True")
        if os.path.exists('pretrained_alexnet.pkl'):
            pret_net = pkl.load(open('pretrained_alexnet.pkl','r'))
        else:
            pret_net = model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
            pkl.dump(pret_net, open('pretrained_alexnet.pkl','wb'), pkl.HIGHEST_PROTOCOL)
        # model.features = models.alexnet(pretrained=True).features
        own_state = model.state_dict()
        for name, param in pret_net.items():
            if name not in own_state or 'features' not in name:
                continue
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
                print('Copied {}'.format(name))
            except:
                print('Did not find {}'.format(name))
                continue
        # model.features = models.alexnet(pretrained=True).features
    # Xavier initialization
    else:
        print("Pre-trained False")
        for i in range(len(model.features)):
            if len(model.features[i].state_dict()):
                nn.init.xavier_normal(model.features[i].weight.data)
    
    # In any case, last 3 conv layers are xavier initialized
    for i in range(len(model.classifier)):
        if len(model.classifier[i].state_dict()):
            nn.init.xavier_normal(model.classifier[i].weight.data)
    return model

class IMDBDataset(data.Dataset):
    """A dataloader that reads imagesfrom imdbs
    Args:
        imdb (object): IMDB from fast-rcnn repository
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, list(+ve class indices)) tuples
    """

    def __init__(self, imdb, transform=None, target_transform=None,
                 loader=default_loader):
        all_class_names, class_to_idx = find_classes(imdb)
        imgs_and_class = make_dataset(imdb, class_to_idx)
        # pdb.set_trace()
        if len(imgs_and_class) == 0:
            raise(RuntimeError("Found 0 images, what's going on?"))
        self.imdb = imdb
        self.imgs_and_class = imgs_and_class

        self.all_class_names = all_class_names # All class names
        self.class_to_idx = class_to_idx # All class names to indexes
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a binary vector with 1s
                                   for +ve classes and 0s for -ve classes
                                   (it can be a numpy array)
        """

        img_path, classes = self.imgs_and_class[index]
        img = default_loader(img_path)
        target = torch.from_numpy(np.zeros(len(self.all_class_names)))
        
        for i in range(target.shape[0]):
            if i in classes:
                target[i]=1
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs_and_class)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
