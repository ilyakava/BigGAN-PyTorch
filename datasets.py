''' Datasets
    This file contains definitions for our CIFAR, ImageFolder, and HDF5 datasets
'''
import csv
import os
import os.path
import sys
from PIL import Image
import numpy as np
from tqdm import tqdm, trange
import random
random.seed(0)

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url, check_integrity
import torch.utils.data as data
from torch.utils.data import DataLoader

import pdb
         
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


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
    
def find_classes_places205(dir):
    class_groups = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes = []
    for g in class_groups:
      group_classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
      classes += group_classes
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
  images = []
  dir = os.path.expanduser(dir)
  for target in tqdm(sorted(os.listdir(dir))):
    d = os.path.join(dir, target)
    if not os.path.isdir(d):
      continue

    for root, _, fnames in sorted(os.walk(d)):
      for fname in sorted(fnames):
        if is_image_file(fname):
          path = os.path.join(root, fname)
          item = (path, class_to_idx[target])
          images.append(item)

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

class ImageFolder(data.Dataset):
  """A generic data loader where the images are arranged in this way: ::

      root/dogball/xxx.png
      root/dogball/xxy.png
      root/dogball/xxz.png

      root/cat/123.png
      root/cat/nsdf3.png
      root/cat/asd932_.png

  Args:
      root (string): Root directory path.
      transform (callable, optional): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
          target and transforms it.
      loader (callable, optional): A function to load an image given its path.

   Attributes:
      classes (list): List of the class names.
      class_to_idx (dict): Dict with items (class_name, class_index).
      imgs (list): List of (image path, class_index) tuples
  """

  def __init__(self, root, transform=None, target_transform=None,
               loader=default_loader, load_in_mem=False, 
               index_filename='imagenet_imgs.npz', **kwargs):
    classes, class_to_idx = find_classes(root)
    # Load pre-computed image directory walk
    if os.path.exists(index_filename):
      print('Loading pre-saved Index file %s...' % index_filename)
      imgs = np.load(index_filename)['imgs']
    # If first time, walk the folder directory and save the 
    # results to a pre-computed file.
    else:
      print('Generating  Index file %s...' % index_filename)
      imgs = make_dataset(root, class_to_idx)
      np.savez_compressed(index_filename, **{'imgs' : imgs})
    if len(imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                           "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    self.root = root
    self.imgs = imgs
    self.classes = classes
    self.class_to_idx = class_to_idx
    self.transform = transform
    self.target_transform = target_transform
    self.loader = loader
    self.load_in_mem = load_in_mem
    
    if self.load_in_mem:
      print('Loading all images into memory...')
      self.data, self.labels = [], []
      for index in tqdm(range(len(self.imgs))):
        path, target = imgs[index][0], imgs[index][1]
        self.data.append(self.transform(self.loader(path)))
        self.labels.append(target)
  
  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is class_index of the target class.
    """
    if self.load_in_mem:
        img = self.data[index]
        target = self.labels[index]
    else:
      path, target = self.imgs[index]
      img = self.loader(str(path))
      if self.transform is not None:
        img = self.transform(img)
    
    if self.target_transform is not None:
      target = self.target_transform(target)
    
    # print(img.size(), target)
    return img, int(target)

  def __len__(self):
    return len(self.imgs)

  def __repr__(self):
    fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    fmt_str += '    Root Location: {}\n'.format(self.root)
    tmp = '    Transforms (if any): '
    fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    tmp = '    Target Transforms (if any): '
    fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    return fmt_str


class ImageFolderTinyImagenet(ImageFolder):
  """
  """
  # sorted list of classes present in http://cs231n.stanford.edu/tiny-imagenet-200.zip
  classes = ['n01443537', 'n01629819', 'n01641577', 'n01644900', 'n01698640', 'n01742172', 'n01768244', 'n01770393', 'n01774384', 'n01774750', 'n01784675', 'n01855672', 'n01882714', 'n01910747', 'n01917289', 'n01944390', 'n01945685', 'n01950731', 'n01983481', 'n01984695', 'n02002724', 'n02056570', 'n02058221', 'n02074367', 'n02085620', 'n02094433', 'n02099601', 'n02099712', 'n02106662', 'n02113799', 'n02123045', 'n02123394', 'n02124075', 'n02125311', 'n02129165', 'n02132136', 'n02165456', 'n02190166', 'n02206856', 'n02226429', 'n02231487', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02281406', 'n02321529', 'n02364673', 'n02395406', 'n02403003', 'n02410509', 'n02415577', 'n02423022', 'n02437312', 'n02480495', 'n02481823', 'n02486410', 'n02504458', 'n02509815', 'n02666196', 'n02669723', 'n02699494', 'n02730930', 'n02769748', 'n02788148', 'n02791270', 'n02793495', 'n02795169', 'n02802426', 'n02808440', 'n02814533', 'n02814860', 'n02815834', 'n02823428', 'n02837789', 'n02841315', 'n02843684', 'n02883205', 'n02892201', 'n02906734', 'n02909870', 'n02917067', 'n02927161', 'n02948072', 'n02950826', 'n02963159', 'n02977058', 'n02988304', 'n02999410', 'n03014705', 'n03026506', 'n03042490', 'n03085013', 'n03089624', 'n03100240', 'n03126707', 'n03160309', 'n03179701', 'n03201208', 'n03250847', 'n03255030', 'n03355925', 'n03388043', 'n03393912', 'n03400231', 'n03404251', 'n03424325', 'n03444034', 'n03447447', 'n03544143', 'n03584254', 'n03599486', 'n03617480', 'n03637318', 'n03649909', 'n03662601', 'n03670208', 'n03706229', 'n03733131', 'n03763968', 'n03770439', 'n03796401', 'n03804744', 'n03814639', 'n03837869', 'n03838899', 'n03854065', 'n03891332', 'n03902125', 'n03930313', 'n03937543', 'n03970156', 'n03976657', 'n03977966', 'n03980874', 'n03983396', 'n03992509', 'n04008634', 'n04023962', 'n04067472', 'n04070727', 'n04074963', 'n04099969', 'n04118538', 'n04133789', 'n04146614', 'n04149813', 'n04179913', 'n04251144', 'n04254777', 'n04259630', 'n04265275', 'n04275548', 'n04285008', 'n04311004', 'n04328186', 'n04356056', 'n04366367', 'n04371430', 'n04376876', 'n04398044', 'n04399382', 'n04417672', 'n04456115', 'n04465501', 'n04486054', 'n04487081', 'n04501370', 'n04507155', 'n04532106', 'n04532670', 'n04540053', 'n04560804', 'n04562935', 'n04596742', 'n04597913', 'n06596364', 'n07579787', 'n07583066', 'n07614500', 'n07615774', 'n07695742', 'n07711569', 'n07715103', 'n07720875', 'n07734744', 'n07747607', 'n07749582', 'n07753592', 'n07768694', 'n07871810', 'n07873807', 'n07875152', 'n07920052', 'n09193705', 'n09246464', 'n09256479', 'n09332890', 'n09428293', 'n12267677']

  def __init__(self, root, transform=None, target_transform=None,
               loader=default_loader, load_in_mem=False, 
               index_filename='tiny_imagenet_imgs.npz', which_half=None, **kwargs):
    """
    args:
      which_half: None = use all, 0 = use even half, 1 = use odd half (by index)
    """
    classes = []
    class_to_idx = {}
    if which_half is not None:
      print('Using %s half of tiny-imagenet classes...' % ('odd' if which_half else 'even'))
    for i, c in enumerate(ImageFolderTinyImagenet.classes):
      if which_half is not None:
        if i % 2 == which_half:
          classes.append(c)
          class_to_idx[c] = (i // 2)
      else:
        classes.append(c)
        class_to_idx[c] = i
      
    
    # Load pre-computed image directory walk
    if os.path.exists(index_filename):
      print('Loading pre-saved Index file %s...' % index_filename)
      imgs = np.load(index_filename)['imgs']
    # If first time, walk the folder directory and save the 
    # results to a pre-computed file.
    else:
      print('Generating  Index file %s...' % index_filename)
      imgs = ImageFolderTinyImagenet.make_dataset(root, class_to_idx)
      np.savez_compressed(index_filename, **{'imgs' : imgs})
      
    msg = "Found %i images in subfolders of: %s \n" % (len(imgs), root)
    if len(imgs) == 0:
      raise(RuntimeError(msg + "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    else:
      print(msg)

    self.root = root
    self.imgs = imgs
    self.classes = classes
    self.class_to_idx = class_to_idx
    self.transform = transform
    self.target_transform = target_transform
    self.loader = loader
    self.load_in_mem = load_in_mem
    
    if self.load_in_mem:
      print('Loading all images into memory...')
      self.data, self.labels = [], []
      for index in tqdm(range(len(self.imgs))):
        path, target = imgs[index][0], imgs[index][1]
        self.data.append(self.transform(self.loader(path)))
        self.labels.append(target)
  
  @staticmethod        
  def make_dataset(dir, class_to_idx, limit=1e9):
    """
    Only get the images within class_to_idx
    """
    images = []
    dir = os.path.expanduser(dir)
    for c, i in class_to_idx.items():
      d = os.path.join(dir, c)
      if not os.path.isdir(d):
        raise ValueError("%s is not a directory" % d)
        
      # walk images in class dir
      for root, _, fnames in sorted(os.walk(d)):
        random.shuffle(sorted(fnames))
        count = 0
        for fname in fnames:
          if is_image_file(fname) and count < limit:
            count += 1
            path = os.path.join(root, fname)
            item = (path, i)
            images.append(item)

    return images


class ImageFolderPlaces205(ImageFolder):
  """
  Places205 comes with a csv with a path to an image and its class idx.
  Directory structure has classes grouped alphabetically also.
  """
  
  def __init__(self, root, transform=None, target_transform=None,
               loader=default_loader, load_in_mem=False, 
               index_filename='train_places205.csv', **kwargs):
    
    self.data_dir = 'data/vision/torralba/deeplearning/images256'
    self.class_to_idx = {}
    # Load pre-computed image directory walk
    index_filepath = os.path.join(root, 'trainvalsplit_places205/train_places205.csv')
    if os.path.exists(index_filepath):
      print('Loading Provided Index file %s...' % index_filepath)
      with open(index_filepath) as csvfile:
        self.imgs = list(csv.reader(csvfile, delimiter=' '))
    
        if len(self.imgs) == 0:
          raise(RuntimeError("EMPTY of %s." % index_filepath))
        
        for path, target in self.imgs:
          target_name = os.path.split(os.path.split(path)[0])[1]
          
          if target_name not in self.class_to_idx:
            self.class_to_idx[target_name] = int(target)
        self.classes = sorted(self.class_to_idx.keys(), key=lambda k: self.class_to_idx[k])
    else:
      raise(RuntimeError("Read FAIL of %s." % index_filepath))

    self.root = root
    self.transform = transform
    self.target_transform = target_transform
    self.loader = loader
    self.load_in_mem = False
    
    
  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is class_index of the target class.
    """
   
    path, target = self.imgs[index]
    full_path = os.path.join(self.root, self.data_dir, path)
    img = self.loader(str(full_path))
    if self.transform is not None:
      img = self.transform(img)
    
    if self.target_transform is not None:
      target = self.target_transform(target)
    
    # print(img.size(), target)
    return img, int(target)


''' ILSVRC_HDF5: A dataset to support I/O from an HDF5 to avoid
    having to load individual images all the time. '''
import h5py as h5
import torch
class ILSVRC_HDF5(data.Dataset):
  def __init__(self, root, transform=None, target_transform=None,
               load_in_mem=False, train=True,download=False, validate_seed=0,
               val_split=0, **kwargs): # last four are dummies
      
    self.root = root
    self.num_imgs = len(h5.File(root, 'r')['labels'])
    
    # self.transform = transform
    self.target_transform = target_transform   
    
    # Set the transform here
    self.transform = transform
    
    # load the entire dataset into memory? 
    self.load_in_mem = load_in_mem
    
    # If loading into memory, do so now
    if self.load_in_mem:
      print('Loading %s into memory...' % root)
      with h5.File(root,'r') as f:
        self.data = f['imgs'][:]
        self.labels = f['labels'][:]

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is class_index of the target class.
    """
    # If loaded the entire dataset in RAM, get image from memory
    if self.load_in_mem:
      img = self.data[index]
      target = self.labels[index]
    
    # Else load it from disk
    else:
      with h5.File(self.root,'r') as f:
        img = f['imgs'][index]
        target = f['labels'][index]
    
   
    # if self.transform is not None:
        # img = self.transform(img)
    # Apply my own transform
    img = ((torch.from_numpy(img).float() / 255) - 0.5) * 2
    
    if self.target_transform is not None:
      target = self.target_transform(target)
    
    return img, int(target)

  def __len__(self):
      return self.num_imgs
      # return len(self.f['imgs'])

import pickle
class CIFAR10(dset.CIFAR10):

  def __init__(self, root, train=True,
           transform=None, target_transform=None,
           download=True, validate_seed=0,
           val_split=0, load_in_mem=True, **kwargs):
    self.root = os.path.expanduser(root)
    self.transform = transform
    self.target_transform = target_transform
    self.train = train  # training set or test set
    self.val_split = val_split

    if download:
      self.download()

    if not self._check_integrity():
      raise RuntimeError('Dataset not found or corrupted.' +
                           ' You can use download=True to download it')

    # now load the picked numpy arrays    
    self.data = []
    self.labels= []
    for fentry in self.train_list:
      f = fentry[0]
      file = os.path.join(self.root, self.base_folder, f)
      fo = open(file, 'rb')
      if sys.version_info[0] == 2:
        entry = pickle.load(fo)
      else:
        entry = pickle.load(fo, encoding='latin1')
      self.data.append(entry['data'])
      if 'labels' in entry:
        self.labels += entry['labels']
      else:
        self.labels += entry['fine_labels']
      fo.close()
        
    self.data = np.concatenate(self.data)
    # Randomly select indices for validation
    if self.val_split > 0:
      label_indices = [[] for _ in range(max(self.labels)+1)]
      for i,l in enumerate(self.labels):
        label_indices[l] += [i]  
      label_indices = np.asarray(label_indices)
      
      # randomly grab 500 elements of each class
      np.random.seed(validate_seed)
      self.val_indices = []           
      for l_i in label_indices:
        self.val_indices += list(l_i[np.random.choice(len(l_i), int(len(self.data) * val_split) // (max(self.labels) + 1) ,replace=False)])
    
    if self.train=='validate':    
      self.data = self.data[self.val_indices]
      self.labels = list(np.asarray(self.labels)[self.val_indices])
      
      self.data = self.data.reshape((int(50e3 * self.val_split), 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    
    elif self.train:
      print(np.shape(self.data))
      if self.val_split > 0:
        self.data = np.delete(self.data,self.val_indices,axis=0)
        self.labels = list(np.delete(np.asarray(self.labels),self.val_indices,axis=0))
          
      self.data = self.data.reshape((int(50e3 * (1.-self.val_split)), 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    else:
      f = self.test_list[0][0]
      file = os.path.join(self.root, self.base_folder, f)
      fo = open(file, 'rb')
      if sys.version_info[0] == 2:
        entry = pickle.load(fo)
      else:
        entry = pickle.load(fo, encoding='latin1')
      self.data = entry['data']
      if 'labels' in entry:
        self.labels = entry['labels']
      else:
        self.labels = entry['fine_labels']
      fo.close()
      self.data = self.data.reshape((10000, 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
      
  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data[index], self.labels[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target
      
  def __len__(self):
      return len(self.data)


class CIFAR100(CIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    
class STL10(dset.STL10):
    """
    In the future could load unlabeled data in some special way too.
    """
    def __init__(self, root, train=True, unlabeled=True,
           transform=None, target_transform=None,
           download=True, validate_seed=0,
           val_split=0, load_in_mem=True, **kwargs):
      if train:
        super(STL10, self).__init__(root, transform=transform, download=download,
                                      target_transform=target_transform)
      elif unlabeled:
        super(STL10, self).__init__(root, split='unlabeled', transform=transform,
                                      target_transform=target_transform)
      else:
        super(STL10, self).__init__(root, split='test', transform=transform,
                                      target_transform=target_transform)