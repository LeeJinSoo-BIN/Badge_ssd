import os
import collections
import xml.etree.ElementTree as ET
from PIL import Image

from torch.utils.data import Dataset
import torch
import numpy as np

labels = {'person':1,
         'bird':2, 'cat':3,'cow':4,'dog':5,'horse':6, 'sheep':7,
         'aeroplane':8, 'bicycle':9, 'boat':10, 'bus':11, 'car':12, 'motorbike':13, 'train':14,
         'bottle':15, 'chair':16, 'diningtable':17, 'pottedplant':18, 'sofa':19, 'tvmonitor':20}

class VOCLoader(Dataset):
    

    def __init__(self,
                 root,
                 image_set='train',                 
                 transform=None,
                 ):
        self.root = root
        self.transform = transform 
        
        voc_root = os.path.join(self.root)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.')
        
        splits_dir = os.path.join(voc_root, 'ImageSets/Main')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        #import pdb;pdb.set_trace()
        
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        category = list()
        boxes = list()
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())
        

        #import pdb; pdb.set_trace()
        #import numpy as np
        #import cv2        
        #img1 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        w, h = img.size
        for x in target['annotation']['object']:            
            category.append(labels[x['name']])
            x1 = int(x['bndbox']['xmin']) / w
            y1 = int(x['bndbox']['ymin']) / h
            x2 = int(x['bndbox']['xmax']) / w 
            y2 = int(x['bndbox']['ymax']) / h 
            
            #img1 = cv2.rectangle(img1,(x1*w,y1*h),(x2*w,y2*h),(0,0,255))
            box = [x1,y1,x2,y2]
            boxes.append(box)
        
        
        #cv2.imwrite('box.png',img1)
        if self.transform is not None:
            img = self.transform(img)
        boxes = torch.Tensor(np.array(boxes))
        category = torch.Tensor(np.array(category))

        return img, category, boxes


    def __len__(self):
        return len(self.images)

    

    def collate_fn(self, batchs):        
        
        #import pdb; pdb.set_trace()

        images = list()                
        categories = list()
        boxes = list()
        
        for mini in batchs:
            images.append(mini[0])            
            categories.append(mini[1])
            boxes.append(mini[2])
            

        images = torch.stack(images, dim=0)
   
        return images, categories, boxes

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
