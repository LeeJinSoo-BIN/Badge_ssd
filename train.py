from torch.utils.data import DataLoader
import cv2
from model.ssd import SSD
from model.loss import Loss
import torch
from PIL import Image
from model.datasets import VOCLoader
import torchvision.transforms as transforms
import os
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


transform = transforms.Compose([
                                transforms.Resize((300,300)),
                                transforms.ToTensor()
                                ])

train_set = VOCLoader(
                    root ='./datasets_raid1/voc/VOC2007',
                    image_set='trainval',
                    transform=transform
                    )
#import pdb; pdb.set_trace()    
train_loader = DataLoader(train_set,
                        batch_size = 32,
                        shuffle=True,
                        collate_fn = train_set.collate_fn
                        )

test_set = VOCLoader(
                    root ='./datasets_raid1/voc/VOC2007',
                    image_set='test',
                    transform=transform
                    )
#import pdb; pdb.set_trace()    
test_loader = DataLoader(test_set,
                        batch_size = 32,
                        shuffle=False,
                        collate_fn = train_set.collate_fn
                        )

model = SSD(21).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,momentum=0.9,weight_decay=0.0005)
criterion = Loss().to(device)

#import pdb; pdb.set_trace()  

epochs = 1
for epoch in range(epochs):
    print("%d/%d"%(epoch,epochs))
    model.train()
    for i, (images, categories, boxes) in enumerate(test_loader):
        
        #import pdb; pdb.set_trace()
        images = images.to(device)
        boxes = [box.to(device) for box in boxes]
        categories = [category.to(device) for category in categories]
        
        predicted_loc, predicted_cls = model(images)
        #import pdb; pdb.set_trace()
        loss = criterion(boxes,categories,predicted_loc,predicted_cls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%10==0:
            print("%d/%d"%(i,len(train_loader)))
            print(loss)
