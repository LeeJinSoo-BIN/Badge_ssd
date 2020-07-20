import json
import cv2
from tqdm import tqdm

with open('COCO_TEST_det_0.json') as j:
    ssd = json.load(j)

with open('./datasets_raid1/voc/VOC2007/ImageSets/Main/test.txt', "r") as f:
    file_names = [x.strip() for x in f.readlines()]

import pdb; pdb.set_trace()



idx_ssd = 0

for x in tqdm(range(len(file_names))):
    img = cv2.imread('./datasets_raid1/voc/VOC2007/JPEGImages/'+file_names[x]+'.jpg')
    img = cv2.resize(img,(300,300))
    this_image = ssd[idx_ssd]['image_id']
    while(1):
        next_image = ssd[idx_ssd]['image_id']
        if this_image != next_image :
            break
        
        
        x1 = int(ssd[idx_ssd]['bbox'][0])
        y1 = int(ssd[idx_ssd]['bbox'][1])
        x2 = int(x1+ssd[idx_ssd]['bbox'][2])
        y2 = int(x2+ssd[idx_ssd]['bbox'][3])
        
        img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255))
        idx_ssd +=1
    import pdb; pdb.set_trace()
    cv2.imwrite('./box/box%04d.png'%(x),img)

