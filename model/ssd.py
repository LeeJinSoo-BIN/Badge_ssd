import torch
import numpy as np
import torchvision
from math import sqrt
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class SSD(torch.nn.Module):

    def __init__(self, num_classes):
        super(SSD, self).__init__()       
        
        

        #//--VGG-16--------------------------------------
        self.vgg_1_1 = torch.nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        self.vgg_1_2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        

        self.vgg_2_1 = torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        self.vgg_2_2 = torch.nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        

        self.vgg_3_1 = torch.nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        self.vgg_3_2 = torch.nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        self.vgg_3_3 = torch.nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        
        self.vgg_4_1 = torch.nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        self.vgg_4_2 = torch.nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        self.vgg_4_3 = torch.nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        
        self.vgg_5_1 = torch.nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        self.vgg_5_2 = torch.nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        self.vgg_5_3 = torch.nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3,3), padding = 1, stride = 1, bias = True)

        self.vgg_maxpool = torch.nn.MaxPool2d(kernel_size = (2,2), stride = 2, ceil_mode = True)
        # -- VGG-16 Conv5_3 layer------------------------//

        self.vgg_6_1 = torch.nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        

        self.vgg_7_1 = torch.nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = (1,1), padding = 0, stride = 1, bias = True)
        

        # //--Extra Feature Layers----------------------
        self.extra_8_1 = torch.nn.Conv2d(in_channels = 1024, out_channels = 256, kernel_size = (1,1), padding = 0, stride = 1, bias = True)
        self.extra_8_2 = torch.nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = (3,3), padding = 1, stride = 2, bias = True)

        self.extra_9_1 = torch.nn.Conv2d(in_channels = 512, out_channels = 128, kernel_size = (1,1), padding = 0, stride = 1, bias = True)
        self.extra_9_2 = torch.nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3,3), padding = 1, stride = 2, bias = True)

        self.extra_10_1 = torch.nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = (1,1), padding = 0, stride = 1, bias = True)
        self.extra_10_2 = torch.nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3,3), padding = 0, stride = 1, bias = True)
        
        self.extra_11_1 = torch.nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = (1,1), padding = 0, stride = 1, bias = True)
        self.extra_11_2 = torch.nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3,3), padding = 0, stride = 1, bias = True)
        # --------------------------------------------//
            
    
        # //--Prediction Convolutions ---------------

        self.num_classes = num_classes

        num_boxes = {'conv4_3': 4,                    
                    'conv7': 6,
                    'conv8_2': 6,
                    'conv9_2': 6,
                    'conv10_2': 4,
                    'conv11_2': 4,}


        self.predict_location_Conv4_3 = torch.nn.Conv2d(in_channels = 512, out_channels = num_boxes['conv4_3']*4, kernel_size = (3,3), padding = 1, stride = 1, bias = True)        
        self.predict_location_Conv7 = torch.nn.Conv2d(in_channels = 1024, out_channels = num_boxes['conv7']*4, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        self.predict_location_Conv8_2 = torch.nn.Conv2d(in_channels = 512, out_channels = num_boxes['conv8_2']*4, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        self.predict_location_Conv9_2 = torch.nn.Conv2d(in_channels = 256, out_channels = num_boxes['conv9_2']*4, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        self.predict_location_Conv10_2 = torch.nn.Conv2d(in_channels = 256, out_channels = num_boxes['conv10_2']*4, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        self.predict_location_Conv11_2 = torch.nn.Conv2d(in_channels = 256, out_channels = num_boxes['conv11_2']*4, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        
        self.predict_class_Conv4_3 = torch.nn.Conv2d(in_channels = 512, out_channels = num_boxes['conv4_3']*num_classes, kernel_size = (3,3), padding = 1, stride = 1, bias = True)        
        self.predict_class_Conv7 = torch.nn.Conv2d(in_channels = 1024, out_channels = num_boxes['conv7']*num_classes, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        self.predict_class_Conv8_2 = torch.nn.Conv2d(in_channels = 512, out_channels = num_boxes['conv8_2']*num_classes, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        self.predict_class_Conv9_2 = torch.nn.Conv2d(in_channels = 256, out_channels = num_boxes['conv9_2']*num_classes, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        self.predict_class_Conv10_2 = torch.nn.Conv2d(in_channels = 256, out_channels = num_boxes['conv10_2']*num_classes, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        self.predict_class_Conv11_2 = torch.nn.Conv2d(in_channels = 256, out_channels = num_boxes['conv11_2']*num_classes, kernel_size = (3,3), padding = 1, stride = 1, bias = True)
        # ------------------------------------------------//


        self.load_pretrained()

    def forward(self, image):
        #conv2d kernel 3,3 stride 1 padding 1
        #maxpooling kernel 2x2 stride 2
        
        #//--VGG-16--------------------------------------
        feature = torch.nn.functional.relu(self.vgg_1_1(image))
        feature = torch.nn.functional.relu(self.vgg_1_2(feature))        
        feature = self.vgg_maxpool(feature)

        feature = torch.nn.functional.relu(self.vgg_2_1(feature))
        feature = torch.nn.functional.relu(self.vgg_2_2(feature))        
        feature = self.vgg_maxpool(feature)

        feature = torch.nn.functional.relu(self.vgg_3_1(feature))
        feature = torch.nn.functional.relu(self.vgg_3_2(feature))
        feature = torch.nn.functional.relu(self.vgg_3_3(feature))
        feature = self.vgg_maxpool(feature)

        feature = torch.nn.functional.relu(self.vgg_4_1(feature))
        feature = torch.nn.functional.relu(self.vgg_4_2(feature))
        feature = torch.nn.functional.relu(self.vgg_4_3(feature))
        Conv4_3 = feature #38x38x512
        feature = self.vgg_maxpool(feature)

        feature = torch.nn.functional.relu(self.vgg_5_1(feature))
        feature = torch.nn.functional.relu(self.vgg_5_2(feature))
        feature = torch.nn.functional.relu(self.vgg_5_3(feature))
        # -- VGG-16 Conv5_3 layer------------------------//


        feature = torch.nn.functional.relu(self.vgg_6_1(feature))

        feature = torch.nn.functional.relu(self.vgg_7_1(feature))
        Conv7 = feature #19x19x1024

        
        # //--Extra Feature Layers----------------------
        feature = torch.nn.functional.relu(self.extra_8_1(feature))
        feature = torch.nn.functional.relu(self.extra_8_2(feature))
        Conv8_2 = feature #10x10x512

        feature = torch.nn.functional.relu(self.extra_9_1(feature))
        feature = torch.nn.functional.relu(self.extra_9_2(feature))        
        Conv9_2 = feature #5x5x256

        feature = torch.nn.functional.relu(self.extra_10_1(feature))
        feature = torch.nn.functional.relu(self.extra_10_2(feature))
        Conv10_2 = feature #3x3x256

        feature = torch.nn.functional.relu(self.extra_11_1(feature))
        feature = torch.nn.functional.relu(self.extra_11_2(feature))
        Conv11_2 = feature #1x1x256
        # --------------------------------------------//       



        # // -- predict ---------------------------------
        batch_size = Conv4_3.size(0)

        loc_Conv4_3 = self.predict_location_Conv4_3(Conv4_3)
        loc_Conv7 = self.predict_location_Conv7(Conv7)
        loc_Conv8_2 = self.predict_location_Conv8_2(Conv8_2)
        loc_Conv9_2 = self.predict_location_Conv9_2(Conv9_2)
        loc_Conv10_2 = self.predict_location_Conv10_2(Conv10_2)
        loc_Conv11_2 = self.predict_location_Conv11_2(Conv11_2)

        cls_Conv4_3 = self.predict_class_Conv4_3(Conv4_3)
        cls_Conv7 = self.predict_class_Conv7(Conv7)
        cls_Conv8_2 = self.predict_class_Conv8_2(Conv8_2)
        cls_Conv9_2 = self.predict_class_Conv9_2(Conv9_2)
        cls_Conv10_2 = self.predict_class_Conv10_2(Conv10_2)
        cls_Conv11_2 = self.predict_class_Conv11_2(Conv11_2)


        loc_Conv4_3 = loc_Conv4_3.permute(0, 2, 3, 1).contiguous()  
        loc_Conv4_3 = loc_Conv4_3.view(batch_size, -1, 4)        
        loc_Conv7 = loc_Conv7.permute(0, 2, 3, 1).contiguous()  
        loc_Conv7 = loc_Conv7.view(batch_size, -1, 4)
        loc_Conv8_2 = loc_Conv8_2.permute(0, 2, 3, 1).contiguous()  
        loc_Conv8_2 = loc_Conv8_2.view(batch_size, -1, 4)
        loc_Conv9_2 = loc_Conv9_2.permute(0, 2, 3, 1).contiguous()  
        loc_Conv9_2 = loc_Conv9_2.view(batch_size, -1, 4)
        loc_Conv10_2 = loc_Conv10_2.permute(0, 2, 3, 1).contiguous()  
        loc_Conv10_2 = loc_Conv10_2.view(batch_size, -1, 4)
        loc_Conv11_2 = loc_Conv11_2.permute(0, 2, 3, 1).contiguous()  
        loc_Conv11_2 = loc_Conv11_2.view(batch_size, -1, 4)

        cls_Conv4_3 = cls_Conv4_3.permute(0, 2, 3, 1).contiguous()  
        cls_Conv4_3 = cls_Conv4_3.view(batch_size, -1, self.num_classes)
        cls_Conv7 = cls_Conv7.permute(0, 2, 3, 1).contiguous()  
        cls_Conv7 = cls_Conv7.view(batch_size, -1, self.num_classes)
        cls_Conv8_2 = cls_Conv8_2.permute(0, 2, 3, 1).contiguous()  
        cls_Conv8_2 = cls_Conv8_2.view(batch_size, -1, self.num_classes)
        cls_Conv9_2 = cls_Conv9_2.permute(0, 2, 3, 1).contiguous()  
        cls_Conv9_2 = cls_Conv9_2.view(batch_size, -1, self.num_classes)
        cls_Conv10_2 = cls_Conv10_2.permute(0, 2, 3, 1).contiguous()  
        cls_Conv10_2 = cls_Conv10_2.view(batch_size, -1, self.num_classes)
        cls_Conv11_2 = cls_Conv11_2.permute(0, 2, 3, 1).contiguous()  
        cls_Conv11_2 = cls_Conv11_2.view(batch_size, -1, self.num_classes)

        #import pdb; pdb.set_trace()
        predict_loc = torch.cat([loc_Conv4_3, loc_Conv7, loc_Conv8_2, loc_Conv9_2, loc_Conv10_2, loc_Conv11_2], dim=1)  
        predict_cls = torch.cat([cls_Conv4_3, cls_Conv7, cls_Conv8_2, cls_Conv9_2, cls_Conv10_2, cls_Conv11_2],dim=1) 
        # ---------------------------------------------------------//
        return predict_loc, predict_cls

    def load_pretrained(self):
        
        model_state_dict = self.state_dict()
        pretrained_state_dict = torchvision.models.vgg16_bn(pretrained=True).state_dict()

        model_keys = list(model_state_dict.keys())
        pretrained_keys = list(pretrained_state_dict.keys())

        cnt = 0
        idx = 0
        for x in range(len(model_keys)):
            
            if cnt%2 == 0 and cnt != 0 :
               idx = idx + 5 
            if x <26:
                model_state_dict[model_keys[x]] = pretrained_state_dict[pretrained_keys[idx]]
            else :
                if 'weight' in model_keys[x]:
                    torch.nn.init.xavier_uniform_(model_state_dict[model_keys[x]])
                else :
                    torch.nn.init.constant_(model_state_dict[model_keys[x]], 0.)
            #print(model_keys[x])
            
            cnt = cnt+1
            idx = idx+1          
        #import pdb; pdb.set_trace()
        self.load_state_dict(model_state_dict)
        print("loaded pretrained layers and initailized")
    
    
