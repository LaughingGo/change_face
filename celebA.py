import torch
import json
import cv2
import os
from torch.utils.data import Dataset
from PIL import Image

class CelebA(Dataset):
    def __init__(self, ann_file, image_dir, index_list, input_transform, target_transform, att_num=6):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.anns = json.load(open(self.ann_file, 'r'))
        self.index_list = index_list
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.att_num = att_num
                    
    def __getitem__(self, index):
        cp_get = self.index_list[index]
        person_id = cp_get['person_id']
        img_1_id = cp_get['img_1']
        img_2_id = cp_get['img_2']
        person_anns = self.anns[person_id]
        img_1_name = person_anns[img_1_id]["image_id"]
        img_1 = cv2.imread(os.path.join(self.image_dir, img_1_name))
        img_1 = Image.fromarray(img_1)
        img_1 = self.input_transform(img_1)
        img_1_atts =  (torch.FloatTensor(person_anns[img_1_id]["attribute"][:self.att_num])+1)/2
        img_2_name = person_anns[img_2_id]["image_id"]
        img_2 = Image.fromarray(cv2.imread(os.path.join(self.image_dir, img_2_name)))
        img_2 = self.target_transform(img_2)
        img_2_atts =  (torch.FloatTensor(person_anns[img_2_id]["attribute"][:self.att_num])+1)/2
        return img_1, img_2, img_1_atts, img_2_atts
    
    def __len__(self):
        return len(self.index_list) 