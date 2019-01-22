import torch
import json
import cv2
import os
from torch.utils.data import Dataset

class CelebA(Dataset):
    def __init__(self, ann_file, image_dir):
        self.ann_file = ann_file
        self.image_dir = image_dir
        anns = json.load(open(self.ann_file, 'r'))
        person_ids = anns.keys()
        self.index_list = []
        for person_id in person_ids:
            person_anns = anns[person_id]
            person_imgs_num = len(person_anns)
            for i in range(person_imgs_num):
                for j in range(person_imgs_num):
                    if j==i:
                        continue
                    self.index_list.append(person_id+'_'+str(i)+'_'+str(j))
                    
    def __getitem__(self, index):
        cp_get = self.index_list[index]
        split_res = cp_get.split('_')
        person_id = split_res[0]
        img_1_id = split_res[1]
        img_2_id = split_res[2]
        person_anns = anns[person_id]
        img_1_name = person_anns[img_1_id]["image_id"]
        img_1 = torch.FloatTensor(cv2.imread(os.path.join(self.image_dir, img_1_name)))
        img_2_name = person_anns[img_2_id]["image_id"]
        img_2 = torch.FloatTensor(cv2.imread(os.path.join(self.image_dir, img_2_name)))
        img_2_atts =  torch.FloatTensor(person_anns[img_2_id]["attributes"])
        person_id_num = torch.LongTensor([int(person_id)])
        return img_1, img_2, img_2_atts, person_id_num
    
    def __len__(self):
        return len(self.index_list) 