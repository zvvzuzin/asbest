import os
import cv2
import numpy as np
from tqdm import tqdm
import rawpy
from datetime import datetime
from torch.utils.data import Dataset

from utils import parse_anno_file, create_mask_file, imp_cont_img_file, get_time


class Asbest_segmentation(Dataset):
    def __init__(self, anno, transporter_file=None, crop_size=(672, 672), img_size=(224, 224), num_frames=30, random_crop=True, random_rotate=True, random_contrast=False, normalize=True, load_in_ram=False, random_state = 17):
        
        np.random.seed(random_state)
        self.load_in_ram = load_in_ram
        self.anno = anno
        self.transporter_file = transporter_file
        self.crop_size = crop_size
        self.img_size = img_size
        self.num_frames = num_frames
        self.random_crop = random_crop
        self.random_rotate = random_rotate
        self.random_contrast = random_contrast
        self.normalize = normalize
        
        if self.load_in_ram:
            self.images = []
            self.stone_masks = []
            self.asbest_masks = []
            for it in tqdm(range(num_frames), total = num_frames, leave=False):
                
                image, stone_mask, asbest_mask = self.get_element(it)
                
                self.images.append(image)
                self.stone_masks.append(stone_mask)
                self.asbest_masks.append(asbest_mask)

            self.images = np.array(self.images)
            self.stone_masks = np.array(self.stone_masks)
            self.asbest_masks = np.array(self.asbest_masks)
#         self.images = self.images
#         self.masks = self.masks
        
                    
    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        if self.load_in_ram:
            img = np.expand_dims(self.images[index],axis=0)
            stone_msk = np.expand_dims(self.stone_masks[index],axis=0)
            asbest_msk = np.expand_dims(self.asbest_masks[index],axis=0)
        else:
            image, stone_mask, asbest_mask = self.get_element(index)
            img = np.expand_dims(image,axis=0)
            stone_msk = np.expand_dims(stone_mask,axis=0)
            asbest_msk = np.expand_dims(asbest_mask,axis=0)
             
        return img, stone_msk, asbest_msk
    
    def get_element(self, index):
        stone_mask = create_mask_file(self.anno[index % len(self.anno)], 'stone').astype(float) / 255
        asbest_mask = create_mask_file(self.anno[index % len(self.anno)], 'asbest').astype(float) / 255

        if self.random_contrast:
            image = imp_cont_img_file(self.anno[index % len(self.anno)]['name'], clip_limit=np.random.rand() * 1.3)
        else:
            clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(8,8))
            image = cv2.imread(self.anno[index % len(self.anno)]['name'], cv2.IMREAD_GRAYSCALE)
            image = clahe.apply(image).astype(float) / 255

        if self.transporter_file is not None:
            bg = cv2.imread(self.transporter_file, cv2.IMREAD_UNCHANGED).astype(float) / 255
            bg = cv2.resize(bg, (image.shape[1], image.shape[0]))
            points = np.where(stone_mask == 0)
            image[points] = bg[points]

        if self.crop_size is None:
            min_crop_size = (2 * img_size[0], 2 * img_size[1])
            max_crop_size = min(image.shape[:2])
            k = np.random.rand()
            crop_size = (int(min_crop_size[0] + (max_crop_size - min_crop_size[0]) * k), int(min_crop_size[1] + (max_crop_size - min_crop_size[1]) * k))

        if self.random_crop:
            smth = np.random.rand()
            if smth < 0.0:
                x = np.random.rand()
                if np.random.rand() > 0.5:
                    y = 0
                else: 
                    y = 1
            elif smth < 0.0:
                y = np.random.rand()
                if np.random.rand() > 0.5:
                    x = 0
                else: 
                    x = 1
            else:
                x = np.random.rand()
                y = np.random.rand()

            coor_y = int((image.shape[0] - self.crop_size[0]) * y)
            coor_x = int((image.shape[1] - self.crop_size[1]) * x)
            image = image[coor_y:coor_y + self.crop_size[0], coor_x:coor_x + self.crop_size[1]]
            stone_mask = stone_mask[coor_y:coor_y + self.crop_size[0], coor_x:coor_x + self.crop_size[1]]
            asbest_mask = asbest_mask[coor_y:coor_y + self.crop_size[0], coor_x:coor_x + self.crop_size[1]]

        if self.random_rotate:
            if np.random.rand() > 0.5:
                image = cv2.flip(image, 0)
                stone_mask = cv2.flip(stone_mask, 0)
                asbest_mask = cv2.flip(asbest_mask, 0)
            if np.random.rand() > 0.5:
                image = cv2.flip(image, 1)
                stone_mask = cv2.flip(stone_mask, 1)  
                asbest_mask = cv2.flip(asbest_mask, 1)
                
        if self.crop_size != self.img_size:
            image = cv2.resize(image, self.img_size, interpolation = 1)
            stone_mask = cv2.resize(stone_mask, self.img_size, interpolation = 0)
            asbest_mask = cv2.resize(asbest_mask, self.img_size, interpolation = 0)
            
        if self.normalize:
                image = (image - 0.5) / 0.5

        return image, stone_mask, asbest_mask
    
    
    
true_results = {
    5 : {
        1 : 1.48,
        2 : 2.74, 
        3 : 2.86,
        4 : 2.52,
        5 : 3.16,
        6 : 1.36,
        7 : 2.98,
        8 : 1.79,
        9 : 2.81,
        11 : 2.57,
        12 : 2.80,
        13 : 2.40,
        14 : 2.44,
        18 : 2.29,
        19 : 2.64,
        20 : 3.28,
    },
    16 : {
        2 : 1.89,
        10 : 0.29,
        12 : 3.17,
        17 : 3.09,
        18 : 2.80,
        19 : 2.38,
        21 : 1.01,
        22 : 4.36,
    }
}
    

class Asbest_regression(Dataset):
    def __init__(self, 
                 path, 
                 subset='train', 
                 test_part=0.2, 
                 random_state=17, 
                 crop_size=(672, 672), 
                 img_size=(448, 448), 
                 num_frames=30, 
                 random_crop=True, 
                 random_rotate=True, 
                 random_contrast=True, 
                 normalize=False):

        np.random.seed(random_state)
        dic = {}
        for k in true_results.keys():
            samples = list(true_results[k].keys())
            np.random.shuffle(samples)
            dic[k] = samples

        self.sub_true_results = {}
        if subset == 'train':
            for k in dic.keys():
                samples = dic[k][:int(len(dic[k]) * (1 - test_part))]
                self.sub_true_results[k] = {}
                for sample in samples:
                    self.sub_true_results[k][sample] = true_results[k][sample]
        elif subset == 'valid':
            for k in dic.keys():
                samples = dic[k][int(len(dic[k]) * (1 - test_part)):]
                self.sub_true_results[k] = {}
                for sample in samples:
                    self.sub_true_results[k][sample] = true_results[k][sample]

        files = []
        for pth, fld, fls in os.walk(path):
            for file in fls:
                try:
                    sample = int(file.split('_')[0])
                    date = get_time(file)
                    if date.day in self.sub_true_results.keys():
                        if sample in self.sub_true_results[date.day].keys():
                            files.append(os.path.join(pth, file))
                except:
                    pass
        
        np.random.shuffle(files)
        self.images = []
        self.labels = []
        
        for it in tqdm(range(num_frames), total = num_frames):
            file = files[it % len(files)]
            date = get_time(file.split('/')[-1])
            self.labels.append(true_results[date.day][int(file.split('/')[-1].split('_')[0])] / 5)
            
            if random_contrast:
                if np.random.rand() > 0.5:
                    image = imp_cont_img(file, clip_limit=np.random.rand() * 1.5)#, cv2.IMREAD_UNCHANGED).astype(float) / 255
                else:
                    image = imp_cont_img(file)
            else:
                image = imp_cont_img(file)
            
            if random_crop:
                smth = np.random.rand()
                if smth < 0.4:
                    x = np.random.rand()
                    if np.random.rand() > 0.5:
                        y = 0
                    else: 
                        y = 1
                elif smth < 0.8:
                    y = np.random.rand()
                    if np.random.rand() > 0.5:
                        x = 0
                    else: 
                        x = 1
                else:
                    x = np.random.rand()
                    y = np.random.rand()
                    
                coor_y = int((image.shape[0] - crop_size[0]) * y)
                coor_x = int((image.shape[1] - crop_size[1]) * x)
                image = image[coor_y:coor_y + crop_size[0], coor_x:coor_x + crop_size[1]]
            
            if random_rotate:
                if np.random.rand() > 0.5:
                    image = cv2.flip(image, 0)
                if np.random.rand() > 0.5:
                    image = cv2.flip(image, 1)               
                    
            self.images.append(cv2.resize(image, img_size, interpolation = 1))

        self.images = np.array(self.images)

        if normalize:
            self.images = (self.images - 0.5) / 0.5
        
                    
    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, index):
        img = np.expand_dims(self.images[index],axis=0)
        lbl = np.expand_dims(self.labels[index],axis=0)
             
        return img, lbl