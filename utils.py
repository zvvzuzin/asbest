from datetime import datetime
from lxml import etree
import numpy as np
import cv2
import torch

class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def parse_anno_file(cvat_xml):
    root = etree.parse(cvat_xml).getroot()
    anno = []
    for image_tag in root.iter('image'):
        image = {}
        for key, value in image_tag.items():
            image[key] = value
        image['shapes'] = []
        for poly_tag in image_tag.iter('polygon'):
            polygon = {'type': 'polygon'}
            for key, value in poly_tag.items():
                polygon[key] = value
            image['shapes'].append(polygon)

        image['shapes'].sort(key=lambda x: int(x.get('z_order', 0)))
        anno.append(image)

    return anno

def create_mask_file(annotation, label):
    size = (int(annotation['height']), int(annotation['width']))
#     labels = set([ob['label'] for ob in annotation['shapes']])
    mask = np.zeros(size, dtype=np.uint8)
    for shape in annotation['shapes']:
        if label == shape['label']:
            points = [tuple(map(float, p.split(','))) for p in shape['points'].split(';')]
            points = np.array([(int(p[0]), int(p[1])) for p in points])

            mask = cv2.fillPoly(mask, [points], color=255)
        
    return mask
        
# def create_mask_file(mask_path, width, height, bitness, color_map, background, shapes):
#     mask = np.full((height, width, bitness // 8), background, dtype=np.uint8)
#     for shape in shapes:
#         color = color_map.get(shape['label'], background)
#         points = [tuple(map(float, p.split(','))) for p in shape['points'].split(';')]
#         points = np.array([(int(p[0]), int(p[1])) for p in points])

#         mask = cv2.fillPoly(mask, [points], color=color)
    
#     return mask
#     cv2.imwrite(mask_path, mask)

# def create_empty_mask_file(mask_path, width, height, bitness, color_map, background, shapes):
#     mask = np.full((height, width, bitness // 8), background, dtype=np.uint8)
#     cv2.imwrite(mask_path, mask)
def get_time(file):
    return datetime.strptime(file.split('_')[1] + '_' + file.split('_')[2], '%H:%M:%S_%d-%m-%Y')


def get_clip_limit(file):
    date = get_time(file)
    sample = file.split('_')[0]
    if date.day == 16:
        clip_limit = 1.1
    elif date.day == 5 and int(sample) <= 4:
        clip_limit = 1.1
    elif date.day == 5 and int(sample) >= 12:
        clip_limit = 2.0
    elif date.day == 5 and int(sample) >= 5 and int(sample) <= 11:
        clip_limit = 2.5
    return clip_limit


def imp_cont_img_file(file, clip_limit=None):
    if clip_limit is None:
        clip_limit = get_clip_limit(file.split('/')[-1])
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype(float) / 255
    img -= 0.1
    img = (np.clip(img / img.max(), 0, 1) * 255).astype(np.uint8)
    img = clahe.apply(img).astype(float) / 255
    return img


def preprocess_image(image, clip_limit=None, tile=(8,8)):
    if clip_limit:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
    else:
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=tile)
    if image.dtype == np.uint8:
        image = image.astype(float) / 255
    image -= 0.1
    image = (np.clip(image / image.max(), 0, 1) * 255).astype(np.uint8)
    image = clahe.apply(image).astype(float) / 255
    return image


def big_image_predict(model, image, crop_size, inp_size, normalize=True, device='cpu'):
    
    h, w = image.shape[:2]
    st_mask = np.zeros(image.shape[:2], dtype = float)
    asb_mask = np.zeros(image.shape[:2], dtype = float)
    mean_mask = np.zeros(image.shape[:2], dtype = float)
    num_img_y = int(np.ceil(h / crop_size[0])) * 2 - 1
    num_img_x = int(np.ceil(w / crop_size[1])) * 2 - 1
    image = preprocess_image(image, 1.2)
    
    
    if normalize:
        image = (image - 0.5) / 0.5 
    
    for j in range(num_img_y):
        for i in range(num_img_x):
            part_image = image[int(j / 2  * crop_size[0]):int((j / 2 + 1) * crop_size[0]), 
                               int(i / 2 * crop_size[1]):int((i / 2 + 1) * crop_size[1])].copy()
            mean_mask[int(j / 2  * crop_size[0]):int((j / 2 + 1) * crop_size[0]), 
                      int(i / 2 * crop_size[1]):int((i / 2 + 1) * crop_size[1])] += 1
            init_shape = part_image.shape[:2]
            
            if part_image.shape[0] < crop_size[0]:
                part_image = np.concatenate((part_image, 
                                             np.zeros((crop_size[1] - part_image.shape[0], part_image.shape[1]))), axis=0)
            
            if part_image.shape[1] < crop_size[1]:
                part_image = np.concatenate((part_image, 
                                             np.zeros((part_image.shape[0], crop_size[0] - part_image.shape[1]))), axis=1)
            
            part_image = cv2.resize(part_image, inp_size, interpolation = 1)
            
            part_image = torch.tensor(np.expand_dims(np.expand_dims(part_image, axis=0), axis=0)).to(device).float()
            
            model.eval()
            out_mask = model(part_image).cpu()
            out_mask = np.squeeze(out_mask.cpu().detach().numpy())
            out_st_mask = out_mask[0]
            out_asb_mask = out_mask[1]
            out_st_mask = cv2.resize(out_st_mask, (crop_size[1], crop_size[0]), interpolation=0) [:init_shape[0], :init_shape[1]]
            out_asb_mask = cv2.resize(out_asb_mask, (crop_size[1], crop_size[0]), interpolation=0) [:init_shape[0], :init_shape[1]]
            
            st_mask[int(j / 2  * crop_size[0]):int((j / 2 + 1) * crop_size[0]), 
                    int(i / 2 * crop_size[1]):int((i / 2 + 1) * crop_size[1])] += out_st_mask
            asb_mask[int(j / 2  * crop_size[0]):int((j / 2 + 1) * crop_size[0]), 
                     int(i / 2 * crop_size[1]):int((i / 2 + 1) * crop_size[1])] += out_asb_mask
    
    if normalize:
        image = (image + 1) / 2
    
    return np.clip(image, 0, 1), np.clip(st_mask / mean_mask, 0, 1), np.clip(asb_mask / mean_mask, 0, 1)