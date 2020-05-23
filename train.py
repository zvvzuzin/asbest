import numpy as np
from datetime import datetime
import cv2
import os
from PIL import Image
import torch
import torchvision
from torchvision import datasets, transforms, models
from dataset import Asbest_segmentation
from tqdm import tqdm
import matplotlib.pyplot as plt
import rawpy
from utils import parse_anno_file, create_mask_file, big_image_predict, AverageMeter
from apex import amp

lr = 1e-5

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

path_to_data = 'asbest'
anno_stones = parse_anno_file(os.path.join(path_to_data, 'images', 'annotation.xml'))
anno_tr_stones = parse_anno_file(os.path.join(path_to_data, 'tr_stones', 'annotation.xml'))
transporter_file = os.path.join('asbest', 'transporter', '2020.03.16', 'TRANS_11:28:05_16-03-2020_36.png')
img_tr_stones_shape = (int(anno_tr_stones[0]['height']), int(anno_tr_stones[0]['width']))

stones_valid_indexes = np.array([3, 7, 12, 15, 20, 30, 40], dtype=int)
stones_train_indexes = np.array(list(set(np.arange(len(anno_stones))) - set(stones_valid_indexes)), dtype=int)

from torch import nn
from torch import sigmoid
import segmentation_models_pytorch as smp


device = torch.device("cuda:" + str(torch.cuda.device_count() - 1) if torch.cuda.is_available() else "cpu")

model = smp.Unet(encoder_name='efficientnet-b7', in_channels=1, classes=2, activation='sigmoid').to(device)

bce = smp.utils.losses.BCEWithLogitsLoss()
dice = smp.utils.losses.DiceLoss()
# criterion = nn.CrossEntropyLoss()
# criterion.__name__= 'loss'


def pixel_acc(pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


metrics = [
    smp.utils.metrics.IoU(eps=1.),
    smp.utils.metrics.Fscore(eps=1.),
]


optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': lr}, 

    {'params': model.encoder.parameters(), 'lr': lr},  
])

model, optimizer = amp.initialize(model, 
                                  optimizer, 
                                  opt_level='O2',
#                                   keep_batchnorm_fp32=True,
                                 )

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4], gamma=0.1)

def save_fig(crop_size, inp_size):
    files = ['asbest/tr_stones/9_12:40:22_05-03-2020_1.png',
             'asbest/tr_stones/1_11:32:12_16-03-2020_1.png',
             'asbest/tr_stones/22_13:21:36_16-03-2020_1.png',
             'asbest/tr_stones/20_12:23:59_16-03-2020_1.png',             
            ]

    full_image = None
    for i, file in enumerate(files):
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        image, st_mask, asb_mask = big_image_predict(model, img, crop_size=crop_size, inp_size=inp_size, device=device)
        if full_image is None:
            full_image = np.concatenate((image, st_mask, asb_mask), axis=0)
        else:
            full_image = np.concatenate((full_image, np.concatenate((image, st_mask, asb_mask), axis=0)), axis=1)
    cv2.imwrite('graphics/' + datetime.now().strftime("%H:%M:%S") + '_segm_images.png', cv2.resize((full_image * 255).astype(np.uint8), (int(full_image.shape[1] / 8), int(full_image.shape[0] / 8))))
#     return full_image

from tqdm import trange, tqdm
from torch.utils.data import DataLoader

#model.load_state_dict(torch.load('stone_asbest_segmentation.pth'))

img_sizes = [(1*224, 1*224),
             (2*224, 2*224), 
             (4*224, 4*224),
            ]

# crop_sizes = [
#               None,
#               None,
#               None,
#              ]

crop_sizes = [(8*224, 8*224),
              (8*224, 8*224),
              (8*224, 8*224),
#               (int(img_stones_shape[0] // 2), int(img_stones_shape[1] // 3)), 
            #  (int(img_stones_shape[0] // 2), int(img_stones_shape[1] // 3)),
             # (int(img_stones_shape[0] // 2), int(img_stones_shape[1] // 3)),
             ]
num_frames = [(400, 70),
              (400, 70),
              (400, 70)] #, (400, 50), (400, 50)]
batches = [8, 4, 1]
num_epochs = [100, 100, 1000]

for epochs, batch, crop_size, img_size, num_frame in zip(num_epochs, batches, crop_sizes, img_sizes, num_frames):

    stones_train_data = Asbest_segmentation(np.array(anno_stones)[stones_train_indexes], 
                                            transporter_file=transporter_file, 
                                            crop_size=crop_size, 
                                            img_size=img_size,
                                            load_in_ram = True,
                                            num_frames=num_frame[0], 
                                            normalize=True
                                           )
    stones_valid_data = Asbest_segmentation(np.array(anno_stones)[stones_valid_indexes], 
                                            transporter_file=transporter_file, 
                                            crop_size=crop_size, 
                                            img_size=img_size,
                                            load_in_ram = True,
                                            num_frames=num_frame[1], 
                                            normalize=True
                                           )
    
    stones_train_loader = DataLoader(stones_train_data, batch_size=batch, shuffle=True, num_workers=4)
    stones_valid_loader = DataLoader(stones_valid_data, batch_size=1, shuffle=False, num_workers=2)
    
#     tr_stones_train_data = Asbest_segmentation(anno_tr_stones[:-30], 
#                                                crop_size=(img_tr_stones_shape[0] // 2, img_tr_stones_shape[1] // 2),
#                                                img_size=img_size, 
#                                                num_frames=100, 
#                                                normalize=True)
#     tr_stones_valid_data = Asbest_segmentation(anno_tr_stones[-30:], 
#                                                crop_size=(img_tr_stones_shape[0] // 2, img_tr_stones_shape[1] // 2), 
#                                                img_size=img_size, 
#                                                num_frames=30, 
#                                                normalize=True)

#     tr_stones_train_loader = DataLoader(tr_stones_train_data, batch_size=2, shuffle=True, num_workers=4)
#     tr_stones_valid_loader = DataLoader(tr_stones_valid_data, batch_size=2, shuffle=False, num_workers=2)
    
    with tqdm(total=len(stones_train_loader) + len(stones_valid_loader),# + len(tr_stones_train_loader) + len(tr_stones_valid_loader), 
              bar_format='{desc} epoch {postfix[0]} ' + 
              '| {n_fmt}/{total_fmt} {elapsed}<{remaining} ' + 
              '| loss : {postfix[1]:>2.4f} ' +  
              '| iou_st: {postfix[2]:>2.4f} ' + 
              '| iou_asb: {postfix[3]:>2.4f} ' + 
              '| val_loss : {postfix[4]:>2.4f} ' + 
              '| val_iou_st: {postfix[5]:>2.4f} ' + 
              '| val_iou_asb: {postfix[6]:>2.4f} '
              , 
              postfix=[0, 0, 0, 0, 0, 0, 0], desc = 'Training', leave=True) as t:
        for epoch in range(epochs):
            
            t.postfix[0] = epoch + 1
            
            average_total_loss = AverageMeter()
            average_iou_stones = AverageMeter()
            average_iou_asbest = AverageMeter()

            model.train()
            for data in stones_train_loader:

#                 torch.cuda.empty_cache()
                inputs, st_masks, asb_masks = data
                masks = torch.cat((st_masks, asb_masks), axis=1)

                inputs=inputs.to(device).float()
                masks=masks.to(device).float()

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = 0.9 * bce(outputs, masks) + 0.1 * dice(outputs[:,1:,:,:], masks[:,1:,:,:])

#                 iou_stones = metrics[0](outputs[:,0:1,:,:], masks[:,0:1,:,:])
#                 fscore_stones = metrics[1](outputs[:,0:1,:,:], masks[:,0:1,:,:])
                iou_asbest = metrics[0](outputs[:,1:,:,:], masks[:,1:,:,:])
#                 fscore_asbest = metrics[1](outputs[:,1:,:,:], masks[:,1:,:,:])

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()

                average_total_loss.update(loss.data.item())
#                 average_iou_stones.update(iou_stones.data.item())
                average_iou_asbest.update(iou_asbest.data.item())

                t.postfix[1] = average_total_loss.average()
#                 t.postfix[2] = average_iou_stones.average()
#                 t.postfix[3] = average_fscore_stones.average()
                t.postfix[3] = average_iou_asbest.average()
#                 t.postfix[5] = average_fscore_asbest.average()
                t.update()
            

        
            ## Validation          
            val_average_total_loss = AverageMeter()
            val_average_iou_stones = AverageMeter()
#             val_average_fscore_stones = AverageMeter()
            val_average_iou_asbest = AverageMeter()
#             val_average_fscore_asbest = AverageMeter()

            with torch.no_grad():
                model.eval()

                for data in stones_valid_loader:

    #                 
                    inputs, st_masks, asb_masks = data
                    masks = torch.cat((st_masks, asb_masks), axis=1)

                    inputs=inputs.to(device).float()
                    masks=masks.to(device).float()

                    outputs = model(inputs)

                    loss = 0.9 * bce(outputs, masks) + 0.1 * dice(outputs[:,1:,:,:], masks[:,1:,:,:])

    #                 iou_stones = metrics[0](outputs[:,0:1,:,:], masks[:,0:1,:,:])
    #                 fscore_stones = metrics[1](outputs[:,0:1,:,:], masks[:,0:1,:,:])
                    iou_asbest = metrics[0](outputs[:,1:,:,:], masks[:,1:,:,:])
    #                 fscore_asbest = metrics[1](outputs[:,1:,:,:], masks[:,1:,:,:])

                    val_average_total_loss.update(loss.data.item())
    #                 val_average_iou_stones.update(iou_stones.data.item())
    #                 val_average_fscore_stones.update(fscore_stones.data.item())
                    val_average_iou_asbest.update(iou_asbest.data.item())
    #                 val_average_fscore_asbest.update(fscore_asbest.data.item())

                    t.postfix[4] = val_average_total_loss.average()
    #                 t.postfix[5] = val_average_iou_stones.average()
    #                 t.postfix[8] = val_average_fscore_stones.average()
                    t.postfix[6] = val_average_iou_asbest.average()
    #                 t.postfix[10] = val_average_fscore_asbest.average()
                    t.update()
                

#             scheduler.step()
            if (epoch + 1) % 50 == 0:
                save_fig(crop_size=(img_tr_stones_shape[0] // 2, img_tr_stones_shape[1] // 2), inp_size=img_size)
            t.reset()

torch.save(model.state_dict(), 'asbest_segmentation_b7.pth')