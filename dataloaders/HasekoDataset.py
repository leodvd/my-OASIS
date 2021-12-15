import random
import torch
import torchvision
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np
import cv2
import json
import copy


class HasekoDataset(torch.utils.data.Dataset):  # with some facilities in alpha channel (RGBA)
    def __init__(self, opt, for_metrics):
        # Defines the way each color is mapped to which channel. Here the different bedroom types are all mapped to the same one, and so they will have the same color in the final output
        # self.channels_inv = {'Background': 0,
        #                      'Entrance': 1,
        #                      'Window': 2,
        #                      'Dead space': 3,
        #                      'Corridor': 4,
        #                      'Living room': 5,
        #                      'Kitchen': 6,
        #                      'Sanitary': 7,
        #                      'Storage': 8,
        #                      'pillars': 9,
        #                      'Bed Room 1': 10,
        #                      'Bed Room 2': 10,
        #                      'Bed Room 3': 10,
        #                      'Bed Room 4': 10,
        #                      # 'Bed Room 2': 11,
        #                      # 'Bed Room 3': 12,
        #                      # 'Bed Room 4': 13,
        #                      'Fridge Holder': 11,
        #                      'Kitchen Counter': 12,
        #                      'Bath Tub': 13,
        #                      'Washing Basin': 14,
        #                      'Washing Machine Holder': 15,
        #                      'Toilet Bowl': 16
        #                      }
        self.channels_inv = {'Dead space': 0,
                             'Corridor': 1,
                             'Living room': 2,
                             'Kitchen': 3,
                             'Sanitary': 4,
                             'Toilets': 5,
                             'Bathroom': 6,
                             'Storage': 7,
                             'Bed Room 1': 8,
                             'Bed Room 2': 8,
                             'Bed Room 3': 8,
                             'Bed Room 4': 8,
                             'Bath Tub': 9,
                             'Washing Basin': 10,
                             'Washing Machine Holder': 11,
                             # 'Fridge Holder': 10,
                             # 'Kitchen Counter': 11,
                             # 'Toilet Bowl': 12
                             }

        # Find out how many layers we have
        n_layer = 0
        for v in self.channels_inv.values():
            if v > n_layer:
                n_layer = v
        n_layer += 1

        # if opt.phase == "test" or for_metrics:
        #     opt.load_size = 128
        # else:
        #     opt.load_size = 128
        # opt.crop_size = 128
        # opt.load_size = 256
        # opt.crop_size = 256
        opt.load_size = opt.im_size
        opt.crop_size = opt.im_size
        opt.label_nc = 5
        opt.contain_dontcare_label = True
        opt.semantic_nc = 6  # label_nc + unknown. Amount of layers in the input label map (black, pillars, entrance, window, white background)
        opt.semantic_nc_image = n_layer + 1  # n_layers + unknown. Amount of layers in the target label map
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels, self.paths = self.list_images()
        self.angles = [i for i in range(-90, 270, 90)]

        self.colors = json.load(open("colors.json", 'r'))

    def __len__(self, ):
        return len(self.images)

    def __getitem__(self, idx):
        image_rgb = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        image_rgba = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGBA')
        label = Image.open(os.path.join(self.paths[1], self.labels[idx])).convert('L')

        # image, label = self.transforms(image_rgb, label)
        image_rgb, label, image_rgba = self.transforms_2im(image_rgb, label, image_rgba)
        image_rgb = image_rgb * 255
        label = label * 255
        image_rgba = image_rgba * 255

        image_rgb = image_rgb.to('cpu').detach().numpy().copy().transpose((1, 2, 0))
        image_rgba = image_rgba.to('cpu').detach().numpy().copy().transpose((1, 2, 0))
        image_alpha_chan = image_rgba[:, :, 3:]
        # # somehow the following 3 lines don't work properly (which is why i import the images twice as above)
        # image= image.to('cpu').detach().numpy().copy().transpose((1, 2, 0))
        # image_rgb = image[:, :, :3]  # [:3] to get just the 3 RGB channels
        # image_alpha_chan = image[:, :, 3:]

        # mask = (127 == image[:, :, 3])
        # print('mask: ', image_rgb[mask])

        _, h, w = label.size()

        # image2 = torch.full(fill_value=255, size=(self.opt.semantic_nc_image, h, w))  # todo: replace the following 4 lines by this one for ablation test of the footprint-shaped background
        image2 = torch.empty(self.opt.semantic_nc_image, h, w)
        for i in range(self.opt.semantic_nc_image):
            image2[i] = copy.deepcopy(label[0])
        # image2[image2 < 254] = 127   # 191   (the background is a white 255 already, and it puts all where there is content - the footprint - at 128)
        # todo: change the above line to keep just the footprint as a background (value == 0) with the following:
        image2[image2 > 1] = 255   # background, pillars, window, entrance
        image2[image2 <= 1] = 191  # footprint

        room_seg_value = 0

        # the channel 0 is already full of 255 values and the 128 values of the footprint because of the label copy and threshold

        # for coord in torch.tensor(np.where((image_rgb == (255, 0, 0)).all(axis=-1))).numpy().transpose():  # Entrance (red)
        #     image2[1, coord[0], coord[1]] = room_seg_value
        #
        # for coord in torch.tensor(np.where((image_rgb == (255, 0, 102)).all(axis=-1))).numpy().transpose():  # Window (dark pink)
        #     image2[2, coord[0], coord[1]] = room_seg_value
        # for coord in torch.tensor(np.where((image_rgb == (255, 0, 100)).all(axis=-1))).numpy().transpose():  # Window (dark pink) (BIM)
        #     image2[2, coord[0], coord[1]] = room_seg_value

        for coord in torch.tensor(np.where((image_rgb == (255, 255, 0)).all(axis=-1))).numpy().transpose():  # Dead Space (yellow)
            image2[0, coord[0], coord[1]] = room_seg_value

        for coord in torch.tensor(np.where((image_rgb == (0, 255, 255)).all(axis=-1))).numpy().transpose():  # Corridor (bright blue)
            image2[1, coord[0], coord[1]] = room_seg_value

        for coord in torch.tensor(np.where((image_rgb == (128, 128, 128)).all(axis=-1))).numpy().transpose():  # Living room (medium grey)
            image2[2, coord[0], coord[1]] = room_seg_value

        for coord in torch.tensor(np.where((image_rgb == (20, 74, 25)).all(axis=-1))).numpy().transpose():  # Kitchen (dark forest green)
            image2[3, coord[0], coord[1]] = room_seg_value
        for coord in torch.tensor(np.where((image_rgb == (0, 75, 36)).all(axis=-1))).numpy().transpose():  # Kitchen (dark forest green) (BIM)
            image2[3, coord[0], coord[1]] = room_seg_value

        for coord in torch.tensor(np.where((image_rgb == (51, 255, 0)).all(axis=-1))).numpy().transpose():  # Sanitary (Sanitary) (bright green)
            image2[4, coord[0], coord[1]] = room_seg_value
        # for coord in torch.tensor(np.where((image_rgb == (136, 0, 21)).all(axis=-1))).numpy().transpose():  # Sanitary (Toilettes) (burgundy)
        #     image2[4, coord[0], coord[1]] = room_seg_value
        # for coord in torch.tensor(np.where((image_rgb == (34, 177, 76)).all(axis=-1))).numpy().transpose():  # Sanitary (Bathroom) (chlorophile medium green)
        #     image2[4, coord[0], coord[1]] = room_seg_value
        for coord in torch.tensor(np.where((image_rgb == (136, 0, 21)).all(axis=-1))).numpy().transpose():  # Sanitary (Toilets) (burgundy)
            image2[5, coord[0], coord[1]] = room_seg_value
        for coord in torch.tensor(np.where((image_rgb == (34, 177, 76)).all(axis=-1))).numpy().transpose():  # Sanitary (Bathroom) (chlorophile medium green)
            image2[6, coord[0], coord[1]] = room_seg_value

        # for coord in torch.tensor(np.where((image_rgb == (204, 0, 255)).all(axis=-1))).numpy().transpose():  # Storage  (bright pink)
        #     image2[5, coord[0], coord[1]] = room_seg_value
        # for coord in torch.tensor(np.where((image_rgb == (213, 45, 247)).all(axis=-1))).numpy().transpose():  # Storage  (bright pink) (BIM)
        #     image2[5, coord[0], coord[1]] = room_seg_value
        for coord in torch.tensor(np.where((image_rgb == (204, 0, 255)).all(axis=-1))).numpy().transpose():  # Storage  (bright pink)
            image2[7, coord[0], coord[1]] = room_seg_value
        for coord in torch.tensor(np.where((image_rgb == (213, 45, 247)).all(axis=-1))).numpy().transpose():  # Storage  (bright pink) (BIM)
            image2[7, coord[0], coord[1]] = room_seg_value

        # for coord in torch.tensor(np.where((image_rgb == (255, 153, 0)).all(axis=-1))).numpy().transpose():  # pillars (orange)
        #     image2[9, coord[0], coord[1]] = room_seg_value

        for coord in torch.tensor(np.where((image_rgb == (0, 0, 255)).all(axis=-1))).numpy().transpose():  # Bed Room 1
            image2[8, coord[0], coord[1]] = room_seg_value
        for coord in torch.tensor(np.where((image_rgb == (0, 12, 102)).all(axis=-1))).numpy().transpose():  # Bed Room 2
            image2[8, coord[0], coord[1]] = room_seg_value
        for coord in torch.tensor(np.where((image_rgb == (24, 154, 180)).all(axis=-1))).numpy().transpose():  # Bed Room 3
            image2[8, coord[0], coord[1]] = room_seg_value
        for coord in torch.tensor(np.where((image_rgb == (177, 212, 224)).all(axis=-1))).numpy().transpose():  # Bed Room 4
            image2[8, coord[0], coord[1]] = room_seg_value
        # for coord in torch.tensor(np.where((image == (0, 12, 102)).all(axis=-1))).numpy().transpose():  # Bed Room 2
        #     image2[11, coord[0], coord[1]] = room_seg_value
        # for coord in torch.tensor(np.where((image == (24, 154, 180)).all(axis=-1))).numpy().transpose():  # Bed Room 3
        #     image2[12, coord[0], coord[1]] = room_seg_value
        # for coord in torch.tensor(np.where((image == (177, 212, 224)).all(axis=-1))).numpy().transpose():  # Bed Room 4
        #     image2[13, coord[0], coord[1]] = room_seg_value

        # facilities:
        for coord in torch.tensor(np.where((image_alpha_chan == 127).all(axis=-1))).numpy().transpose():  # Bath Tub
            image2[9, coord[0], coord[1]] = room_seg_value
        for coord in torch.tensor(np.where((image_alpha_chan == 128).all(axis=-1))).numpy().transpose():  # Washing Basin
            image2[10, coord[0], coord[1]] = room_seg_value
        for coord in torch.tensor(np.where((image_alpha_chan == 129).all(axis=-1))).numpy().transpose():  # Washing Machine Holder
            image2[11, coord[0], coord[1]] = room_seg_value

        # for coord in torch.tensor(np.where((image_alpha_chan == 125).all(axis=-1))).numpy().transpose():  # Fridge Holder
        #     image2[10, coord[0], coord[1]] = room_seg_value
        # for coord in torch.tensor(np.where((image_alpha_chan == 126).all(axis=-1))).numpy().transpose():  # Kitchen Counter
        #     image2[11, coord[0], coord[1]] = room_seg_value
        # for coord in torch.tensor(np.where((image_alpha_chan == 130).all(axis=-1))).numpy().transpose():  # Toilet Bowl
        #     image2[12, coord[0], coord[1]] = room_seg_value

        # for i in range(11):
        #     print(f'image2[{i}] (min/max/mean): ', image2[i].min(), image2[i].max(), image2[i].mean())

        # Converting the input image to a label map
        label[label == 255] = 1  # White, Background
        label[label == 0] = 2  # Black, footprint
        label[label == 166] = 3  # Orange, pillars
        label[label == 88] = 4  # Pink, Window
        label[label == 76] = 5  # Red, Entrance

        # print('image size: ', image2.size())
        return {"image": image2/510, "label": label, "name": self.images[idx]}  # range: 0.5 (white background) / 0.251 (apartment shape) / 0 (room)
        # return {"image": image2/256, "label": label, "name": self.images[idx]}  # the range is now: 0.996 (white background) / 0.5 (apartment shape)  / 0 (room)

    def list_images(self):
        # mode = "val" if self.opt.phase == "test" or self.for_metrics else "train"
        mode = "test" if self.opt.phase == "test" or self.for_metrics else "train"
        path_img = os.path.join(self.opt.dataroot, mode, 'B')
        path_lab = os.path.join(self.opt.dataroot, mode, 'A')
        # path_img = os.path.join(self.opt.dataroot, 'train', 'B')  # no val dataset so i put 'test' here instead   todo: change if we make a val folder
        # path_lab = os.path.join(self.opt.dataroot, 'train', 'A')  # no val dataset so i put 'test' here instead   todo: change if we make a val folder
        images = sorted(os.listdir(path_img))
        labels = sorted(os.listdir(path_lab))
        assert len(images) == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            assert os.path.splitext(images[i])[0] == os.path.splitext(labels[i])[0], '%s and %s are not matching' % (
                images[i], labels[i])
        return images, labels, (path_img, path_lab)

    def transforms(self, image, label):
        assert image.size == label.size
        # resize
        new_width, new_height = (self.opt.load_size, self.opt.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.NEAREST)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        # crop
        crop_x = random.randint(0, np.maximum(0, new_width - self.opt.crop_size))
        crop_y = random.randint(0, np.maximum(0, new_height - self.opt.crop_size))
        image = image.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
        label = label.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
        # flip
        rotation = False
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
            if random.random() < 0.5:
                image = TR.functional.vflip(image)
                label = TR.functional.vflip(label)
            if rotation:
                angle = random.choice(self.angles)
                image = TR.functional.rotate(image, angle, fill=(255, 255, 255))
                label = TR.functional.rotate(label, angle, fill=(255,))
        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        # normalize
        # image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label

    def transforms_2im(self, image, label, image2):
        assert image.size == label.size
        # resize
        new_width, new_height = (self.opt.load_size, self.opt.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.NEAREST)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        # crop
        crop_x = random.randint(0, np.maximum(0, new_width - self.opt.crop_size))
        crop_y = random.randint(0, np.maximum(0, new_height - self.opt.crop_size))
        image = image.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
        label = label.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
        if image2 is not None:
            image2 = TR.functional.resize(image2, (new_width, new_height), Image.NEAREST)
            image2 = image2.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
        # flip     
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            # todo: remove the following horizontal flips if training with all entrances on one side):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
                if image2 is not None:
                    image2 = TR.functional.hflip(image2)
            if random.random() < 0.5:
                image = TR.functional.vflip(image)
                label = TR.functional.vflip(label)
                if image2 is not None:
                    image2 = TR.functional.vflip(image2)
            # if True:
            #     angle = random.choice(self.angles)

            #     image = TR.functional.rotate(image, angle, fill=(255,255,255))
            #     label = TR.functional.rotate(label, angle, fill=(255,))
        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        image2 = TR.functional.to_tensor(image2)
        # normalize
        # image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label, image2


# class HasekoDataset(torch.utils.data.Dataset):   # dataloader for dataset9 / dataset10 (same images but more rooms) / BIM data (nenw images, with more room types and sometimes different colors)
#     def __init__(self, opt, for_metrics):
#         # Defines the way each color is mapped to which channel. Here the different bedroom types are all mapped to the same one, and so they will have the same color in the final output
#         self.channels_inv = {'Background': 0,
#                              'Entrance': 1,
#                              'Window': 2,
#                              'Dead space': 3,
#                              'Corridor': 4,
#                              'Living room': 5,
#                              'Kitchen': 6,
#                              'Sanitary': 7,
#                              'Storage': 8,
#                              'pillars': 9,
#                              'Bed Room 1': 10,
#                              'Bed Room 2': 10,
#                              'Bed Room 3': 10,
#                              'Bed Room 4': 10,
#                              # 'Bed Room 2': 11,
#                              # 'Bed Room 3': 12,
#                              # 'Bed Room 4': 13,
#                              }
#
#         # Find out how many layers we have
#         n_layer = 0
#         for v in self.channels_inv.values():
#             if v > n_layer:
#                 n_layer = v
#         n_layer += 1
#
#         if opt.phase == "test" or for_metrics:
#             opt.load_size = 128
#         else:
#             opt.load_size = 128
#         opt.crop_size = 128
#         opt.label_nc = 5
#         opt.contain_dontcare_label = True
#         opt.semantic_nc = 6  # label_nc + unknown. Amount of layers in the input label map (black, pillars, entrance, window, white background)
#         opt.semantic_nc_image = n_layer + 1  # n_layers + unknown. Amount of layers in the target label map
#         opt.cache_filelist_read = False
#         opt.cache_filelist_write = False
#         opt.aspect_ratio = 1.0
#
#         self.opt = opt
#         self.for_metrics = for_metrics
#         self.images, self.labels, self.paths = self.list_images()
#         self.angles = [i for i in range(-90, 270, 90)]
#
#         self.colors = json.load(open("colors.json", 'r'))
#
#     def __len__(self, ):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
#         # image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGBA')
#         label = Image.open(os.path.join(self.paths[1], self.labels[idx])).convert('L')
#
#         image, label = self.transforms(image, label)
#
#         label = label * 255
#         image = image * 255
#         # image = image[:3] * 255   [:3] because to get just the 3 RGB channels
#         image = image.to('cpu').detach().numpy().copy().transpose((1, 2, 0))
#
#         _, h, w = label.size()
#
#         # image2 = torch.full(fill_value=255, size=(self.opt.semantic_nc_image, h, w))  # replace the following 4 lines by this one fot ablation test of the footprint-shaped background
#         image2 = torch.empty(self.opt.semantic_nc_image, h, w)
#         for i in range(self.opt.semantic_nc_image):
#             image2[i] = copy.deepcopy(label[0])
#         image2[image2 < 254] = 128  # 85     (the background is a white 255 already, and it puts all where there is content - the footprint - at 128)
#
#         room_seg_value = 0
#
#         # the channel 0 is already full of 255 values and the 128 values of the footprint because of the label copy and threshold
#
#         for coord in torch.tensor(np.where((image == (255, 0, 0)).all(axis=-1))).numpy().transpose():  # Entrance (red)
#             image2[1, coord[0], coord[1]] = room_seg_value
#
#         for coord in torch.tensor(np.where((image == (255, 0, 102)).all(axis=-1))).numpy().transpose():  # Window (dark pink)
#             image2[2, coord[0], coord[1]] = room_seg_value
#         for coord in torch.tensor(np.where((image == (255, 0, 100)).all(axis=-1))).numpy().transpose():  # Window (dark pink) (BIM)
#             image2[2, coord[0], coord[1]] = room_seg_value
#
#         for coord in torch.tensor(np.where((image == (0, 255, 255)).all(axis=-1))).numpy().transpose():  # Dead Space (bright blue)
#             image2[3, coord[0], coord[1]] = room_seg_value
#
#         for coord in torch.tensor(np.where((image == (255, 255, 0)).all(axis=-1))).numpy().transpose():  # Corridor (yellow)
#             image2[4, coord[0], coord[1]] = room_seg_value
#
#         for coord in torch.tensor(np.where((image == (128, 128, 128)).all(axis=-1))).numpy().transpose():  # Living room (medium grey)
#             image2[5, coord[0], coord[1]] = room_seg_value
#
#         for coord in torch.tensor(np.where((image == (20, 74, 25)).all(axis=-1))).numpy().transpose():  # Kitchen (dark forest green)
#             image2[6, coord[0], coord[1]] = room_seg_value
#         for coord in torch.tensor(np.where((image == (0, 75, 36)).all(axis=-1))).numpy().transpose():  # Kitchen (dark forest green) (BIM)
#             image2[6, coord[0], coord[1]] = room_seg_value
#
#         for coord in torch.tensor(np.where((image == (51, 255, 0)).all(axis=-1))).numpy().transpose():  # Sanitary (Sanitary) (bright green)
#             image2[7, coord[0], coord[1]] = room_seg_value
#         for coord in torch.tensor(np.where((image == (136, 0, 21)).all(axis=-1))).numpy().transpose():  # Sanitary (Toilettes) (burgundy)
#             image2[7, coord[0], coord[1]] = room_seg_value
#         for coord in torch.tensor(np.where((image == (34, 177, 76)).all(axis=-1))).numpy().transpose():  # Sanitary (Bathroom) (chlorophile medium green)
#             image2[7, coord[0], coord[1]] = room_seg_value
#
#         for coord in torch.tensor(np.where((image == (204, 0, 255)).all(axis=-1))).numpy().transpose():  # Storage  (bright pink)
#             image2[8, coord[0], coord[1]] = room_seg_value
#         for coord in torch.tensor(np.where((image == (213, 45, 247)).all(axis=-1))).numpy().transpose():  # Storage  (bright pink) (BIM)
#             image2[8, coord[0], coord[1]] = room_seg_value
#
#         for coord in torch.tensor(np.where((image == (255, 153, 0)).all(axis=-1))).numpy().transpose():  # pillars (orange)
#             image2[9, coord[0], coord[1]] = room_seg_value
#
#         for coord in torch.tensor(np.where((image == (0, 0, 255)).all(axis=-1))).numpy().transpose():  # Bed Room 1
#             image2[10, coord[0], coord[1]] = room_seg_value
#         for coord in torch.tensor(np.where((image == (0, 12, 102)).all(axis=-1))).numpy().transpose():  # Bed Room 2
#             image2[10, coord[0], coord[1]] = room_seg_value
#         for coord in torch.tensor(np.where((image == (24, 154, 180)).all(axis=-1))).numpy().transpose():  # Bed Room 3
#             image2[10, coord[0], coord[1]] = room_seg_value
#         for coord in torch.tensor(np.where((image == (177, 212, 224)).all(axis=-1))).numpy().transpose():  # Bed Room 4
#             image2[10, coord[0], coord[1]] = room_seg_value
#         # for coord in torch.tensor(np.where((image == (0, 12, 102)).all(axis=-1))).numpy().transpose():  # Bed Room 2
#         #     image2[11, coord[0], coord[1]] = room_seg_value
#         # for coord in torch.tensor(np.where((image == (24, 154, 180)).all(axis=-1))).numpy().transpose():  # Bed Room 3
#         #     image2[12, coord[0], coord[1]] = room_seg_value
#         # for coord in torch.tensor(np.where((image == (177, 212, 224)).all(axis=-1))).numpy().transpose():  # Bed Room 4
#         #     image2[13, coord[0], coord[1]] = room_seg_value
#
#         # for i in range(11):
#         #     print(f'image2[{i}] (min/max/mean): ', image2[i].min(), image2[i].max(), image2[i].mean())
#
#         # Converting the input image to a label map
#         label[label == 255] = 1  # White, background
#         label[label == 0] = 2  # Black, footprint
#         label[label == 166] = 3  # Orange, pillars
#         label[label == 88] = 4  # Entrance
#         label[label == 76] = 5  # Windows
#
#         return {"image": image2/510, "label": label, "name": self.images[idx]}  # range: 0.5 (white background) / 0.251 (apartment shape) / 0 (room)
#         # return {"image": image2/256, "label": label, "name": self.images[idx]}  # the range is now: 0.996 (white background) / 0.5 (apartment shape)  / 0 (room)
#
#     def list_images(self):
#         # mode = "val" if self.opt.phase == "test" or self.for_metrics else "train"
#         mode = "test" if self.opt.phase == "test" or self.for_metrics else "train"
#         path_img = os.path.join(self.opt.dataroot, mode, 'B')
#         path_lab = os.path.join(self.opt.dataroot, mode, 'A')
#         # path_img = os.path.join(self.opt.dataroot, 'train', 'B')  # no val dataset so i put 'test' here instead   todo: change if we make a val folder
#         # path_lab = os.path.join(self.opt.dataroot, 'train', 'A')  # no val dataset so i put 'test' here instead   todo: change if we make a val folder
#         images = sorted(os.listdir(path_img))
#         labels = sorted(os.listdir(path_lab))
#         assert len(images) == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
#         for i in range(len(images)):
#             assert os.path.splitext(images[i])[0] == os.path.splitext(labels[i])[0], '%s and %s are not matching' % (
#                 images[i], labels[i])
#         return images, labels, (path_img, path_lab)
#
#     def transforms(self, image, label):
#         assert image.size == label.size
#         # resize
#         new_width, new_height = (self.opt.load_size, self.opt.load_size)
#         image = TR.functional.resize(image, (new_width, new_height), Image.NEAREST)
#         label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
#         # crop
#         crop_x = random.randint(0, np.maximum(0, new_width - self.opt.crop_size))
#         crop_y = random.randint(0, np.maximum(0, new_height - self.opt.crop_size))
#         image = image.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
#         label = label.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
#         # flip
#         if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
#             if random.random() < 0.5:
#                 image = TR.functional.hflip(image)
#                 label = TR.functional.hflip(label)
#             if random.random() < 0.5:
#                 image = TR.functional.vflip(image)
#                 label = TR.functional.vflip(label)
#             # if True:
#             #     angle = random.choice(self.angles)
#
#             #     image = TR.functional.rotate(image, angle, fill=(255,255,255))
#             #     label = TR.functional.rotate(label, angle, fill=(255,))
#         # to tensor
#         image = TR.functional.to_tensor(image)
#         label = TR.functional.to_tensor(label)
#         # normalize
#         # image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         return image, label

