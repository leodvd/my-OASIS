import torch
import numpy as np
import random
import time
import os
import models.models as models
import matplotlib.pyplot as plt
from PIL import Image
import json


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_start_iters(start_iter, dataset_size):
    if start_iter == 0:
        return 0, 0
    start_epoch = (start_iter + 1) // dataset_size
    start_iter = (start_iter + 1) % dataset_size
    return start_epoch, start_iter


class results_saver():  # used in test.py
    def __init__(self, opt):
        self.path = os.path.join(opt.results_dir, opt.name, opt.ckpt_iter)
        # self.path_label = os.path.join(self.path, "label")
        # self.path_image = os.path.join(self.path, "image")
        # self.path_handcrafted = os.path.join(self.path, "handcrafted")

        # self.path_to_save = {"label": self.path_label, "image": self.path_image}
        # os.makedirs(self.path_label, exist_ok=True)
        # os.makedirs(self.path_image, exist_ok=True)
        # os.makedirs(self.path_handcrafted, exist_ok=True)
        os.makedirs(self.path, exist_ok=True)
        self.num_cl = opt.label_nc + 2
        self.num_cl_image = opt.semantic_nc_image + 1

    def __call__(self, label, generated, image, name):
        assert len(label) == len(generated)
        for i in range(len(label)):
            # im = tens_to_lab(label[i], self.num_cl)
            # self.save_im(im, "label", name[i])
            # im = tens_to_im(generated[i]) * 255
            # self.save_im(im, "image", name[i])
            self.save_trio(label[i], generated[i], image[i], name[i])

    # def save_im(self, im, mode, name):
    #     im = Image.fromarray(im.astype(np.uint8))
    #     im.save(os.path.join(self.path_to_save[mode], name.split("/")[-1]).replace('.jpg', '.png'))

    def save_trio(self, label, fake, image, name):
        fig = plt.figure()
        imgs = [label, fake, image]
        for i in range(3):
            if i == 0:
                im = tens_to_lab(imgs[i], self.num_cl)
            else:
                im = tens_to_im(imgs[i], self.num_cl_image)
            plt.axis("off")
            fig.add_subplot(1, 3, i + 1)
            plt.axis("off")
            plt.imshow(im)
        fig.tight_layout()
        plt.savefig(os.path.join(self.path, name))
        plt.close()


class timer():
    def __init__(self, opt):
        self.prev_time = time.time()
        self.prev_epoch = 0
        self.num_epochs = opt.num_epochs
        self.file_name = os.path.join(opt.checkpoints_dir, opt.name, "progress.txt")

    def __call__(self, epoch, cur_iter):
        if cur_iter != 0:
            avg = (time.time() - self.prev_time) / (cur_iter - self.prev_epoch)
        else:
            avg = 0
        self.prev_time = time.time()
        self.prev_epoch = cur_iter

        with open(self.file_name, "a") as log_file:
            log_file.write('[epoch %d/%d - iter %d], time:%.3f \n' % (epoch, self.num_epochs, cur_iter, avg))
        print('[epoch %d/%d - iter %d], time:%.3f' % (epoch, self.num_epochs, cur_iter, avg))
        return avg


class losses_saver():
    def __init__(self, opt):
        self.name_list = ["Generator", "Vgg", "D_fake", "D_real", "LabelMix"]
        self.opt = opt
        self.freq_smooth_loss = opt.freq_smooth_loss
        self.freq_save_loss = opt.freq_save_loss
        self.losses = dict()
        self.cur_estimates = np.zeros(len(self.name_list))
        self.path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses")
        self.is_first = True
        os.makedirs(self.path, exist_ok=True)
        for name in self.name_list:
            if opt.continue_train:
                self.losses[name] = np.load(self.path + "/losses.npy", allow_pickle=True).item()[name]
            else:
                self.losses[name] = list()

    def __call__(self, epoch, losses):
        for i, loss in enumerate(losses):
            if loss is None:
                self.cur_estimates[i] = None
            else:
                self.cur_estimates[i] += loss.detach().cpu().numpy()
        if epoch % self.freq_smooth_loss == self.freq_smooth_loss - 1:
            for i, loss in enumerate(losses):
                if not self.cur_estimates[i] is None:
                    self.losses[self.name_list[i]].append(self.cur_estimates[i] / self.opt.freq_smooth_loss)
                    self.cur_estimates[i] = 0
        if epoch % self.freq_save_loss == self.freq_save_loss - 1:
            self.plot_losses()
            np.save(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", "losses"), self.losses)

    def plot_losses(self):
        for curve in self.losses:
            fig, ax = plt.subplots(1)
            n = np.array(range(len(self.losses[curve]))) * self.opt.freq_smooth_loss
            plt.plot(n[1:], self.losses[curve][1:])
            plt.ylabel('loss')
            plt.xlabel('epochs')

            plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", '%s.png' % (curve)), dpi=600)
            plt.close(fig)

        fig, ax = plt.subplots(1)
        for curve in self.losses:
            if np.isnan(self.losses[curve][0]):
                continue
            plt.plot(n[1:], self.losses[curve][1:], label=curve)
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", 'combined.png'), dpi=600)
        plt.close(fig)


def update_EMA(model, cur_iter, dataloader, opt, force_run_stats=False):
    # update weights based on new generator weights
    with torch.no_grad():
        for key in model.module.netEMA.state_dict():
            model.module.netEMA.state_dict()[key].data.copy_(
                model.module.netEMA.state_dict()[key].data * opt.EMA_decay +
                model.module.netG.state_dict()[key].data * (1 - opt.EMA_decay)
            )
    # collect running stats for batchnorm before FID computation, image or network saving
    condition_run_stats = (force_run_stats or
                           cur_iter % opt.freq_print == 0 or
                           cur_iter % opt.freq_fid == 0 or
                           cur_iter % opt.freq_save_ckpt == 0 or
                           cur_iter % opt.freq_save_latest == 0
                           )
    if condition_run_stats:
        with torch.no_grad():
            num_upd = 0
            for i, data_i in enumerate(dataloader):
                image, label = models.preprocess_input(opt, data_i)
                fake = model.module.netEMA(label)
                num_upd += 1
                if num_upd > 50:
                    break


def save_networks(opt, cur_iter, model, latest=False, best=False):
    path = os.path.join(opt.checkpoints_dir, opt.name, "models")
    os.makedirs(path, exist_ok=True)
    if latest:
        torch.save(model.module.netG.state_dict(), path + '/%s_G.pth' % ("latest"))
        torch.save(model.module.netD.state_dict(), path + '/%s_D.pth' % ("latest"))
        if not opt.no_EMA:
            torch.save(model.module.netEMA.state_dict(), path + '/%s_EMA.pth' % ("latest"))
        with open(os.path.join(opt.checkpoints_dir, opt.name) + "/latest_iter.txt", "w") as f:
            f.write(str(cur_iter))
    elif best:
        torch.save(model.module.netG.state_dict(), path + '/%s_G.pth' % ("best"))
        torch.save(model.module.netD.state_dict(), path + '/%s_D.pth' % ("best"))
        if not opt.no_EMA:
            torch.save(model.module.netEMA.state_dict(), path + '/%s_EMA.pth' % ("best"))
        with open(os.path.join(opt.checkpoints_dir, opt.name) + "/best_iter.txt", "w") as f:
            f.write(str(cur_iter))
    else:
        torch.save(model.module.netG.state_dict(), path + '/%d_G.pth' % (cur_iter))
        torch.save(model.module.netD.state_dict(), path + '/%d_D.pth' % (cur_iter))
        if not opt.no_EMA:
            torch.save(model.module.netEMA.state_dict(), path + '/%d_EMA.pth' % (cur_iter))


class image_saver():  # used in train.py
    def __init__(self, opt):
        self.cols = 4
        self.rows = 3
        self.grid = 5
        self.path = os.path.join(opt.checkpoints_dir, opt.name, "images") + "/"
        self.opt = opt
        self.num_cl = opt.label_nc + 2
        self.num_cl_image = opt.semantic_nc_image + 1

        os.makedirs(self.path, exist_ok=True)

    def visualize_batch(self, model, image, label, cur_iter):
        self.save_images(label, "label", cur_iter, is_label=True)

        # self.save_images(image, "real", cur_iter)
        self.save_images(image, "real", cur_iter, label_to_pick_info_from=label)
        with torch.no_grad():
            model.eval()
            fake = model.module.netG(label)

            # self.save_images(fake, "fake", cur_iter)
            self.save_images(fake, "fake", cur_iter, label_to_pick_info_from=label)
            model.train()

            if not self.opt.no_EMA:
                model.eval()
                fake = model.module.netEMA(label)
                self.save_images(fake, "fake_ema", cur_iter)
                model.train()

    # def save_images(self, batch, name, cur_iter, is_label=False):
    def save_images(self, batch, name, cur_iter, is_label=False, label_to_pick_info_from=None):
        fig = plt.figure()
        for i in range(min(self.rows * self.cols, len(batch))):
            if is_label:
                im = tens_to_lab(batch[i], self.num_cl)
            else:
                # im = tens_to_im(batch[i], self.num_cl_image)
                im = tens_to_im(batch[i], label_to_pick_info_from[i], self.num_cl_image)
            plt.axis("off")
            fig.add_subplot(self.rows, self.cols, i + 1)
            plt.axis("off")
            plt.imshow(im)
        fig.tight_layout()
        plt.savefig(self.path + str(cur_iter) + "_" + name)
        plt.close()


# class image_saver():  # used in train.py  # old version that doesn't take the label tensor into account to generated the output (pillars, entry, background, windows)
#     def __init__(self, opt):
#         self.cols = 4
#         self.rows = 3
#         self.grid = 5
#         self.path = os.path.join(opt.checkpoints_dir, opt.name, "images") + "/"
#         self.opt = opt
#         self.num_cl = opt.label_nc + 2
#         self.num_cl_image = opt.semantic_nc_image + 1
#
#         os.makedirs(self.path, exist_ok=True)
#
#     def visualize_batch(self, model, image, label, cur_iter):
#         self.save_images(label, "label", cur_iter, is_label=True)
#         self.save_images(image, "real", cur_iter)
#         with torch.no_grad():
#             model.eval()
#             fake = model.module.netG(label)
#             self.save_images(fake, "fake", cur_iter)
#             model.train()
#
#             if not self.opt.no_EMA:
#                 model.eval()
#                 fake = model.module.netEMA(label)
#                 self.save_images(fake, "fake_ema", cur_iter)
#                 model.train()
#
#     def save_images(self, batch, name, cur_iter, is_label=False):
#         fig = plt.figure()
#         for i in range(min(self.rows * self.cols, len(batch))):
#             if is_label:
#                 im = tens_to_lab(batch[i], self.num_cl)
#             else:
#                 im = tens_to_im(batch[i], self.num_cl_image)
#             plt.axis("off")
#             fig.add_subplot(self.rows, self.cols, i + 1)
#             plt.axis("off")
#             plt.imshow(im)
#         fig.tight_layout()
#         plt.savefig(self.path + str(cur_iter) + "_" + name)
#         plt.close()


# def tens_to_im(tens, num_cl_image):
def tens_to_im(tens, label_tens, num_cl_image):
    # label_tensor = Colorize_im(tens, num_cl_image)
    label_tensor = Colorize_im(tens, label_tens, num_cl_image)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy


def tens_to_lab(tens, num_cl):
    label_tensor = Colorize(tens, num_cl)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy


###############################################################################
# Code below from
# https://github.com/visinf/1-stage-wseg/blob/38130fee2102d3a140f74a45eec46063fcbeaaf8/datasets/utils.py
# Modified so it complies with the Cityscapes label map colors (fct labelcolormap)
###############################################################################

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


# def Colorize_im(tens, num_cl_image):   # (mine, with pieces of furniture) faster(?)
def Colorize_im(tens, label_tens, num_cl_image):   # (mine, with pieces of furniture) faster(?)
    channels = {0: 'Dead space',
                1: 'Corridor',
                2: 'Living room',
                3: 'Kitchen',
                4: 'Sanitary',
                5: 'Toilets',
                6: 'Bathroom',
                7: 'Storage',
                8: 'Bed Room 1',
                9: 'Bath Tub',
                10: 'Washing Basin',
                11: 'Washing Machine Holder'
                # 5: 'Storage',
                # 6: 'Bed Room 1',
                # 7: 'Bath Tub',
                # 8: 'Washing Basin',
                # 9: 'Washing Machine Holder',
                # 10: 'Fridge Holder',
                # 11: 'Kitchen Counter',
                # 12: 'Toilet Bowl'
                }
    label_channels = {1: 'Background',
                      2: 'Footprint',
                      3: 'pillars',
                      4: 'Window',
                      5: 'Entrance'
                      }

    colors = json.load(open("colors.json", 'r'))

    size = tens.size()
    color_image = torch.ByteTensor(3, size[1], size[2]).fill_(255)
    # print(tens.shape)

    # px_rejected_from_threshold = (tens <= 0.21).sum(dim=0) == 0  # pixels without any channel having a strong enough (ie low enough) value. These pixels will remain white
    # tens = torch.argmin(tens, dim=0, keepdim=True)   # Compute argmin normally
    # tens = torch.where(px_rejected_from_threshold, torch.zeros_like(tens), tens)  # relabel rejected pixels as 0, ie background label (so that they endup white)

    tens_min = torch.argmin(tens[:len(channels)-3], dim=0, keepdim=True)   # Compute argmin normally (on the room channels)

    furniture_min = torch.argmin(tens[len(channels)-3:], dim=0, keepdim=True) + len(channels) - 3   # (on the furniture channels)
    # print('min: ', furniture_min.min(), '  max: ', furniture_min.max())
    furniture_threshold = 0.21
    furniture_px_rejected_from_threshold = (tens[len(channels)-3:] <= furniture_threshold).sum(dim=0) == 0  # pixels without any channel having a strong enough (ie low enough) value. These pixels will remain white
    furniture_min = torch.where(furniture_px_rejected_from_threshold, torch.zeros_like(furniture_min), furniture_min)  # relabel rejected pixels as 0

    # for label in range(0, len(channels)):
    for label in range(0, len(channels)-3):   # taking the strongest room from the room layers
        # mask = (label == tens[0]).cpu()
        mask = (label == tens_min[0]).cpu()
        color_image[0][mask] = colors[channels[label]][0]
        color_image[1][mask] = colors[channels[label]][1]
        color_image[2][mask] = colors[channels[label]][2]

    for label in range(len(channels) - 3, len(channels)):   # taking the strongest furniture (after thresholding the weak ones) from the furniture layers
        mask = (label == furniture_min[0]).cpu()
        color_image[0][mask] = colors[channels[label]][0]
        color_image[1][mask] = colors[channels[label]][1]
        color_image[2][mask] = colors[channels[label]][2]

    label_tens = torch.argmax(label_tens, dim=0, keepdim=True)   # pasting the rooms and background available in the input label on top of the output
    for label in range(1, 6):
        if label != 2:
            mask = (label == label_tens[0]).cpu()
            color_image[0][mask] = colors[label_channels[label]][0]
            color_image[1][mask] = colors[label_channels[label]][1]
            color_image[2][mask] = colors[label_channels[label]][2]

    return color_image

    # for label in range(0, 7):
    # # for label in range(0, 11):
    # # for label in range(0, 17):
    #     # a = torch.tensor(np.where(tens.cpu()[label] <= 0.15)).transpose(0, 1)
    #     a = torch.tensor(np.where(tens.cpu()[label] <= 0.21)).transpose(0, 1)
    #     # a = torch.tensor(np.where(tens.cpu()[label] <= 0.23)).transpose(0, 1)
    #     # print(f'label{label} (min/max/mean): ', tens.cpu()[label].min(), tens.cpu()[label].max(), tens.cpu()[label].mean())
    #     # print(f'label {label} :', a.size())
    #     for coord in a:
    #         if tens.cpu()[label, coord[0], coord[1]] < min_value[coord[0], coord[1]]:  # todo: check si les coordonnées sont bonnes pour tens
    #             min_value[coord[0], coord[1]] = tens.cpu()[label, coord[0], coord[1]]
    #             color_image[0, coord[0], coord[1]] = colors[channels[label]][0]
    #             color_image[1, coord[0], coord[1]] = colors[channels[label]][1]
    #             color_image[2, coord[0], coord[1]] = colors[channels[label]][2]
    #             # color_image[:, coord[0], coord[1]] = colors[channels[label]][:]  # au lieu des 3 lignes au dessus ?
    #     # print(label, channels[label], colors[channels[label]])
    #
    # # # for label in range(11, 17):  # i added this part so that furniture is added "on top"
    # # for label in range(7, 10):  # i added this part so that furniture is added "on top"
    # #
    # #     a = torch.tensor(np.where(tens.cpu()[label] <= 0.21)).transpose(0, 1)
    # #     for coord in a:
    # #         color_image[0, coord[0], coord[1]] = colors[channels[label]][0]
    # #         color_image[1, coord[0], coord[1]] = colors[channels[label]][1]
    # #         color_image[2, coord[0], coord[1]] = colors[channels[label]][2]
    #
    # return color_image


# def Colorize_im(tens, num_cl_image):   # (mine, with pieces of furniture)
#     channels = {0: 'Background',
#                 1: 'Entrance',
#                 2: 'Window',
#                 3: 'Dead space',
#                 4: 'Corridor',
#                 5: 'Living room',
#                 6: 'Kitchen',
#                 7: 'Sanitary',
#                 8: 'Storage',
#                 9: 'pillars',
#                 10: 'Bed Room 1',
#                 # 11:'Bed Room 2',
#                 # 12:'Bed Room 3',
#                 # 13:'Bed Room 4',
#                 11: 'Fridge Holder',
#                 12: 'Kitchen Counter',
#                 13: 'Bath Tub',
#                 14: 'Washing Basin',
#                 15: 'Washing Machine Holder',
#                 16: 'Toilet Bowl'
#                 }
#
#     # 'Fridge Holder', (40, 40, 40) very dark grey
#     # 'Kitchen Counter': (0, 0, 0) black
#     # 'Bath Tub' (20, 80, 90) petrol blue
#     # 'Washing Basin' (45, 25, 45) dark purple / aubergine
#     # 'Washing Machine Holder': (80, 80, 20) ocher
#     # 'Toilet Bowl': (100, 15, 35) burgundy
#
#     colors = json.load(open("colors.json", 'r'))
#
#     size = tens.size()
#     min_value = torch.ones(size=(size[1], size[2]))
#     color_image = torch.ByteTensor(3, size[1], size[2]).fill_(255)
#
#
#     for label in range(0, 11):
#     # for label in range(0, 17):
#         # a = torch.tensor(np.where(tens.cpu()[label] <= 0.15)).transpose(0, 1)
#         a = torch.tensor(np.where(tens.cpu()[label] <= 0.21)).transpose(0, 1)
#         # a = torch.tensor(np.where(tens.cpu()[label] <= 0.23)).transpose(0, 1)
#         # print(f'label{label} (min/max/mean): ', tens.cpu()[label].min(), tens.cpu()[label].max(), tens.cpu()[label].mean())
#         # print(f'label {label} :', a.size())
#         for coord in a:
#             if tens.cpu()[label, coord[0], coord[1]] < min_value[coord[0], coord[1]]:  # todo: check si les coordonnées sont bonnes pour tens
#                 min_value[coord[0], coord[1]] = tens.cpu()[label, coord[0], coord[1]]
#                 color_image[0, coord[0], coord[1]] = colors[channels[label]][0]
#                 color_image[1, coord[0], coord[1]] = colors[channels[label]][1]
#                 color_image[2, coord[0], coord[1]] = colors[channels[label]][2]
#                 # color_image[:, coord[0], coord[1]] = colors[channels[label]][:]  # au lieu des 3 lignes au dessus ?
#         # print(label, channels[label], colors[channels[label]])
#
#     for label in range(11, 17):  # i added this part so that furniture is added "on top"
#         a = torch.tensor(np.where(tens.cpu()[label] <= 0.21)).transpose(0, 1)
#         for coord in a:
#             color_image[0, coord[0], coord[1]] = colors[channels[label]][0]
#             color_image[1, coord[0], coord[1]] = colors[channels[label]][1]
#             color_image[2, coord[0], coord[1]] = colors[channels[label]][2]
#
#     return color_image


# def Colorize_im(tens, num_cl_image):   # (mine, for bedrooms in the same channel, no furniture)
#     channels = {0: 'Background',
#                 1: 'Entrance',
#                 2: 'Window',
#                 3: 'Dead space',
#                 4: 'Corridor',
#                 5: 'Living room',
#                 6: 'Kitchen',
#                 7: 'Sanitary',
#                 8: 'Storage',
#                 9: 'pillars',
#                 10: 'Bed Room 1',
#                 # 11:'Bed Room 2',
#                 # 12:'Bed Room 3',
#                 # 13:'Bed Room 4',
#                 }
#
#     colors = json.load(open("colors.json", 'r'))
#
#     size = tens.size()
#     min_value = torch.ones(size=(size[1], size[2]))
#     color_image = torch.ByteTensor(3, size[1], size[2]).fill_(255)
#
#     for label in range(0, 11):
#         # a = torch.tensor(np.where(tens.cpu()[label] <= 0.15)).transpose(0, 1)
#         a = torch.tensor(np.where(tens.cpu()[label] <= 0.21)).transpose(0, 1)
#         # a = torch.tensor(np.where(tens.cpu()[label] <= 0.23)).transpose(0, 1)
#         # print(f'label{label} (min/max/mean): ', tens.cpu()[label].min(), tens.cpu()[label].max(), tens.cpu()[label].mean())
#         # print(f'label {label} :', a.size())
#         for coord in a:
#             if tens.cpu()[label, coord[0], coord[1]] < min_value[coord[0], coord[1]]:  # todo: check si les coordonnées sont bonnes pour tens
#                 min_value[coord[0], coord[1]] = tens.cpu()[label, coord[0], coord[1]]
#                 color_image[0, coord[0], coord[1]] = colors[channels[label]][0]
#                 color_image[1, coord[0], coord[1]] = colors[channels[label]][1]
#                 color_image[2, coord[0], coord[1]] = colors[channels[label]][2]
#                 # color_image[:, coord[0], coord[1]] = colors[channels[label]][:]  # au lieu des 3 lignes au dessus ?
#         # print(label, channels[label], colors[channels[label]])
#     return color_image


def Colorize(tens, num_cl):
    cmap = labelcolormap(num_cl)
    cmap = torch.from_numpy(cmap[:num_cl])
    size = tens.size()
    color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
    tens = torch.argmax(tens, dim=0, keepdim=True)

    for label in range(0, len(cmap)):
        mask = (label == tens[0]).cpu()
        color_image[0][mask] = cmap[label][0]
        color_image[1][mask] = cmap[label][1]
        color_image[2][mask] = cmap[label][2]
    return color_image


def labelcolormap(N):
    if N == 35:
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70),
                         (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153),
                         (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
                         (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap
