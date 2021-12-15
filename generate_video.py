import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
import config
import torch
from tqdm import tqdm
import cv2
import os
from scipy.ndimage import gaussian_filter
from math import sqrt, ceil
import matplotlib.pyplot as plt
import dataloaders.HasekoDataset as haseko

# --- read options --- #
opt = config.read_arguments(train=False)

# --- create dataloader --- #
_, dataloader_val = dataloaders.get_dataloaders(opt)
# dataset_val = haseko.HasekoDatasetInfer(opt)
# dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = 12, shuffle = False, drop_last=False)

# --- create utils --- #
image_saver = utils.results_saver(opt)

FPS = 30
SMOOTHING = 2.0
TIME = 5
OUTDIR = 'variations_' + opt.ckpt_iter[:-3] + 'k'
if not opt.no_EMA:
    OUTDIR += '_ema'
else:
    OUTDIR += '_no-ema'
os.makedirs(OUTDIR, exist_ok=True)

INTERPOLATION = False   # todo: set as True to do INTERPOLATION // set as False to do VARIATIONS

# total_frames in the video:
total_frames = int(TIME * FPS)

# --- create models --- #
model = models.OASIS_model(opt)
model = models.put_on_multi_gpus(model, opt)
model.eval()


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


# --- iterate over validation set --- #
for i, data_i in enumerate(dataloader_val):

    if i in [0, 1, 2, 3, 4, 5, 6]:
        continue

    label = models.preprocess_label(opt, data_i)
    num_cl_image = opt.semantic_nc_image + 1

    # Let's create the animation video from the latent space interpolation
    # all latent vectors:

    # all_latents = 2.0 * torch.randn(total_frames, opt.z_dim).to(label.get_device())
    if INTERPOLATION:
        latents_low = torch.randn(1, opt.z_dim).to(label.get_device())
        latents_high = torch.randn(1, opt.z_dim).to(label.get_device())
        ratios = torch.linspace(0., 8., total_frames)
        all_latents = [slerp(ratio, latents_low, latents_high) for ratio in ratios]
    else:
        # all_latents = 2.0 * torch.randn(total_frames, opt.z_dim).to(label.get_device())
        all_latents = 1.0 * torch.randn(total_frames, opt.z_dim).to(label.get_device())

    # all_latents = gaussian_filter(all_latents.cpu(), [SMOOTHING * FPS, 0])
    # all_latents = torch.from_numpy(all_latents)
    # all_latents = (all_latents / all_latents.norm(dim=-1, keepdim=True)) * (sqrt(opt.z_dim))

    global_frame_counter = 1

    # Run the main loop for the interpolation:
    print("Generating the video frames ...")
    for latent in tqdm(all_latents):

        # latent = torch.unsqueeze(latent, dim=0).repeat(label.size(0), 1)
        if INTERPOLATION:
            latent = latent.repeat(label.size(0), 1)
        else:
            latent = torch.unsqueeze(latent, dim=0).repeat(label.size(0), 1)

        # generate the image for this point:
        generated = model(None, label, "generate", None, z=latent)

        # fig = plt.figure()
        fig = plt.figure(figsize=(4, 3), dpi=256, frameon=False)    # i changed this line to set the figure to the right size
        for j in range(len(generated)):
            # im = utils.tens_to_im(generated[j], num_cl_image)
            im = utils.tens_to_im(generated[j], label[j], num_cl_image)  # i changed this line (to add the label)

            # plt.axis("off")              # i removed this
            fig.add_subplot(3, 4, j + 1)
            plt.axis("off")
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)   # i added this to remove any gaps and borders

            plt.imshow(im)
        # fig.tight_layout()     # i removed this (to keep the layout as decided)
        plt.savefig(os.path.join(OUTDIR, str(global_frame_counter) + ".png"))
        plt.close()

        # increment the counter:
        global_frame_counter += 1
    break


# video frames have been generated
print("Video frames have been generated at:", OUTDIR)

# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter(os.path.join(OUTDIR, 'out.mp4'), fourcc, 30.0, (640, 480))
#
# for i in range(1, total_frames + 1):
#     frame = cv2.imread(os.path.join(OUTDIR, "{}.png".format(i)))
#     out.write(frame)

# out.release()
