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
OUTDIR = 'videos-out'
os.makedirs(OUTDIR, exist_ok=True)

# total_frames in the video:
total_frames = int(TIME * FPS)

# --- create models --- #
model = models.OASIS_model(opt)
model = models.put_on_multi_gpus(model, opt)
model.eval()

# --- iterate over validation set --- #
for i, data_i in enumerate(dataloader_val):

    if i in [0, 1, 2, 3, 4, 5, 6]:
        continue

    label = models.preprocess_label(opt, data_i)

    num_cl_image = opt.semantic_nc_image + 1

    # Let's create the animation video from the latent space interpolation
    # all latent vectors:
    all_latents = 2.0 * torch.randn(total_frames, opt.z_dim).to(label.get_device())
    # all_latents = gaussian_filter(all_latents.cpu(), [SMOOTHING * FPS, 0])
    # all_latents = torch.from_numpy(all_latents)
    # all_latents = (all_latents / all_latents.norm(dim=-1, keepdim=True)) \
    #                 * (sqrt(opt.z_dim))

    global_frame_counter = 1
    # Run the main loop for the interpolation:
    print("Generating the video frames ...")
    for latent in tqdm(all_latents):
        latent = torch.unsqueeze(latent, dim=0).repeat(label.size(0), 1)

        # generate the image for this point:
        generated = model(None, label, "generate", None, z=latent)

        fig = plt.figure()
        for i in range(len(generated)):
            im = utils.tens_to_im(generated[i], num_cl_image)

            plt.axis("off")
            fig.add_subplot(3, 4, i + 1)
            plt.axis("off")
            plt.imshow(im)
        fig.tight_layout()
        plt.savefig(os.path.join(OUTDIR, str(global_frame_counter) + ".png"))
        plt.close()

        # increment the counter:
        global_frame_counter += 1
    break

# video frames have been generated
print("Video frames have been generated at:", OUTDIR)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(os.path.join(OUTDIR, 'out.mp4'), fourcc, 30.0, (640, 480))

for i in range(1, total_frames + 1):
    frame = cv2.imread(os.path.join(OUTDIR, "{}.png".format(i)))
    out.write(frame)

out.release()
