# OASIS layer-based model code for Haseko

Based on : https://github.com/boschresearch/OASIS

The code has been repurposed to work with a per-room layer stack instead of an image as a target for generation (10+ channels instead of 3).


## Training

By default, most of the options are tailored to the Haseko dataset. A look into the config.py file will provide with info on what they do and their usage (or alternatively, running something like python train.py -h does that too)

Some notable ones are the following:

--no_EMA : Should always be used. The Exponential Moving Average (EMA) adds a second EMA model during training, where the weights are moved in an average fashion over several previous epochs. It doesn't bring much value during training, and subtantially slows it down, therefore it should be kept off

--channels_G and --channels_D : number of channels in the discriminator and generator. Subtantially changes parameter size. This is a good vector for experimentation, as more channel will yield bigger models, as well as the opposite. 16/32/64 are good values


Example of starting a training run:


```bash
python train.py --dataroot ../dataset_rooms/ --name run_name_1 --no_EMA --num_epochs 400 --channels_D 64 --channels_G 64
```

Example of starting a test run:


```bash
python test.py --dataroot ../dataset_rooms/ --name run_name_1 --no_EMA --channels_D 64 --channels_G 64
```


## Data augmentation

Check the HasekoDataset.py for it. Includes a horizontal and vertical flip by default. 90 degree rotation is also present but commented out.


## VGG loss and FID, labelmix

As we are not working with images, anything related to VGG loss and FID from the original image-based repository is not implemented/adapted to our case, and therefore should not be used.


The same goes for labelmix. All these features are off by default so they should just be left untouched.


## Code structure


OASIS-layer

├ utils

│├ utils.py : Contains code related to converting layer stacks back into images, saving images, networks ect

│├ fid_scores.py : code for computing fid scores, not implemented for this layer-based version

│└ fid_folder : ditto

├ models : Model file, generator, discriminator code... vggloss is not implemented in this layer-based version

├ dataloaders

│├ HasekoDataset.py : Where input images and targets are turned into label maps (layer stacks) instead of images

│└ dataloaders.py : Small file for providing the correct dataloader (The Haseko one only in our case)

├ checkpoints : Training runs with loss curves, parameters, model .pt files, images..

├ results : Validation set results are stored when test.py is ran

├ train.py

├ test.py : Will apply a trained model on the validation set of the data. Produces outputs in the results folder

├ generate_video.py : Used for generating a video where different 3D noises are sampled to produce different outputs on the same given input footprint

├ config.py : All the parameters for training/testing

└ colors.json : contains a dictonary for mapping layers to a given color, specified in RGB format. Used by utils.py and HasekoDataset.py

