"""
Pix2Pix Training Script (Paired Image-to-Image Translation)

Goal
----
Learn a mapping:  BEFORE image  ->  AFTER image
Example: (before face) -> (after botox face)

Pix2Pix uses TWO networks:
1) Generator (G): makes a fake "after" image from a "before" image
2) Discriminator (D): judges if (before, after) pairs look real or fake

"""


import os
# Used to work with folders and file paths
# Example in this code: loading dataset images and creating output directories

from PIL import Image  # (Python Imaging Library)
# Used to open and read image files (before/after images)

from tqdm import tqdm  # Progress bar library
# Displays a progress bar during training loop

import torch
# Core PyTorch library for tensor operations and GPU computation
# Used for model training, tensor math, and device selection (CPU/GPU)

import torch.nn as nn
# Contains neural network layers and loss functions
# Used to build Generator and Discriminator architectures

from torch.utils.data import Dataset, DataLoader
# Dataset: creates a custom dataset class (PairedDataset) for loading image pairs
# DataLoader: loads data in batches and feeds it to the training loop

import torchvision.transforms as T
# Used for image preprocessing and augmentation
# Example: resize, random flip, rotation, normalization, converting images to tensors

from torchvision.utils import save_image
# Saves generated images during training
# Used to store before | generated | real comparison samples


# evice Selection
# Use GPU (CUDA) if available for faster training.
# If no GPU is available, fall back to CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Dataset class for loading paired before/after images
class PairedDataset(Dataset):

    def __init__(self, root, split="train", image_size=256, augment=False):
        # Path to before images folder
        self.before_dir = os.path.join(root, split, "before")
        # Path to after images folder
        self.after_dir  = os.path.join(root, split, "after")
        # Get all image filenames from the before folder
        self.files = sorted([
            f for f in os.listdir(self.before_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        # Flag to enable/disable augmentation
        self.augment = augment
        # Basic transform pipeline (no randomness)
        self.base = T.Compose([
            T.Resize((image_size, image_size)),  # resize image
            T.ToTensor(),                        # convert image to tensor
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # scale pixels to [-1,1]
        ])
        # Augmentation pipeline (includes random transforms)
        self.aug = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),       # randomly flip image
            T.RandomRotation(5),                 # randomly rotate image
            T.ColorJitter(brightness=0.10, contrast=0.10),  # random brightness/contrast
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    # Returns number of samples in the dataset
    def __len__(self):
        return len(self.files)
    # Loads and processes one sample
    def __getitem__(self, idx):
        # Get filename for this index
        fname = self.files[idx]
        # Build full paths for before and after images
        bpath = os.path.join(self.before_dir, fname)
        apath = os.path.join(self.after_dir, fname)
        # Open images and convert to RGB
        before = Image.open(bpath).convert("RGB")
        after  = Image.open(apath).convert("RGB")
        if self.augment:
            # Generate random seed so both images get identical random transforms
            seed = torch.randint(0, 10_000_000, (1,)).item()
            torch.manual_seed(seed)
            before = self.aug(before)   # apply augmentation to before image
            torch.manual_seed(seed)
            after  = self.aug(after)    # apply same augmentation to after image
        else:
            before = self.base(before)  # apply basic transform
            after  = self.base(after)
        # Return processed images and filename
        return before, after, fname



# Pix2Pix Generator (U-Net) - Downsampling Block
def down_block(in_c, out_c, norm=True):
    # Convolution layer  
    # [in_c  = number of input channels (features coming in), 
    # out_c = number of output channels (features produced)]
    layers = [  nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)  ]
    # If normalization is enabled
    if norm:
        # Batch Normalization layer
        layers.append(nn.BatchNorm2d(out_c))
    # LeakyReLU activation function
    # Adds non-linearity so the network can learn complex patterns
    # 0.2 = slope for negative values (prevents dead neurons)
    # inplace=True modifies tensor in memory to save RAM
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    # Combine all layers into a single sequential block
    return nn.Sequential(*layers)

# Pix2Pix Generator (U-Net) - Upsampling Block
def up_block(in_c, out_c, dropout=False):
    # Transposed Convolution layer (upsampling)
    layers = [
        nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
        # Batch Normalization layer
        nn.BatchNorm2d(out_c),
        # ReLU activation function
        # Introduces non-linearity so model can learn patterns
        nn.ReLU(inplace=True),
    ]
    # Optional Dropout layer
    # Randomly disables 50% of neurons during training
    if dropout:
        layers.append(nn.Dropout(0.5))
    # Combine layers into one sequential block
    return nn.Sequential(*layers)

class UNetGenerator(nn.Module):

    def __init__(self, in_c=3, out_c=3):
        super().__init__()  # initialize PyTorch neural network module

        # -------- Encoder (Downsampling path) --------
        # Gradually reduces image size while increasing feature depth

        self.d1 = down_block(in_c, 64, norm=False)   # RGB image → 64 features, 256 → 128
        self.d2 = down_block(64, 128)                # 128 → 64
        self.d3 = down_block(128, 256)               # 64 → 32
        self.d4 = down_block(256, 512)               # 32 → 16
        self.d5 = down_block(512, 512)               # 16 → 8
        self.d6 = down_block(512, 512)               # 8 → 4
        self.d7 = down_block(512, 512)               # 4 → 2
        self.d8 = down_block(512, 512, norm=False)   # Bottleneck: 2 → 1

        # -------- Decoder (Upsampling path) --------
        # Gradually rebuilds image size using skip connections

        self.u1 = up_block(512, 512, dropout=True)   # 1 → 2 (start reconstruction)
        self.u2 = up_block(1024, 512, dropout=True)  # concat encoder features → 1024 channels
        self.u3 = up_block(1024, 512, dropout=True)  # 4 → 8
        self.u4 = up_block(1024, 512)                # 8 → 16
        self.u5 = up_block(1024, 256)                # 16 → 32
        self.u6 = up_block(512, 128)                 # 32 → 64
        self.u7 = up_block(256, 64)                  # 64 → 128

        # -------- Final Output Layer --------
        # Converts feature maps back to RGB image
        self.out = nn.Sequential(
            nn.ConvTranspose2d(128, out_c, 4, 2, 1),  # Upscale 128 → 256 resolution
            nn.Tanh()  # scale pixel values to [-1,1] to match normalized images
        )

def forward(self, x):
    # -------- Encoder (Downsampling) --------
    # Image is compressed while extracting deeper features
    d1 = self.d1(x)     # first down block: 256 → 128
    d2 = self.d2(d1)    # 128 → 64
    d3 = self.d3(d2)    # 64 → 32
    d4 = self.d4(d3)    # 32 → 16
    d5 = self.d5(d4)    # 16 → 8
    d6 = self.d6(d5)    # 8 → 4
    d7 = self.d7(d6)    # 4 → 2
    d8 = self.d8(d7)    # bottleneck: 2 → 1 (most compressed representation)
    # -------- Decoder (Upsampling) --------
    # Image is reconstructed using skip connections
    u1 = self.u1(d8)  # start upsampling: 1 → 2
    # torch.cat combines decoder output with encoder features (skip connection)
    u2 = self.u2(torch.cat([u1, d7], dim=1))  # concat with d7 → 2 → 4
    u3 = self.u3(torch.cat([u2, d6], dim=1))  # concat with d6 → 4 → 8
    u4 = self.u4(torch.cat([u3, d5], dim=1))  # concat with d5 → 8 → 16
    u5 = self.u5(torch.cat([u4, d4], dim=1))  # concat with d4 → 16 → 32
    u6 = self.u6(torch.cat([u5, d3], dim=1))  # concat with d3 → 32 → 64
    u7 = self.u7(torch.cat([u6, d2], dim=1))  # concat with d2 → 64 → 128
    # -------- Final Output --------
    # Combine with earliest features and generate RGB image
    return self.out(torch.cat([u7, d1], dim=1))  # 128 → 256 output image


# Pix2Pix Discriminator (PatchGAN)
# This network checks if the generated image looks real or fake

class PatchDiscriminator(nn.Module):

    def __init__(self, in_c=6):  
        # in_c=6 because Pix2Pix stacks input image (3 channels) 
        # and output image (3 channels) → 3 + 3 = 6 channels
        super().__init__()

        # Sequential model: layers run one after another
        self.net = nn.Sequential(

            # First convolution layer
            # Reduces image size and extracts basic features
            nn.Conv2d(in_c, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),  # activation function

            # Second convolution block
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),              # normalize features for stable training
            nn.LeakyReLU(0.2, inplace=True),

            # Third convolution block
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Fourth convolution block
            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Final output layer
            # Produces a map of real/fake predictions for image patches
            nn.Conv2d(512, 1, 4, 1, 1)  # logits map (PatchGAN output)
        )

    # Defines how the input flows through the network
    def forward(self, x):
        return self.net(x)  # returns patch-wise real/fake predictions


# Helpers
def denorm(x):
    # Convert image values from [-1, 1] back to [0, 1]
    # Needed because save_image expects pixels in [0,1]
    return (x * 0.5) + 0.5



# Training Loop
def train_pix2pix(
    data_root="botox_dataset",      # dataset folder
    out_dir="runs_botox_pix2pix",   # where to save results
    image_size=256,                # resize images to 256x256
    epochs=500,                    # repeat training many times (small dataset)
    batch_size=1,                  # train on 1 image-pair at a time (tiny dataset)
    lr=2e-4,                       # learning rate
    lambda_L1=100.0                # strength of L1 loss (match real "after")
):
    # Create output folders (no error if they already exist)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)  # saved image results
    os.makedirs(os.path.join(out_dir, "weights"), exist_ok=True)  # saved model files

    # Build datasets
    # Training uses augmentation; validation does not (true evaluation)
    train_ds = PairedDataset(data_root, "train", image_size=image_size, augment=True)
    val_ds   = PairedDataset(data_root, "val",   image_size=image_size, augment=False)

    # DataLoaders feed data in batches
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Create Generator (G) and Discriminator (D) and move to GPU/CPU
    G = UNetGenerator().to(DEVICE)
    D = PatchDiscriminator().to(DEVICE)

    # Optimizers (update weights of G and D)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss functions
    bce = nn.BCEWithLogitsLoss()  # GAN loss: real vs fake
    l1  = nn.L1Loss()             # L1 loss: pixel similarity (fake_after vs real_after)

    # Loop over epochs (one full pass over training dataset)
    for epoch in range(1, epochs + 1):
        G.train()  # enable training mode (dropout/bn behavior)
        D.train()

        # Progress bar over training batches
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}")

        # Loop over batches
        for before, after, _ in pbar:
            # Move batch data to GPU/CPU
            before = before.to(DEVICE)
            after  = after.to(DEVICE)

            # -----------------------------
            # (A) Train Discriminator (D)
            # -----------------------------

            # Create fake output WITHOUT updating G (saves memory + isolates D training)
            with torch.no_grad():
                fake_after = G(before)

            # Build input pairs for discriminator (6 channels total)
            real_pair = torch.cat([before, after], dim=1)       # (before + real_after)
            fake_pair = torch.cat([before, fake_after], dim=1)  # (before + fake_after)

            # Discriminator predictions (PatchGAN score maps)
            pred_real = D(real_pair)  # should be "real"
            pred_fake = D(fake_pair)  # should be "fake"

            # Discriminator loss:
            # real targets = 1, fake targets = 0
            loss_D = 0.5 * (
                bce(pred_real, torch.ones_like(pred_real)) +
                bce(pred_fake, torch.zeros_like(pred_fake))
            )

            # Update discriminator weights
            opt_D.zero_grad(set_to_none=True)  # clear old gradients
            loss_D.backward()                  # compute new gradients
            opt_D.step()                       # apply update

            # -----------------------------
            # (B) Train Generator (G)
            # -----------------------------

            # Generate fake output AGAIN (this time we want gradients for G)
            fake_after = G(before)

            # Discriminator sees the fake pair again
            fake_pair  = torch.cat([before, fake_after], dim=1)
            pred_fake  = D(fake_pair)

            # Generator adversarial loss:
            # wants D to predict "real" for fake outputs
            adv_loss = bce(pred_fake, torch.ones_like(pred_fake))

            # Generator L1 loss:
            # wants fake_after to look like the real after image (pixel match)
            l1_loss  = l1(fake_after, after) * lambda_L1

            # Total generator loss
            loss_G = adv_loss + l1_loss

            # Update generator weights
            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            opt_G.step()

            # Show losses on progress bar
            pbar.set_postfix({
                "D": f"{loss_D.item():.3f}",
                "G": f"{loss_G.item():.3f}",
                "L1": f"{l1_loss.item():.1f}",
            })

        # -----------------------------
        # Save validation samples + model weights
        # -----------------------------
        if epoch == 1 or epoch % 25 == 0:
            G.eval()  # evaluation mode (no dropout randomness)
            with torch.no_grad():
                # Save up to 6 sample outputs from validation set
                for i, (b, a, fname) in enumerate(val_dl):
                    if i >= 6:
                        break

                    # Move images to GPU/CPU
                    b = b.to(DEVICE)
                    a = a.to(DEVICE)

                    # Generate output
                    f = G(b)

                    # Stack: before | generated | real_after (for easy viewing)
                    stack = torch.cat([denorm(b), denorm(f), denorm(a)], dim=0)

                    # Save image file to samples folder
                    save_image(
                        stack,
                        os.path.join(out_dir, "samples", f"epoch{epoch:04d}_{fname[0]}"),
                        nrow=1
                    )

            # Save model weights to disk (.pt files)
            torch.save(G.state_dict(), os.path.join(out_dir, "weights", f"G_epoch{epoch:04d}.pt"))
            torch.save(D.state_dict(), os.path.join(out_dir, "weights", f"D_epoch{epoch:04d}.pt"))

    # Training finished message
    print("\n Training finished.")
    print("Check samples in:", os.path.join(out_dir, "samples"))
    print("Generator weights in:", os.path.join(out_dir, "weights"))


# Run training only if this file is executed directly
if __name__ == "__main__":
    train_pix2pix(
        data_root="botox_dataset",
        out_dir="runs_botox_pix2pix",
        image_size=256,
        epochs=500,
        batch_size=1,
        lr=2e-4,
        lambda_L1=100.0
    )