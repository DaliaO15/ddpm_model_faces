# I followed this notebook for the basic training:
# https://colab.research.google.com/drive/14Pez9Bs21I6Phw27Byu0jLk23_YFKqNb#scrollTo=1f740dfe-e610-4479-ac30-cce1f9e62553

from datasets import load_dataset
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os
import json
from PIL import Image
import time
# For data processing
from torchvision import transforms
from torchvision.utils import make_grid
import torch
import gc 
# Import some classes
import utils
import config as cf
# To track the losses 
from torch.utils.tensorboard import SummaryWriter

# ----------------------

config = cf.TrainingConfig()
run_name = "full_data_ema_long"
#run_name = "test1"
config.num_epochs = 300


# For the board
writer = SummaryWriter(os.path.join(config.output_dir, os.path.join("tensorboard", run_name))) # initate tensorboard


# ------------------ Getting the dataset from HF

config.dataset_name = "HuggingFaceM4/FairFace"

# Making a small dataset for testing 

#big_valid_dataset = load_dataset(config.dataset_name, "0.25", split="validation") # uncomment for testing
#small_ds = big_valid_dataset.train_test_split(test_size=0.2,seed=123,stratify_by_column="race",)# uncomment for testing
#small_ds = small_ds["train"].train_test_split(test_size=0.2,seed=123,stratify_by_column="race",)# uncomment for testing

#train_dataset = small_ds["train"] # uncomment for testing
#valid_dataset = small_ds["test"] # uncomment for testing

train_dataset = load_dataset(config.dataset_name, "0.25", split="train") # comment for testing
valid_dataset = load_dataset(config.dataset_name, "0.25", split="validation") # comment for testing

print(f"The small training dataset has {train_dataset.shape[0]} instances.")
print(f"The small validation dataset has {valid_dataset.shape[0]} instances.")


# ------------------ Processing data to fit in the model requirements

# Note: It may be a good idea to implement something to improve the quality of the images

train_dataset.set_transform(lambda examples: utils.transform_examples(examples, config.image_size))
valid_dataset.set_transform(lambda examples: utils.transform_examples(examples, config.image_size))

# Initialise data loader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.eval_batch_size, shuffle=False)

# ------------------ Define the difussion model

from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D"
      ),
)


# ------------------ Define noisy scheduler

from diffusers import DDPMScheduler # For the noise scheduler
from diffusers import DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup # for the learning rate
from diffusers import DDPMPipeline
from diffusers.training_utils import EMAModel # >>> EMA: added import
import torch.nn.functional as F # To compute the loss
# For training
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path



noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
#noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs))

early_stopper = utils.EarlyStopping(patience=config.early_stopping_patience, min_delta=config.early_stopping_min_delta)


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, val_dataloader, run_name):# earystopper):
    # Initialize accelerator and tensorboard logging
    
    logging_dir = os.path.join(config.output_dir, os.path.join("logs", run_name))
    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=Path(config.output_dir).name, exist_ok=True
            ).repo_id
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers(run_name)

    # EMA: Initialize EMA model
    ema_model = EMAModel(model.parameters(), power=0.9999)
    ema_model.to(accelerator.device)
    
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, val_dataloader
    )

    global_step = 0
    train_losses_per_epoch = []
    eval_losses_per_epoch = []

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        # ------ Training section
        model.train()
        epoch_train_loss = 0.0 # To save the loss per epoch
        
        for _, batch in enumerate(train_dataloader):
            clean_images = batch['images']
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                epoch_train_loss += loss.item() # Store the running loss
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                 # EMA: update EMA model
                ema_model.step(model)

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            

            # Track loss inside training step
            if global_step % 100 == 0:
                writer.add_scalar("Loss/train_step", loss.item(), global_step)

            global_step += 1

        # Save the train loss in this epoch
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        train_losses_per_epoch.append(avg_train_loss)

        
        # ------ Evaluation section
        
        epoch_eval_loss = 0.0
        num_eval_batches = 0

        # EMA: store current model params and apply EMA weights
        ema_model.store(model.parameters()) # Saves the current parameters for restoring later.
        ema_model.copy_to(model.parameters()) # Copy current averaged parameters into given collection of parameters
        model.eval()

        for _, batch in enumerate(val_dataloader):
            clean_images = batch['images']
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)#.to(accelerator.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            with torch.no_grad():
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                epoch_eval_loss += loss.item()
                num_eval_batches += 1
        
        
        # Save the evaluation loss in this epoch
        avg_eval_loss = epoch_eval_loss / num_eval_batches
        eval_losses_per_epoch.append(avg_eval_loss)
        
        # >>> EMA: restore original weights
        ema_model.restore(model.parameters())

        # Come back to training
        model.train()

        #  ------ Early stopping & sample generation
        if accelerator.is_main_process:
            # >>> EMA: unwrap EMA version for sample generation
            ema_model.store(model.parameters())
            ema_model.copy_to(model.parameters())

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                #ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config) # Added for faster sampling
                #ddim_scheduler = DDIMScheduler(
                #                num_train_timesteps=1000,  # Must match the training scheduler
                #                beta_start=0.0001,         # Match training
                #                beta_end=0.02,             # Match training
                #                beta_schedule="linear",  # match training scheduler
                #                clip_sample=False,
                #                set_alpha_to_one=False,   # True for DDPM, but False works better for DDIM in many cases
                #                steps_offset=0)
                #ddim_scheduler.set_timesteps(num_inference_steps=50)  # Can try 25, 50, 100
                #pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=ddim_scheduler)
                # TODO: Add a toggle to evaluate with DDIM

                # Defer pipeline creation until needed
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                
                utils.evaluate(config, epoch, pipeline, run_name)

                # Explicitly delete pipeline to free memory
                del pipeline
                gc.collect()
                torch.cuda.empty_cache()

            # Save best model based on EMA weights
            improved = early_stopper.step(avg_eval_loss)
            if improved:
                best_model_path = os.path.join(config.output_dir, run_name, "best_model")
                
                # Create pipeline only when needed
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                pipeline.save_pretrained(best_model_path)

                # Clean up
                del pipeline
                gc.collect()
                torch.cuda.empty_cache()

                print(f"Saved new best model at epoch {epoch} with loss {avg_eval_loss:.4f}")
            else:
                print(f"No improvement. Patience counter: {early_stopper.counter}/{early_stopper.patience}")

            ema_model.restore(model.parameters())

        if early_stopper.should_stop:
            print("Early stopping triggered.")
            break  

        # Print progress bar and add scalars to the board
        print(f"Epoch {epoch}: Avg train Loss = {avg_train_loss:.4f}, Avg eval Loss = {avg_eval_loss:.4f}")
        writer.add_scalars("Losses per epoch", { "train": avg_train_loss, "eval": avg_eval_loss}, epoch)
                    
    writer.close()

    # To store the losses
    losses_folder = os.path.join(config.output_dir, os.path.join("losses", run_name))
    os.makedirs(losses_folder, exist_ok=True)
    with open(os.path.join(losses_folder,"train_losses.json"), "w") as f:
        json.dump(train_losses_per_epoch, f)
    with open(os.path.join(losses_folder,"eval_losses.json"), "w") as f:
        json.dump(eval_losses_per_epoch, f)

print(f"Run EMA model\n on project {run_name}\
        with train size {train_dataset.shape[0]} and val size {valid_dataset.shape[0]} \
        number of epochs {config.num_epochs}")


if __name__ == "__main__":

    torch.cuda.synchronize()
    start_time = time.time()

    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, val_dataloader, run_name)#, early_stopper)
    
    print(f"\n\nRun EMA model on project {run_name}\n\
        with train size {train_dataset.shape[0]} and val size {valid_dataset.shape[0]}\n \
        number of epochs {config.num_epochs}\n\n")
    
    torch.cuda.synchronize()
    print(f"Training time: {(time.time() - start_time):.2f} seconds")


