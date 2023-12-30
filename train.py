import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from PIL import Image

if __name__ == '__main__':
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=3
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=128,
        timesteps=1000,
        sampling_timesteps=250,
    ).cuda()

    trainer = Trainer(
        diffusion,
        r'dataset',
        train_batch_size=16,
        train_lr=8e-5,
        save_and_sample_every=1000,
        train_num_steps=700000,
        gradient_accumulate_every=1,
        calculate_fid=False,
        ema_decay=0.995,
        amp=True
    )

    if torch.cuda.is_available():
        torch.multiprocessing.freeze_support()

    trainer.train()
    
