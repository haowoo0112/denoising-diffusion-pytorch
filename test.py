import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from PIL import Image
import os

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

trainer.load(100) # load model-4.pt # load the checkpoint

sampled_images = diffusion.sample(batch_size=8)

samples_root = r"./samples"
os.makedirs(samples_root , exist_ok=True)
len_samples = len(os.listdir(samples_root))

for i in range(sampled_images.size(0)):

	current_image_tensor = sampled_images [i]
	current_image = Image.fromarray((current_image_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype('uint8'))
	file_name = f"output__image_{i + len_samples}.png"
	current_image.save(os.path.join(os.getcwd(),"samples/" + file_name))

print("all samples are save in folder")
