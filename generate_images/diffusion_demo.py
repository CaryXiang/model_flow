import os

import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda"
model_id_or_path = "Cyberpunk-Anime-Diffusion"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path)
pipe = pipe.to(device)


prompt = "Woman with cat ears"

for i in os.listdir("diffsuion"):
    if ".jpg" in i:
        dir_path = "diffsuion/" + i[:-4]
        file_path = "diffsuion/" + i
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    init_image = Image.open(file_path).convert("RGB")
    init_image = init_image.resize((768, 512))
    images = pipe(prompt=prompt, init_image=init_image, strength=0.5, guidance_scale=20, num_images_per_prompt=4, num_inference_steps=50).images
    for i in range(len(images)):
        images[i].save(dir_path + "/"+str(i)+".png", quality=100)