import torch 
from PIL import Image
import numpy as np
from pkg_resources import parse_version
import random
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
import cv2
from insightface.app import FaceAnalysis
from instantid.pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps


def set_seed():
    seed = random.randint(42,4294967295)
    return seed


def create_pipeline(face_adapter, controlnet_path):
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    # Create the pipe 
    base_model = 'SG161222/Realistic_Vision_V6.0_B1_noVAE'
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=torch.float16
    )

    if torch.backends.mps.is_available():
        device = "mps"
    else: 
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe.to(device)
    # load adapter
    pipe.load_ip_adapter_instantid(face_adapter)
    
    return pipe


app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# prepare models under ./checkpoints
face_adapter = f'./checkpoints/ip-adapter.bin'
controlnet_path = f'./checkpoints/ControlNetModel'
pipe = create_pipeline(face_adapter, controlnet_path)


def postprocess(image_path, gender, race, seed, strength, total_number_of_photos):
    generator = torch.Generator().manual_seed(set_seed()) if float(seed) == -1 else torch.Generator().manual_seed(int(seed))
    prompt = """photo of a 2 years old {} {}, detailed (blemishes, folds, moles, veins, pores, skin imperfections:1.1),
        highly detailed glossy eyes, specular lighting, dslr, ultra quality, sharp focus, tack sharp, dof, 
        film grain, centered, Fujifilm XT3, crystal clear""".format(race, gender)
    negative_prompt = """beard, moustache, naked, nude, out of frame, tattoo, b&w, sepia,
        (blurry un-sharp fuzzy un-detailed skin:1.4), (twins:1.4), (geminis:1.4), (wrong eyeballs:1.1),
        (cloned face:1.1), (perfect skin:1.2), (mutated hands and fingers:1.3), disconnected hands,
        disconnected limbs, amputation, (semi-realistic, cgi, 3d, render, sketch, cartoon, drawing,
        anime, doll, overexposed, photoshop, oversaturated:1.4)"""
    
    # Load the image
    face_image = load_image(image_path)

    # Prepare face emb
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
    face_emb = face_info['embedding']
    face_kps = draw_kps(face_image, face_info['kps'])
    
    photos = []
    for _ in total_number_of_photos:
        image: Image = pipe(prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=0.8,
            generator = generator,
            ip_adapter_scale=0.8
        ).images[0]

        if not image.getbbox():
            image: Image = pipe(prompt,
                negative_prompt=negative_prompt,
                strength=strength + 0.1,
                image_embeds=face_emb,
                image=face_kps,
                controlnet_conditioning_scale=0.8,
                generator = generator,
                ip_adapter_scale=0.8
            ).images[0]
            
        photos.append(image)
        
    return photos
