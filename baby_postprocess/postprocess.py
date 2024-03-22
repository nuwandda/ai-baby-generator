import torch 
from PIL import Image
import numpy as np
import random
import random
import uuid
import os
import subprocess
from io import BytesIO
import base64
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
from diffusers import StableDiffusionPipeline
from deepface import DeepFace


TEMP_PATH = 'temp'
MODEL_PATH = os.getenv('MODEL_PATH')
if MODEL_PATH is None:
    MODEL_PATH = 'baby_postprocess/weights/realisticVisionV60B1_v20Novae.safetensors'
background_prompts = ['in the park', 'in the school', 'in the street', 'in the city', 'in the pool', 'in the bedroom']

def set_seed():
    seed = random.randint(42,4294967295)
    return seed


def create_temp():
    if not os.path.exists(TEMP_PATH):
        os.makedirs(TEMP_PATH)


def remove_temp_image(id, photo_number):
    os.remove(TEMP_PATH + '/' + id + '_' + str(photo_number) + '_generated.png')
    os.remove(TEMP_PATH + '/' + id + '_' + str(photo_number) + '_out.png')


def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))

    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)

    #Remove original file
    remove(file_path)

    #Move new file
    move(abs_path, file_path)


def create_pipeline(model_path):
    # Create the pipe 
    pipe = StableDiffusionPipeline.from_single_file(
        model_path, 
        revision="fp16", 
        torch_dtype=torch.float16
        )
    
    # pipe.load_lora_weights(pretrained_model_name_or_path_or_dict="weights/lora_disney.safetensors", adapter_name="disney")

    if torch.backends.mps.is_available():
        device = "mps"
    else: 
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe.to(device)
    
    return pipe

pipe = create_pipeline(MODEL_PATH)
# Update the paths in submodule
replace("facefusion/facefusion/core.py", "available_frame_processors = list_directory('facefusion/processors/frame/modules')",
        "available_frame_processors = list_directory('facefusion/facefusion/processors/frame/modules')")
replace("facefusion/facefusion/core.py", "available_ui_layouts = list_directory('facefusion/uis/layouts')",
        "available_ui_layouts = list_directory('facefusion/facefusion/uis/layouts')")
replace("facefusion/facefusion/core.py", "available_frame_processors = list_directory('facefusion/processors/frame/modules')",
        "available_frame_processors = list_directory('facefusion/facefusion/processors/frame/modules')")


def generate(image_path, gender, seed, strength, total_number_of_photos, img_height, guidance_scale, num_inference_steps):
    temp_id = str(uuid.uuid4())
    create_temp()

    generator = torch.Generator().manual_seed(set_seed()) if float(seed) == -1 else torch.Generator().manual_seed(int(seed))
    objs = DeepFace.analyze(img_path = image_path, actions = ['race'])
    prompt_gender = 'girl' if gender == 'female' else 'boy'
    
    photos = {}
    for photo_number in range(total_number_of_photos):
        prompt = """photo of a 2 years old {} {}, {}, detailed (blemishes, folds, moles, veins, pores, skin imperfections:1.1),
            highly detailed glossy eyes, specular lighting, dslr, ultra quality, sharp focus, tack sharp, dof, 
            film grain, centered, Fujifilm XT3, crystal clear""".format(objs[0]['dominant_race'], prompt_gender, random.choice(background_prompts))
        negative_prompt = """beard, moustache, naked, nude, out of frame, tattoo, b&w, sepia,
            (blurry un-sharp fuzzy un-detailed skin:1.4), (twins:1.4), (geminis:1.4), (wrong eyeballs:1.1),
            (cloned face:1.1), (perfect skin:1.2), (mutated hands and fingers:1.3), disconnected hands,
            disconnected limbs, amputation, (semi-realistic, cgi, 3d, render, sketch, cartoon, drawing,
            anime, doll, overexposed, photoshop, oversaturated:1.4)"""
        image: Image = pipe(prompt,
                        height=img_height,
                        strength=strength,
                        negative_prompt=negative_prompt, 
                        guidance_scale=guidance_scale, 
                        num_inference_steps=num_inference_steps, 
                        generator = generator,
                        cross_attention_kwargs={"scale": strength}
                        ).images[0]

        if not image.getbbox():
            image: Image = pipe(prompt,
                                height=img_height,
                                strength=strength + 0.1,
                                negative_prompt=negative_prompt,
                                guidance_scale=guidance_scale, 
                                num_inference_steps=num_inference_steps, 
                                generator = generator,
                                cross_attention_kwargs={"scale": strength}
                                ).images[0]
        
        image.save(TEMP_PATH + '/' + temp_id + '_' + str(photo_number) + '_generated.png')

        # Swap the input face with the generated image
        subprocess.call(['python', 'facefusion/run.py', '-s', '{}'.format(image_path), 
                        '-t', '{}'.format(TEMP_PATH + '/' + temp_id + '_' + str(photo_number) + '_generated.png'),
                        '-o', '{}'.format(TEMP_PATH + '/' + temp_id + '_' + str(photo_number) + '_out.png'),
                        '--headless', '--frame-processors', 'face_swapper', 'face_enhancer', '--face-swapper-model',
                        'simswap_512_unofficial'])

        final_image = Image.open(TEMP_PATH + '/' + temp_id + '_' + str(photo_number) + '_out.png')
        buffered = BytesIO()
        final_image.save(buffered, format="JPEG")
        encoded_img = base64.b64encode(buffered.getvalue())
        remove_temp_image(temp_id, photo_number)
        photos[photo_number] = encoded_img
        
    return photos
