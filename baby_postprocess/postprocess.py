from PIL import Image
import random
import uuid
import os
import subprocess
from io import BytesIO
import base64
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
from os import listdir
from os.path import isfile, join

import cv2
from insightface.app import FaceAnalysis
import torch
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID, IPAdapterFaceIDXL


TEMP_PATH = 'temp'
background_prompts = ['park', 'school', 'street', 'amusement']
base_model_path = "SG161222/RealVisXL_V4.0"
vae_model_path = "stabilityai/sd-vae-ft-mse"
ip_ckpt = "weights/ip-adapter-faceid_sdxl.bin"

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.2)


def set_seed():
    seed = random.randint(42,4294967295)
    return seed


def create_temp():
    if not os.path.exists(TEMP_PATH):
        os.makedirs(TEMP_PATH)


def remove_temp_image(id, photo_number):
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


def create_pipe(device='cuda'):
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        add_watermarker=False
    )

    ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, device)

    return ip_model


ip_model = create_pipe()


# Update the paths in submodule
# replace("ip_adapter/ip_adapter/ip_adapter_faceid.py", "available_frame_processors = list_directory('facefusion/processors/frame/modules')",
#         "available_frame_processors = list_directory('facefusion/facefusion/processors/frame/modules')")
# replace("facefusion/facefusion/core.py", "available_ui_layouts = list_directory('facefusion/uis/layouts')",
#         "available_ui_layouts = list_directory('facefusion/facefusion/uis/layouts')")
# replace("facefusion/facefusion/core.py", "available_frame_processors = list_directory('facefusion/processors/frame/modules')",
#         "available_frame_processors = list_directory('facefusion/facefusion/processors/frame/modules')")


def generate(image_path, temp_id, gender, total_number_of_photos, hair_color, ethnicity):
    race = ''
    if ethnicity == 'light' or ethnicity == 'fair':
        race = 'white'
    elif ethnicity == 'tan':
        race = 'asian'
    elif ethnicity == 'brown':
        race = 'latino'
    else:
        race = 'black'

    prompt_gender = 'girl' if gender == 'female' else 'boy'
    theme = random.choice(background_prompts)

    image = cv2.imread(image_path)
    faces = app.get(image)
    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    
    photos = {}
    prompt = "centered, portrait photo of a 2 years old {} {}, natural skin, dark shot, in the {}".format(race, prompt_gender, theme)
    negative_prompt = """
        (octane render, render, drawing, anime, bad photo, bad photography:1.3), 
        (worst quality, low quality, blurry:1.2), (bad teeth, deformed teeth, deformed lips), 
        (bad anatomy, bad proportions:1.1), (deformed iris, deformed pupils), (deformed eyes, bad eyes), 
        (deformed face, ugly face, bad face), (deformed hands, bad hands, fused fingers), 
        morbid, mutilated, mutation, disfigured
    """

    images = ip_model.generate(prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, 
                               guidance_scale=7.5, num_samples=total_number_of_photos, 
                               width=1024, height=1024, num_inference_steps=30)
    
    for photo_number in range(total_number_of_photos):
        generated_image = images[photo_number]
        buffered = BytesIO()
        generated_image.save(buffered, format="JPEG")
        encoded_img = base64.b64encode(buffered.getvalue())
        photos[photo_number] = encoded_img

    # for photo_number in range(total_number_of_photos):
    #     theme = random.choice(background_prompts)
    #     reference_path = 'reference_photos/{}/{}/{}/{}'.format(prompt_gender, race, hair_color, theme)
    #     onlyfiles = [f for f in listdir(reference_path) if isfile(join(reference_path, f))]
    #     reference_image_name = random.choice(onlyfiles)
    #     reference_image_path = reference_path + '/' + reference_image_name
    #     print(reference_image_path)

    #     # Swap the input face with the generated image
    #     subprocess.call(['python', 'run.py', '-s', '{}'.format(image_path), 
    #                     '-t', '{}'.format(reference_image_path),
    #                     '-o', '{}'.format(TEMP_PATH + '/' + temp_id + '_' + str(photo_number) + '_out.png'),
    #                     '--headless', '--frame-processors', 'face_swapper', 'face_enhancer', '--face-swapper-model',
    #                     'simswap_256'])

    #     final_image = Image.open(TEMP_PATH + '/' + temp_id + '_' + str(photo_number) + '_out.png')
    #     buffered = BytesIO()
    #     final_image.save(buffered, format="JPEG")
    #     encoded_img = base64.b64encode(buffered.getvalue())
    #     remove_temp_image(temp_id, photo_number)
    #     photos[photo_number] = encoded_img

    # os.remove(TEMP_PATH + '/' + temp_id + '_child.png')
    # os.remove(TEMP_PATH + '/' + temp_id + '_hair.png')
        
    return photos
