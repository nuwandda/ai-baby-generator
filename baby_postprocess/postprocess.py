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
from deepface import DeepFace
from os import listdir
from os.path import isfile, join


TEMP_PATH = 'temp'
MODEL_PATH = os.getenv('MODEL_PATH')
if MODEL_PATH is None:
    MODEL_PATH = 'baby_postprocess/weights/realisticVisionV60B1_v20Novae.safetensors'
background_prompts = ['park', 'school', 'street', 'amusement']


def set_seed():
    seed = random.randint(42,4294967295)
    return seed


def create_temp():
    if not os.path.exists(TEMP_PATH):
        os.makedirs(TEMP_PATH)


def remove_temp_image(id, photo_number):
    os.remove(TEMP_PATH + '/' + id + '_' + str(photo_number) + '_out.png')
    os.remove(TEMP_PATH + '/' + id + '_child.png')
    os.remove(TEMP_PATH + '/' + id + '_hair.png')


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


# Update the paths in submodule
replace("facefusion/facefusion/core.py", "available_frame_processors = list_directory('facefusion/processors/frame/modules')",
        "available_frame_processors = list_directory('facefusion/facefusion/processors/frame/modules')")
replace("facefusion/facefusion/core.py", "available_ui_layouts = list_directory('facefusion/uis/layouts')",
        "available_ui_layouts = list_directory('facefusion/facefusion/uis/layouts')")
replace("facefusion/facefusion/core.py", "available_frame_processors = list_directory('facefusion/processors/frame/modules')",
        "available_frame_processors = list_directory('facefusion/facefusion/processors/frame/modules')")


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
    
    photos = {}
    for photo_number in range(total_number_of_photos):
        theme = random.choice(background_prompts)
        reference_path = 'reference_photos/{}/{}/{}/{}'.format(prompt_gender, race, hair_color, theme)
        onlyfiles = [f for f in listdir(reference_path) if isfile(join(reference_path, f))]
        reference_image_name = random.choice(onlyfiles)
        reference_image_path = reference_path + '/' + reference_image_name
        print(reference_image_path)

        # Swap the input face with the generated image
        subprocess.call(['python', 'facefusion/run.py', '-s', '{}'.format(image_path), 
                        '-t', '{}'.format(reference_image_path),
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
