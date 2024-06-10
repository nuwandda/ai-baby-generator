import schemas as _schemas
import os
# from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import numpy as np
import uuid
import io
from pkg_resources import parse_version
import cv2
import base64
from preprocess import preprocess
from baby_process.baby_process import send_request
from baby_postprocess.postprocess import generate
from hair_color_detection import get_hair_color


TEMP_PATH = 'temp'

# Helper functions
def create_temp():
    if not os.path.exists(TEMP_PATH):
        os.makedirs(TEMP_PATH)


def remove_temp_image(id):
    os.remove(TEMP_PATH + '/' + id + '_child.png')


def image_to_base64(pil_image):
    # Create a BytesIO object to temporarily hold the image data
    img_byte_array = io.BytesIO()

    # Save the PIL image to the BytesIO object
    pil_image.save(img_byte_array, format='PNG')

    # Convert the BytesIO object to bytes
    img_byte_array = img_byte_array.getvalue()

    # Encode the bytes to base64
    base64_encoded_image = base64.b64encode(img_byte_array)

    # Convert bytes to a UTF-8 string
    base64_string = base64_encoded_image.decode('utf-8')

    return base64_string


async def generate_image(babyCreate: _schemas.BabyCreate) -> Image:
    temp_id = str(uuid.uuid4())
    create_temp()

    """
        ---PREPROCESS---
        Preprocess the input images to find the best father and mother images 
    """
    # Find the best father image
    current_father_score = 100
    best_father_image = None
    for encoded_image in babyCreate.encoded_dad_imgs:
        init_image = Image.open(BytesIO(base64.b64decode(encoded_image)))
        aspect_ratio = init_image.width / init_image.height
        target_height = round(babyCreate.img_height / aspect_ratio)

        # Resize the image
        if parse_version(Image.__version__) >= parse_version('9.5.0'):
            resized_image = init_image.resize((babyCreate.img_height, target_height), Image.LANCZOS)
        else:
            resized_image = init_image.resize((babyCreate.img_height, target_height), Image.ANTIALIAS)

        _, score = preprocess.preprocess(np.array(resized_image), babyCreate.focal_length)
        if score < current_father_score:
            current_father_score = score
            best_father_image = resized_image

    # Find the best mother image
    current_mother_score = 100
    best_mother_image = None
    for encoded_image in babyCreate.encoded_mom_imgs:
        init_image = Image.open(BytesIO(base64.b64decode(encoded_image)))
        aspect_ratio = init_image.width / init_image.height
        target_height = round(babyCreate.img_height / aspect_ratio)

        # Resize the image
        if parse_version(Image.__version__) >= parse_version('9.5.0'):
            resized_image = init_image.resize((babyCreate.img_height, target_height), Image.LANCZOS)
        else:
            resized_image = init_image.resize((babyCreate.img_height, target_height), Image.ANTIALIAS)

        _, score = preprocess.preprocess(np.array(resized_image), babyCreate.focal_length)
        if score < current_mother_score:
            current_mother_score = score
            best_mother_image = resized_image

    """
        ---PROCESS---
        Generate the baby using the best images
    """
    father_encoded_img = image_to_base64(best_father_image)
    mother_encoded_img = image_to_base64(best_mother_image)
    
    child = send_request(str(father_encoded_img), str(mother_encoded_img), token=babyCreate.token,
                 gender=babyCreate.gender, power_of_dad=str(babyCreate.power_of_dad), ethnicity=babyCreate.ethnicity)
    
    child.save(TEMP_PATH + '/' + temp_id + '_child.png')

    # Opens a image in RGB mode
    # child = Image.open('baby_test2.png')
    
    # Size of the image in pixels (size of original image)
    # (This is not mandatory)
    width, height = child.size
    
    # Setting the points for cropped image
    left = 270
    top = 0
    right = 720
    bottom = 150
    
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = child.crop((left, top, right, bottom))
    im1.save(TEMP_PATH + '/' + temp_id + '_hair.png')
    hair_color = get_hair_color(TEMP_PATH + '/' + temp_id + '_hair.png')

    """
        ---POSTPROCESS---
        Generate realistic baby picture using face swap and Stable Diffusion
    """
    return_images = generate(TEMP_PATH + '/' + temp_id + '_child.png', babyCreate.gender, babyCreate.total_number_of_photos, hair_color)

    # remove_temp_image(temp_id)
    return return_images
        