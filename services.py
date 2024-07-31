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
from file_operations import download_folder_from_s3


TEMP_PATH = 'temp'

# Helper functions
def create_temp():
    """
    Creates a temporary directory if it does not exist.

    This function checks if the directory specified by the `TEMP_PATH` constant exists.
    If the directory does not exist, it creates the directory using the `os.makedirs()` function.

    Parameters:
    None

    Returns:
    None
    """
    if not os.path.exists(TEMP_PATH):
        os.makedirs(TEMP_PATH)


def remove_temp_image(id: str) -> None:
    """
    Removes temporary images from the specified temporary directory.

    Parameters:
    id (str): The unique identifier for the baby images. This identifier is used to construct the file names of the temporary images.

    Returns:
    None: This function does not return any value. It only removes the temporary images.
    """
    os.remove(TEMP_PATH + '/' + id + '_child.png')


def image_to_base64(pil_image):
    """
    Converts a PIL image to a base64-encoded string.

    This function takes a PIL image as input, converts it to bytes, encodes the bytes to base64,
    and finally decodes the base64 bytes to a UTF-8 string. The resulting string can be used
    in various contexts, such as sending images over a network or storing them in a database.

    Parameters:
    pil_image (PIL.Image): The input PIL image to be converted to base64.

    Returns:
    str: A base64-encoded string representation of the input PIL image.
    """
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
    """
    This function generates a baby image based on the input parameters. It performs preprocessing,
    image generation, and postprocessing steps to produce a realistic baby image.

    Parameters:
    babyCreate (_schemas.BabyCreate): An object containing the necessary information for generating the baby image.
        This object includes attributes such as encoded_dad_imgs, encoded_mom_imgs, img_height, focal_length, token,
        gender, power_of_dad, ethnicity, total_number_of_photos, and skin_tone.

    Returns:
    Image: The generated baby image.
    """
    temp_id = str(uuid.uuid4())
    create_temp()
    message = 'Successfully generated baby image.'

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
        # We check if the face is not proper in the line below and continue with the next one
        if score == -1:
            continue

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
        # We check if the face is not proper in the line below and continue with the next one
        if score == -1:
            continue

        if score < current_mother_score:
            current_mother_score = score
            best_mother_image = resized_image

    """
        ---PROCESS---
        Generate the baby using the best images
    """
    # Returns None and message if one of the best images are missing
    if best_father_image is None or best_mother_image is None:
        return None, "One of the parent images is missing!"
     
    father_encoded_img = image_to_base64(best_father_image)
    mother_encoded_img = image_to_base64(best_mother_image)
    
    child = send_request(str(father_encoded_img), str(mother_encoded_img), token=babyCreate.token,
                 gender=babyCreate.gender, power_of_dad=str(babyCreate.power_of_dad), ethnicity=babyCreate.ethnicity)
    
    child.save(TEMP_PATH + '/' + temp_id + '_child.png')

    """
        ---POSTPROCESS---
        Generate realistic baby picture using face swap and Stable Diffusion
    """
    return_images = generate(TEMP_PATH + '/' + temp_id + '_child.png', temp_id, babyCreate.gender, babyCreate.total_number_of_photos, babyCreate.skin_tone)

    remove_temp_image(temp_id)
    return return_images, message
        