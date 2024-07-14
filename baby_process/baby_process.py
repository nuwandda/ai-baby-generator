import requests
from PIL import Image
import io
import base64


def base64_to_image(base64_string):
    # Decode base64 string to bytes
    img_bytes = base64.b64decode(base64_string)

    # Create a BytesIO object to wrap the bytes
    img_byte_io = io.BytesIO(img_bytes)

    # Open the image using PIL
    img = Image.open(img_byte_io)

    return img


def send_request(father_image, mother_image, token='1230pol>EUe208tq', gender='female', power_of_dad='50',
                 ethnicity='unknown'):
    service = {
        'prod': True,
        'api':'http://172.210.31.221/predict_v3', 
        'headers': {'accept': 'application/json',
            'access-token': token,
            'Content-type': 'application/json'
        },
        'params':{
            'id': '312dsa',
        },
        'json_data' : {
            'automatic': False,
            'gender': gender,
            'powerOfDad': power_of_dad,
            'ethnicity': ethnicity,
            'mum_encoded_img': mother_image,
            'dad_encoded_img': father_image
        }
    }

    response = requests.post(service['api'], params=service['params'], headers=service['headers'], json=service['json_data'], timeout=40)
    if response.status_code == 200:
        print('Child server returned successfully')
    else:
        print('Child server is not responding')
    data = response.json()
    child_img =  base64_to_image(data['data']['baby_encoded_img'])

    return child_img
