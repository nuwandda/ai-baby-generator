from stylegene.preprocess.align_images import align_face
import cv2
from stylegene.models.stylegene.api import synthesize_descendant


def run(father, mother, gamma=0.47, eta=0.4, age='0-2', gender='female', dim_to_edit=0, 
        increase_value=0.5, ethnicity='Automatic', power_of_dad=0.5):
    attributes = {'age': age, 'gender': gender, 'gamma': float(gamma), 'eta': float(eta),
                  'dim_to_edit': dim_to_edit, 'increase_value': increase_value, 'ethnicity': ethnicity, 'power_of_dad': power_of_dad}
    img_F, img_M, img_C = synthesize_descendant(father, mother, attributes)
    return img_F, img_M, img_C


def process(father_image, mother_image, father_save_path, mother_save_path, 
            gamma: float = 0.47, eta: float = 0.4, gender: str = 'female',
            dim_to_edit: int = 0, increase_value: float = 0.5, ethnicity: str = 'Automatic', power_of_dad: float = 0.5):
    try:
        father_img = align_face(father_image)
        cv2.imwrite(father_save_path, father_img)
        mother_img = align_face(mother_image)
        cv2.imwrite(mother_save_path, mother_img)
    except Exception:
        return {"message": "There was an error aligning the images."}
    
    child = None
    if ethnicity != 'Automatic': 
        _, _, child = run(father_save_path,
                          mother_save_path,
                          gamma=1, eta=1, gender=gender, 
                          dim_to_edit=dim_to_edit, increase_value=increase_value, 
                          ethnicity=ethnicity, power_of_dad=power_of_dad)
    else:
        _, _, child = run(father_save_path,
                          mother_save_path,
                          gamma=gamma, eta=eta, gender=gender, 
                          dim_to_edit=dim_to_edit, increase_value=increase_value, 
                          ethnicity=ethnicity, power_of_dad=power_of_dad)
        
    return child
