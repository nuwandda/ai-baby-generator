import os
import cv2
import sys
import dlib
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import preprocess.reference_world as world


PREDICTOR_PATH = 'preprocess/weights/shape_predictor_68_face_landmarks.dat'
if not os.path.isfile(PREDICTOR_PATH):
    print("[ERROR] Download the predictor!")
    sys.exit()


def print_exif_data(exif_data):
    """
    This function takes in an exif_data dictionary and prints out the focal length of the image.

    Args:
        exif_data (dict): The exif data of the image

    Returns:
        float: The focal length of the image

    """
    for tag_id in exif_data:
        tag = TAGS.get(tag_id, tag_id)
        content = exif_data.get(tag_id)
        if tag == 'FocalLength':
            return content
        
    
def get_focal_length(im: Image) -> float:
    """
    This function takes in an image and returns the focal length of the image.

    Args:
        im (Image): The image for which the focal length is to be calculated.

    Returns:
        float: The focal length of the image.

    """
    exif = im.getexif()
    focal = print_exif_data(exif.get_ifd(0x8769))
    return focal


def preprocess(im: np.ndarray, focal: float) -> str:
    """
    This function takes in an image and a focal length and returns the gaze direction of the person in the image.

    Args:
        im (np.ndarray): The image of the person
        focal (float): The focal length of the camera

    Returns:
        str: The gaze direction of the person

    """
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(PREDICTOR_PATH)

        faces = detector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 0)
        face3Dmodel = world.ref3DModel()

        for face in faces:
            shape = predictor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), face)
            refImgPts = world.ref2dImagePoints(shape)

            height, width, channel = im.shape
            focalLength = focal * width
            cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))

            mdists = np.zeros((4, 1), dtype=np.float64)

            # Calculate rotation and translation vector using solvePnP
            success, rotationVector, translationVector = cv2.solvePnP(
                face3Dmodel, refImgPts, cameraMatrix, mdists)

            noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            noseEndPoint2D, jacobian = cv2.projectPoints(
                noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

            # Draw nose line 
            p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
            p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))

            # Calculating angle
            rmat, jac = cv2.Rodrigues(rotationVector)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            x = np.arctan2(Qx[2][1], Qx[2][2])
            y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
            z = np.arctan2(Qz[0][0], Qz[1][0])

            gaze = "Looking: "
            if angles[1] < -15:
                gaze += "Left"
            elif angles[1] > 15:
                gaze += "Right"
            else:
                gaze += "Forward"

        return gaze, angles[1]
    except Exception as e:
        print(f"[ERROR] {e}")
        return False, -1
