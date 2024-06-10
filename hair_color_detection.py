from colorthief import ColorThief
import numpy as np


reference_colors = [[233,227,208], [109,83,69], [119,69,44], [18,12,12]]
reference_hair_names = ['blonde', 'auburn', 'ginger', 'brunette']


def get_hair_color(image_path):
    color_thief = ColorThief(image_path)
    dominant_color = color_thief.get_color(quality=1)
    colors = np.array(reference_colors)
    color = np.array(dominant_color)
    distances = np.sqrt(np.sum((colors-color)**2,axis=1))
    index_of_smallest = np.where(distances==np.amin(distances))
    hair_color = reference_hair_names[index_of_smallest[0][0]]

    return hair_color 

# closest_color = get_hair_color(dominant_color)
# print(closest_color)