import os
import glob

image_directory = '/home/zwei/datasets/PublicEmotion/UnBiasedEmo/images'

image_list = glob.glob(os.path.join(image_directory, '*/*/*'))

image_name_dict = {}

for s_image_path in image_list:
    s_image_name = os.path.basename(s_image_path)
    if s_image_name in image_name_dict:
        image_name_dict[s_image_name].append(s_image_path)
    else:
        image_name_dict[s_image_name] = [s_image_path]

for s_image_name in image_name_dict:
    if len(image_name_dict[s_image_name]) > 1:
        for s_path in image_name_dict[s_image_name]:
            print(s_image_path)

print("DB")