import os

def build_meta_file(cfg):
    root = cfg.train.path
    images_folder_name = 'image_frames'
    image_frames_folder = os.path.join(root,images_folder_name)
    text_file_path = cfg.test.test_list_path

    image_frames = [ os.path.join(images_folder_name,file + '\n')  for file in os.listdir(image_frames_folder)]

    with open(text_file_path, 'w') as file:
        file.writelines(image_frames)
