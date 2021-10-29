import os
import json
import tqdm
import pickle
from torchvision.transforms import Resize
from torchvision.io import read_image, write_png
from torchvision.io.image import ImageReadMode


def label_preprocess(load_json, export):
    with open(load_json, 'r', encoding='UTF-8') as f:
        train_json = json.load(f)

    images = train_json['images']
    ants = train_json['annotations']

    train_list = [None for _ in range(ants.__len__())]
    bar = tqdm.tqdm(ants)
    for index, ant in enumerate(bar):
        image_id = ant['image_id']
        for image in images:
            if image['id'] == image_id:
                train_list[index] = {'file_name': image['file_name'], 'stair_captions': ant['stair_captions']}
                break

    with open(export, 'wb') as f:
        pickle.dump(train_list, f)


def resize_images(target_dir, export_dir):
    resize = Resize((160, 160))
    image_files = os.listdir(target_dir)
    for image_file in image_files:
        image = read_image(target_dir + image_file, mode=ImageReadMode.RGB)
        image = resize(image)
        write_png(image, export_dir + image_file)


if __name__ == '__main__':
    resize_images(
        target_dir='D:/data_set/COCO_images/train2014/',
        export_dir='D:/data_set/COCO_images/resized_train2014/'
    )
