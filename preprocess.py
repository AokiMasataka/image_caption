import json
import tqdm
import pickle


def make_ant(json_path, prefix):
    with open(json_path, 'r', encoding='UTF-8') as f:
        json_data = json.load(f)

    prefix = prefix + '/'

    images = json_data['images']
    ants = json_data['annotations']

    data = ''

    bar = tqdm.tqdm(ants)
    for ant in bar:
        image_id = ant['image_id']
        for image in images:
            if image['id'] == image_id:
                data += prefix + image['file_name'] + '||' + ant['caption'] + '\n'
                break

    return data.strip()


def label_preprocess(load_json):
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
                train_list[index] = {'file_name': image['file_name'], 'caption': ant['caption']}
                break
    return train_list


def write(export, obj):
    with open(export, 'wb') as f:
        pickle.dump(obj, f)


def main():
    train_data = make_ant('stair_captions/stair_captions_v1.2_train.json', prefix='D:/Dataset/coco_images/train2014')
    with open('stair_captions/stair_captions_train_data.txt', 'w', encoding='UTF-8') as f:
        f.write(train_data)

    train_data = make_ant('stair_captions/stair_captions_v1.2_val.json', prefix='D:/Dataset/coco_images/val2014')
    with open('stair_captions/stair_captions_valid_data.txt', 'w', encoding='UTF-8') as f:
        f.write(train_data)


if __name__ == '__main__':
    main()
