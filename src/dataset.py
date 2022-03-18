import cv2
import csv
import os


def read_dataset(data_dir):
    path_to_boxes = os.path.join(data_dir, 'groundtruth_rect.txt')
    path_to_images = os.path.join(data_dir, 'img')
    images = sorted(os.listdir(path_to_images), key=lambda x: int(x[:-4]))
    boxes = []
    with open(path_to_boxes, encoding='utf8') as file:
        reader = csv.reader(file, delimiter='\t')
        for col1, col2, col3, col4 in reader:
            boxes.append(
                (int(col1), int(col2), int(col3), int(col4))
            )

    for image, box in zip(images, boxes):
        x0, y0, width, height = box
        image = cv2.imread(os.path.join(path_to_images, image))
        yield (image, (x0, y0, width, height))