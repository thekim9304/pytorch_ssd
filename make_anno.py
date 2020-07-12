import os
import cv2
import csv

anno_path = 'E:/DB_WIDERFACE/wider_face_split/wider_face_train_bbx_gt.txt'

with open(anno_path) as anno_txts:
    data = anno_txts.read()
annotations = data.split('\n')

f = open('train.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow(['ImageID', 'XMin', 'YMin', 'XMax', 'YMax', 'ClassName'])

anno = 0
while True:
    box_list = []
    print(annotations[anno])
    image_name = annotations[anno]
    boxes = int(annotations[anno+1])

    img = cv2.imread(os.path.join('E:/DB_WIDERFACE/WIDER_train/images', image_name))

    h, w, _ = img.shape

    for i in range(1, boxes+1):
        box = list(map(int, annotations[anno+1+i].split(' ')))[:4]
        wr.writerow([image_name[:-4], box[0]/w, box[1]/h, (box[0]+box[2])/w, (box[1]+box[3])/h, 'Face'])
        print([image_name[:-4], box[0]/w, box[1]/h, (box[0]+box[2])/w, (box[1]+box[3])/h, 'Face'])

    if boxes == 0:
        boxes = 1
    anno += (boxes + 2)

    if anno + 1 >= len(annotations):
        break

f.close()