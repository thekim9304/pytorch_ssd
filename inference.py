import cv2
import numpy as np

import torch

from mobilenetv1 import MobileNetV1
from ssd import SSD, Predictor
from utils import box_utils

def main():
    model = SSD(2, MobileNetV1(2), is_training=False)
    state = torch.load('C:/Users/th_k9/Desktop/pytorch_ssd/models/394_0.0001.pth')
    model.load_state_dict(state['model_state_dict'])

    predictor = Predictor(model, 300)

    # img = cv2.imread('C:/Users/th_k9/Desktop/git_pytorch-ssd-master/data/train/353c4209e7467509.jpg')
    img = cv2.imread('E:/DB_WIDERFACE/WIDER_train/images/0--Parade/0_Parade_marchingband_1_799.jpg')

    boxes, labels, probs = predictor.predict(img, 10, 0.4)

    print(boxes.size(0))

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"Face: {probs[i]:.2f}"
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(img, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imshow('annotated', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()