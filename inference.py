import cv2
import time
import numpy as np

import torch

from mobilenetv1 import MobileNetV1
from ssd import SSD, Predictor
from utils import box_utils

def infer_frame():
    model = SSD(2, MobileNetV1(2), is_training=False)
    state = torch.load('C:/Users/th_k9/Desktop/pytorch_ssd/models/ssd-mobilev1-face-2134_0.0192.pth')
    model.load_state_dict(state['model_state_dict'])

    predictor = Predictor(model, 300)

    # img = cv2.imread('C:/Users/th_k9/Desktop/git_pytorch-ssd-master/data/train/353c4209e7467509.jpg')
    # img = cv2.imread('E:/DB_WIDERFACE/WIDER_train/images/0--Parade/0_Parade_marchingband_1_799.jpg')
    img = cv2.imread('../KakaoTalk_20200617_193006623.jpg')

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

def infer_cam():
    model = SSD(2, MobileNetV1(2), is_training=False)
    state = torch.load('C:/Users/th_k9/Desktop/pytorch_ssd/models/ssd-mobilev1-face-2134_0.0192.pth')
    model.load_state_dict(state['model_state_dict'])

    predictor = Predictor(model, 300)

    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # out = cv2.VideoWriter('C:/Users/th_k9/Desktop/t.avi', fourcc, 30.0, (640, 480))
    # out2 = cv2.VideoWriter('C:/Users/th_k9/Desktop/t_f.avi', fourcc, 30.0, (300, 300))

    frm_cnt = 1
    fps = 0
    fps_res = 0
    while True:
        ret, img = cap.read()

        if ret:
            prevTime = time.time()
            boxes, labels, probs = predictor.predict(img, 1, 0.5)
            curTime = time.time()
            sec = curTime - prevTime

            if (1/sec) > 30:
                fps += 1/sec
                fps_res = fps / frm_cnt
                frm_cnt += 1

            cv2.putText(img, f'{fps_res:.2f}',
                        (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (0, 0, 255),
                        2)  # line type

            for i in range(boxes.size(0)):
                box = boxes[i, :]
                label = f"Face: {probs[i]:.2f}"

                face = img[int(box[1].item()):int(box[3].item()), int(box[0].item()):int(box[2].item())].copy()

                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 128), 4)
                cv2.putText(img, label,
                            (box[0], box[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,  # font scale
                            (128, 0, 255),
                            2)  # line type

            white_numpy = np.full((480, 300, 3), 255, dtype=np.uint8)

            if boxes.size(0):
                white_numpy[:300, :] = cv2.resize(face, (300, 300))
                cv2.imshow('face', cv2.resize(face, (40, 40)))

            concat_img = cv2.hconcat([white_numpy, img])
            cv2.imshow('annotated', concat_img)

            if cv2.waitKey(1) == 27:
                break
        else:
            break
    cv2.destroyAllWindows()
    cap.release()
    # out.release()
    # out2.release()

if __name__=='__main__':
    # infer_frame()
    infer_cam()