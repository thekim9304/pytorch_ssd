import os
import math
import datetime

import torch
from torch.utils.data import DataLoader

from ssd import *
from mobilenetv1 import MobileNetV1
from dataloader import CustomDataset
import mobilenetv1_ssd_config as cfg

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(data_loader, model, criterion, optimizer, device, debug_steps=100, epochs=-1):
    model.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0

    for i, data in enumerate(data_loader):
        name, images, boxes, labels = data

        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        confidence, locations = model(images)

        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)

        if math.isinf(regression_loss.item()):
            print(name)
            print('='*100)

        loss = regression_loss + classification_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

    loss = running_loss / len(data_loader)
    reg_loss = running_regression_loss / len(data_loader)
    class_loss = running_classification_loss / len(data_loader)

    return loss, reg_loss, class_loss

def main():
    root_path = 'E:/DB_WIDERFACE/WIDER_train'
    anno_name = 'train'
    dataset_type = 'images'
    model_root = 'E:/models'

    # root_path = 'C:/Users/th_k9/Desktop/git_pytorch-ssd-master/data'
    # anno_name = 'sub-train-annotations-bbox'
    # dataset_type = 'train'
    img_size = 300

    dataset = CustomDataset(root_path, anno_name, dataset_type, img_size,
                            MatchPrior(cfg.priors,
                                       cfg.center_variance,
                                       cfg.size_variance, iou_threshold=0.5))
    print("Train dataset size: {}".format(len(dataset)))
    train_loader = DataLoader(dataset,
                              batch_size=8,
                              num_workers=0,
                              shuffle=False)
    class_num =len(dataset.class_names)

    backbone = MobileNetV1(class_num)
    model = SSD(num_classes=class_num, backbone=backbone, device=DEVICE)
    model.to(DEVICE)

    criterion = SSDLoss(cfg.priors, iou_threshold=0.5, neg_pos_ratio=3,
                    center_variance=0.1, size_variance=0.2, device=DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10000
    min_loss = 0.3
    for epoch in range(epochs):
        print(f'{epoch} epoch start! : {datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")}')

        loss, reg_loss, class_loss = train(train_loader, model, criterion, optimizer, DEVICE, 10, epoch)
        print(f"  Epoch: {epoch}, " +
              f"Average Loss: {loss:.4f}, " +
              f"Average Regression Loss {reg_loss:.4f}, " +
              f"Average Classification Loss: {class_loss:.4f}")

        if min_loss > loss:
            min_loss = loss
            state = {
                'Epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }
            model_path = os.path.join(model_root, f'ssd-mobilev1-face-{epoch}_{loss:.4f}.pth')
            torch.save(state, model_path)
            print(f'Saved model _ [loss : {loss:.4f}, save_path : {model_path}\n')

        if loss < 0.0001:
            break

if __name__=='__main__':
    main()