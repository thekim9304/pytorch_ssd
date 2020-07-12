from mobilenetv1 import MobileNetV1

import torch

import time

x = torch.randn(1, 3, 300, 300).cuda()

mv1 = MobileNetV1().model.cuda().eval()

if __name__=='__main__':

    for _ in range(100):
        prev_time = time.time()
        d = x
        for layer in mv1:
            d = layer(d)
        print(time.time() - prev_time)

        prev_time = time.time()
        mv1(x)
        print(time.time() - prev_time)

        print()