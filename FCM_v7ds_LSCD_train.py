# -*- coding: utf-8 -*-
import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11-FCM_v7ds_LSCD.yaml') # YOLO11
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0, 
                workers=4, 
                optimizer='SGD',
                project='results',
                name='YOLOv11_FCM_v7ds_LSCD',
                )
               