from ultralytics import YOLO
# import matplotlib.pyplot as plt


# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='data.yaml', epochs=200, verbose=True, cache=True,
            imgsz=256, pretrained = False, device= [0,1,2,3], translate = 0.2, 
            degrees = 0.2, perspective= 0.1, scale= 0.4, flipud= 0.3, 
            fliplr= 0.3, mosaic = 0.0, mixup= 0.0, 
            hsv_h=0.1, hsv_s=0.1, hsv_v=0.1,
            name="/scratch/guest190/models/yolo/train_aug")