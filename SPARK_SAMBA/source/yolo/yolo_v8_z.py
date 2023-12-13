from ultralytics import YOLO
# import matplotlib.pyplot as plt


# Load a model
model = YOLO('yolov8l.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='data_z.yaml', epochs=200, verbose=True, cache=True,
            imgsz=256, pretrained = True, device= [0,1,2,3], translate = 0.2, 
            degrees = 0.2, perspective= 0.0, scale= 0.4, flipud= 0.4, 
            fliplr= 0.4, mosaic = 0.0, mixup= 0.0, 
            hsv_h=0.2, hsv_s=0.2, hsv_v=0.2,
            name="/scratch/guest190/models/yolo/train_z")