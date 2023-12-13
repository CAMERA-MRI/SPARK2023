from ultralytics import YOLO
print("YOLO_Z_EVAL")
# Load a model
model = YOLO('/scratch/guest190/models/yolo/train_aug/weights/best.pt')  # load a custom model

# Validate the model
metrics = model.val(data="data_eval.yaml", device=[0,1,2,3])     # no arguments needed, dataset and settings remembered

print("metrics.box.map", metrics.box.map)       # map50-95
print("metrics.box.map50", metrics.box.map50)    # map50
print("metrics.box.map75", metrics.box.map75)    # map75
print("metrics.box.maps", metrics.box.maps)      # a list contains map50-95 of each category
