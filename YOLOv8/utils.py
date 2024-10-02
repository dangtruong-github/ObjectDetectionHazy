from ultralytics import YOLO
import torch
import torchvision.transforms as tfs


def GetYOLO(path=None):
    # Load a pretrained YOLOv8 model
    if path == None:
        path = "yolov8n.pt"
    model = YOLO(path)
    return model


def TrainYOLO(model_path=None, save_path=None):
    model = GetYOLO(model_path)

    train_results = model.train(
        data="coco8.yaml",  # path to dataset YAML
        epochs=100,  # number of training epochs
        imgsz=640,  # training image size
        device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or 
    )

    model.export(path=save_path)

    return train_results


def EvaluateYOLO(model_path=None):
    model = GetYOLO(model_path)

    metrics = model.val()

    return metrics


def InferenceYOLO(img, model_path=None):
    model = GetYOLO(model_path)

    mean = torch.tensor([0.64, 0.6, 0.58])
    std = torch.tensor([0.14, 0.15, 0.152])

    transform = tfs.Compose([
        tfs.Normalize(mean, std)
    ])

    with torch.no_grad():
        bb = model(transform(img))

    return bb[0]
