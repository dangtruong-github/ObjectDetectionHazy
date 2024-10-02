import torch

from FFANet import FFA

device = "cuda" if torch.cuda.is_available() else "cpu"


def GetFFA(path=None):
    model = FFA(gps=3, blocks=19)

    if path is not None:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    return model


def TrainFFA(model_path=None):
    pass


def EvaluateFFA(model_path=None):
    pass


def InferenceFFA(img, model_path=None):
    model = GetFFA(model_path)

    return model(img)