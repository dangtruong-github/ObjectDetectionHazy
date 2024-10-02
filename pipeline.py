from YOLOv8.utils import (
    GetYOLO, TrainYOLO, EvaluateYOLO, InferenceYOLO
)
from DehazeModel.utils import (
    GetFFA, TrainFFA, EvaluateFFA, InferenceFFA
)


def TrainPipeline(mode="ffa"):
    if mode == "ffa":
        TrainFFA()
    
    TrainYOLO()


def EvaluatePipeline(mode="ffa"):
    if mode == "ffa":
        EvaluateFFA()
    
    EvaluateYOLO()


def InferencePipeline(img, model_path=None, mode="ffa"):
    img_dehaze = None
    if mode == "ffa":
        img_dehaze = InferenceFFA(img)
    
    obj_res = InferenceYOLO(img_dehaze)

    
