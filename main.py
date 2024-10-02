from pipeline import (
    TrainPipeline, EvaluatePipeline, InferencePipeline
)

import cv2
import numpy as np

if __name__ == "__main__":
    img = cv2.imread("./DatasetSample/RTTS/images/YT_Bing_739.png")
    InferencePipeline(img)
