from pipeline import (
    TrainPipeline, EvaluatePipeline, InferencePipeline
)

import cv2

if __name__ == "__main__":
    img = cv2.imread("./DatasetSample/RTTS/")
    InferencePipeline()