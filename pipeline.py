import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torchvision.transforms as tfs

from YOLOv8.utils import TrainYOLO, EvaluateYOLO, InferenceYOLO
from DehazeModel.utils import TrainFFA, EvaluateFFA, InferenceFFA


def TrainPipeline(mode="ffa"):
    if mode == "ffa":
        TrainFFA()
    
    TrainYOLO()


def EvaluatePipeline(mode="ffa"):
    if mode == "ffa":
        EvaluateFFA()
    
    EvaluateYOLO()


def Preprocess(img):
    img = img.astype(np.float32)
    img /= 255
    transform = tfs.Compose([
        tfs.ToTensor(),
        tfs.Resize((640, 640))
    ])

    return transform(img).unsqueeze(0)

def InferencePipeline(img, model_path=None, mode="ffa"):
    img = Preprocess(img)
    img_dehaze = img
    if mode == "ffa":
        img_dehaze = InferenceFFA(img)
    
    # Object detection inference
    obj_res = InferenceYOLO(img_dehaze)  # Assume it returns a list of bounding boxes [x_min, y_min, x_max, y_max]
    obj_res_orig = InferenceYOLO(img)

    img_dehaze = np.moveaxis(img_dehaze.squeeze().numpy(), 0, 2)
    img = np.moveaxis(img.squeeze().numpy(), 0, 2)
    
    # Plot and save the original and dehazed/predicted images side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show original image
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    # Show dehazed or predicted result
    ax[1].imshow(img_dehaze)
    ax[1].set_title('Dehazed Image with Bounding Boxes')
    ax[1].axis('off')

    
    # Plot bounding boxes on the second image
    for idx, box in enumerate(obj_res_orig.boxes.xyxy):  # obj_res should be a list of [x_min, y_min, x_max, y_max] for each bounding box
        x_min, y_min, x_max, y_max = box
        
        # Create a Rectangle patch
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
        
        # Add the patch to the Axes
        ax[0].add_patch(rect)

        class_idx = int(obj_res.boxes.cls[idx])  # Class index for this box
        class_name = obj_res.names[class_idx]    # Class name from the dict
        
        # Annotate the box with the class name
        ax[0].text(x_min, y_min - 5, class_name, color='white', fontsize=10, weight='bold',
                   bbox=dict(facecolor='red', edgecolor='none', pad=1))

    
    # Plot bounding boxes on the second image
    for idx, box in enumerate(obj_res.boxes.xyxy):  # obj_res should be a list of [x_min, y_min, x_max, y_max] for each bounding box
        x_min, y_min, x_max, y_max = box
        
        # Create a Rectangle patch
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
        
        # Add the patch to the Axes
        ax[1].add_patch(rect)

        class_idx = int(obj_res.boxes.cls[idx])  # Class index for this box
        class_name = obj_res.names[class_idx]    # Class name from the dict
        
        # Annotate the box with the class name
        ax[1].text(x_min, y_min - 5, class_name, color='white', fontsize=10, weight='bold',
                   bbox=dict(facecolor='red', edgecolor='none', pad=1))
    
    # Save the plot
    plt.savefig('prediction_with_bboxes.png')
    
    # Show the plot
    plt.show()


    
