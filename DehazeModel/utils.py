import torch
import torch.nn as nn
import torchvision.transforms as tfs

from .FFANet import FFA

device = "cuda" if torch.cuda.is_available() else "cpu"


def GetFFA(path=None):
    model = FFA(gps=3, blocks=19)
    model = nn.DataParallel(model)

    if path is None:
        path = "./DehazeModel/trained_models/ffa_net.pk"
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    return model


def TrainFFA(
    model_path=None,
    train_loader=None,
    epochs=10,
    learning_rate=0.001
):
    # Initialize the model
    model = GetFFA(model_path)
    model.train()
    
    # Define loss function (MAE) and optimizer
    criterion = torch.nn.L1Loss()  # MAE is implemented as L1Loss in PyTorch
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images in train_loader:
            optimizer.zero_grad()
            
            # Generate reconstructed images
            generated_images = model(images)
            
            # Compute the loss using MAE
            loss = criterion(generated_images, images)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss/len(train_loader)}")
    
    # Save the trained model
    torch.save(model.state_dict(), model_path if model_path else "FFANet_trained_model.pth")
    print("Training completed and model saved.")


def EvaluateFFA(model_path=None, test_loader=None):
    # Initialize the model
    model = GetFFA(model_path)
    model.eval()
    
    with torch.no_grad():
        for i, images in enumerate(test_loader):
            # Generate images using the model
            generated_images = model(images)
            
            # Save the generated images
            # save_image(generated_images, f"output_{i}.png")

    print("Evaluation completed. Generated images saved.")


def InferenceFFA(img, model_path=None):
    model = GetFFA(model_path)

    mean = torch.tensor([0.64, 0.6, 0.58])
    std = torch.tensor([0.14, 0.15, 0.152])

    transform = tfs.Compose([
        tfs.Normalize(mean, std)
    ])

    img_norm = transform(img)

    print(img_norm.shape)

    with torch.no_grad():
        img_pred = model(img_norm)

    transform_unnorm = tfs.Compose([
        tfs.Normalize((-mean / std), (1.0 / std)),
    ])

    img_pred_unnorm = transform_unnorm(img_pred.clamp(0,1))

    return img_pred_unnorm