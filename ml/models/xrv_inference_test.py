import torchxrayvision as xrv
import skimage, torch, torchvision
import numpy as np

# Prepare the image:
img = skimage.io.imread(r"C:\Users\User\Sadeepa\X-Lite\data\clahe_cache\00012141_015.png") # Load your image here
img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
if img.ndim == 3:
    img = img.mean(2)
img = img[None, ...] # Make single color channel

transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224),
])

img = transform(img)
img = torch.from_numpy(img)

# Load model and process image
model = xrv.models.DenseNet(weights="densenet121-res224-all")
outputs = model(img[None,...]) # or model.features(img[None,...])

# Print results
results = dict(zip(model.pathologies, outputs[0].detach().numpy()))
print(results)
