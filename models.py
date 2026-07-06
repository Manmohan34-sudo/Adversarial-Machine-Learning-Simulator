import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os
import urllib.request
import json

# Custom CNN for MNIST digit classification
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Input channel is 1 (grayscale), outputting 16 feature maps
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # Conv2 maps 16 features to 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2) # Divides dimensions by 2
        # MaxPool is applied after Conv1: 28x28 -> 14x14
        # Conv2 keeps it 14x14
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 14 * 14)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def simple_blur(img):
    """Applies a simple 3x3 mean filter to smooth synthetic strokes."""
    padded = np.pad(img, ((0,0), (1,1), (1,1)), mode='constant')
    blurred = np.zeros_like(img)
    for i in range(1, padded.shape[1]-1):
        for j in range(1, padded.shape[2]-1):
            blurred[0, i-1, j-1] = np.mean(padded[0, i-1:i+2, j-1:j+2])
    return blurred

def train_toy_mnist_model(model, filepath):
    """Trains the model on synthetically generated digit-like shapes so it works instantly offline."""
    print("Training toy MNIST model on synthetic shapes...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    x_train = []
    y_train = []
    
    np.random.seed(42)
    for digit in range(10):
        for _ in range(80):  # 80 synthetic samples per class
            img = np.zeros((1, 28, 28), dtype=np.float32)
            
            # Synthesize basic visual strokes representation of digits
            if digit == 0:  # Circle
                for theta in np.linspace(0, 2 * np.pi, 25):
                    r = 7 + np.random.uniform(-0.5, 0.5)
                    cx = int(14 + r * np.cos(theta))
                    cy = int(14 + r * np.sin(theta))
                    if 0 <= cx < 28 and 0 <= cy < 28:
                        img[0, cx, cy] = 1.0
            elif digit == 1:  # Vertical stroke
                img[0, 4:24, 14] = 1.0
            elif digit == 2:  # Upper curve + base
                img[0, 6, 10:18] = 1.0
                img[0, 6:13, 18] = 1.0
                for i in range(7):
                    img[0, 13+i, 18-int(i*1.2)] = 1.0
                img[0, 20, 10:19] = 1.0
            elif digit == 3:  # Top curve + mid + bottom curve
                img[0, 6, 10:18] = 1.0
                img[0, 6:21, 18] = 1.0
                img[0, 13, 11:18] = 1.0
                img[0, 20, 10:18] = 1.0
            elif digit == 4:  # Open/closed vertical plus horizontal cross
                img[0, 6:14, 9] = 1.0
                img[0, 13, 9:19] = 1.0
                img[0, 6:22, 17] = 1.0
            elif digit == 5:  # Top line + left curve + bottom curve
                img[0, 6, 9:19] = 1.0
                img[0, 6:13, 9] = 1.0
                img[0, 13, 9:19] = 1.0
                img[0, 13:21, 18] = 1.0
                img[0, 20, 9:19] = 1.0
            elif digit == 6:  # Loop at bottom left + full outline
                img[0, 6:21, 9] = 1.0
                img[0, 13:21, 17] = 1.0
                img[0, 13, 9:18] = 1.0
                img[0, 20, 9:18] = 1.0
            elif digit == 7:  # Top horizontal + diagonal down
                img[0, 6, 9:19] = 1.0
                for i in range(15):
                    img[0, 6+i, 18-int(i*0.6)] = 1.0
            elif digit == 8:  # Two stacked loops
                img[0, 6, 10:18] = 1.0
                img[0, 13, 10:18] = 1.0
                img[0, 20, 10:18] = 1.0
                img[0, 6:21, 9] = 1.0
                img[0, 6:21, 18] = 1.0
            elif digit == 9:  # Loop at top right + vertical line
                img[0, 6:14, 9] = 1.0
                img[0, 6, 9:19] = 1.0
                img[0, 13, 9:19] = 1.0
                img[0, 6:22, 18] = 1.0
            
            # Smooth the strokes to look hand-drawn and add random noise
            img = simple_blur(img)
            img += np.random.normal(0, 0.05, img.shape)
            img = np.clip(img, 0.0, 1.0)
            x_train.append(img)
            y_train.append(digit)
            
    x_train = torch.tensor(np.array(x_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.long)
    
    # Simple training loop (15 epochs is enough for high train accuracy on these toy shapes)
    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print("MNIST toy model trained and saved.")

def get_mnist_model(weights_path="backend/data/mnist_toy.pth"):
    model = MNISTNet()
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            print("Loaded MNIST model weights successfully.")
        except Exception:
            train_toy_mnist_model(model, weights_path)
    else:
        train_toy_mnist_model(model, weights_path)
    model.eval()
    # Disable gradient tracking on parameters
    for p in model.parameters():
        p.requires_grad = False
    return model

def get_imagenet_model():
    """Loads pre-trained MobileNetV2. Reverts to dummy classifier if offline."""
    try:
        model = models.mobilenet_v2(pretrained=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model
    except Exception as e:
        print(f"Offline or failed to load pre-trained MobileNetV2: {e}. Instantiating fallback dummy network.")
        class DummyImageNetNet(nn.Module):
            def __init__(self):
                super(DummyImageNetNet, self).__init__()
                self.conv = nn.Conv2d(3, 10, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(10, 1000)
            def forward(self, x):
                # Basic forward pass to satisfy dimensions
                x = self.pool(F.relu(self.conv(x)))
                x = x.view(x.size(0), -1)
                return self.fc(x)
        model = DummyImageNetNet()
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

# Download ImageNet class names for user friendliness
def get_imagenet_classes(filepath="backend/data/imagenet_classes.json"):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            pass
            
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    classes_url = "https://raw.githubusercontent.com/senthilthyagarajan/Imagenet-GPUDistribution/master/imagenet1000_clsidx_to_labels.txt"
    try:
        # Fetch mapping file
        with urllib.request.urlopen(classes_url, timeout=5) as response:
            html = response.read().decode('utf-8')
            # The file is formatted like a python dict representation: {0: 'tench, Tinca tinca', 1: ...}
            # We can convert it to json format
            import ast
            raw_dict = ast.literal_eval(html)
            classes_dict = {str(k): v for k, v in raw_dict.items()}
            with open(filepath, 'w') as f:
                json.dump(classes_dict, f)
            return classes_dict
    except Exception as e:
        print(f"Could not load ImageNet classes online: {e}. Using fallback names.")
        # Fallback names for common indexes (first 10, rest are generic)
        fallback = {str(i): f"Class {i}" for i in range(1000)}
        fallback.update({
            "263": "corgi, Pembroke Welsh Corgi",
            "281": "tabby, tabby cat",
            "282": "tiger cat",
            "285": "Egyptian cat",
            "1": "goldfish, Carassius auratus",
            "18": "laboratory coat, lab coat",
            "340": "zebra",
            "386": "African elephant, Loxodonta africana",
            "970": "alp",
            "985": "daisy"
        })
        return fallback
