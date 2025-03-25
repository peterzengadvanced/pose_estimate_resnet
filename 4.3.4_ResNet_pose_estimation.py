import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from scipy.spatial.transform import Rotation as R
from PIL import Image
import matplotlib.pyplot as plt
import json
import random

# import torchviz for model architecture visualization
try:
    from torchviz import make_dot
    torchviz_available = True
except ImportError:
    torchviz_available = False
    print("Warning: torchviz is not installed. Model architecture visualization will be skipped.")

# visualizations
VIS_DIR = "C:/Users/Peter Zeng/Desktop/morevisualizations/resnetdatawitha/morevisualizations/"
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(os.path.join(VIS_DIR, "dataset_samples"), exist_ok=True)

# Define dataset class
class PoseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        print("Loading dataset from:", root_dir)
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.data_pairs = []
        for img_file in self.image_files:
            txt_file = os.path.splitext(img_file)[0] + '.txt'
            txt_path = os.path.join(root_dir, txt_file)
            if os.path.exists(txt_path):
                self.data_pairs.append((img_file, txt_path))
            else:
                print(f"Warning: {txt_file} not found for image {img_file}")
        print(f"Dataset loaded with {len(self.data_pairs)} image-text pairs.")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_name, txt_name = self.data_pairs[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        with open(txt_name, 'r') as f:
            lines = f.readlines()
            try:
                part_pose = np.array([float(x) for x in lines[1].strip().split()] + 
                                     [float(x) for x in lines[2].strip().split()] + 
                                     [float(x) for x in lines[3].strip().split()] + 
                                     [float(x) for x in lines[4].strip().split()] + 
                                     [float(x) for x in lines[5].strip().split()] + 
                                     [float(x) for x in lines[6].strip().split()][:6])
                cam_pose = np.array([float(x) for x in lines[13].strip().split()] + 
                                    [float(x) for x in lines[14].strip().split()] + 
                                    [float(x) for x in lines[15].strip().split()] + 
                                    [float(x) for x in lines[16].strip().split()] + 
                                    [float(x) for x in lines[17].strip().split()] + 
                                    [float(x) for x in lines[18].strip().split()][:6])
            except IndexError:
                print(f"Error reading pose data from {txt_name}")
                part_pose = np.zeros(6)
                cam_pose = np.zeros(6)

        def compute_relative_pose(part_pose, cam_pose):
            def pose_to_matrix(pose):
                x, y, z, roll, pitch, yaw = pose
                rot = R.from_euler('XYZ', [roll, pitch, yaw]).as_matrix()
                T = np.eye(4)
                T[:3, :3] = rot
                T[:3, 3] = [x, y, z]
                return T

            T_part = pose_to_matrix(part_pose)
            T_cam = pose_to_matrix(cam_pose)
            T_rel = np.dot(np.linalg.inv(T_cam), T_part)
            translation = T_rel[:3, 3]
            euler = R.from_matrix(T_rel[:3, :3]).as_euler('XYZ')
            return np.concatenate([translation, euler]).astype(np.float32)

        relative_pose = compute_relative_pose(part_pose, cam_pose)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(relative_pose)

# Define model
class PoseRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 6)

    def forward(self, x):
        return self.resnet(x)

# Visualization functions

def visualize_dataset_samples(dataset, num_samples=6):
    print("Visualizing dataset samples...")
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 5))
    if len(indices) == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        image, pose = dataset[idx]
        # Denormalize image for visualization
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        image_denorm = inv_normalize(image).clamp(0, 1)
        np_img = image_denorm.permute(1, 2, 0).numpy()
        ax.imshow(np_img)
        ax.set_title(f"Pose:\n{pose.numpy().round(2)}", fontsize=8)
        ax.axis('off')
    sample_path = os.path.join(VIS_DIR, "dataset_samples", "sample_images.png")
    plt.tight_lamet()
    plt.savefig(sample_path)
    plt.close()
    print(f"Saved dataset sample visualizations to {sample_path}")

def visualize_pose_distribution(dataset):
    print("Visualizing pose distribution...")
    poses = []
    for i in range(len(dataset)):
        _, pose = dataset[i]
        poses.append(pose.numpy())
    poses = np.array(poses)
    labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for i, ax in enumerate(axs.flatten()):
        ax.hist(poses[:, i], bins=30, color='skyblue', edgecolor='black')
        ax.set_title(f"Distribution of {labels[i]}")
    plt.tight_lamet()
    dist_path = os.path.join(VIS_DIR, "pose_distribution.png")
    plt.savefig(dist_path)
    plt.close()
    print(f"Saved pose distribution visualization to {dist_path}")

def visualize_model_architecture(model, device):
    if not torchviz_available:
        print("Skipping model architecture visualization (torchviz not available).")
        return
    print("Visualizing model architecture...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    model.eval()
    output = model(dummy_input)
    dot = make_dot(output, params=dict(model.named_parameters()))
    arch_path = os.path.join(VIS_DIR, "model_architecture")
    dot.format = 'png'
    dot.render(arch_path, cleanup=True)
    print(f"Saved model architecture visualization to {arch_path}.png")

def visualize_predictions(model, val_loader, device, num_samples=6):
    print("Visualizing predictions on validation data...")
    model.eval()
    images = []
    gt_poses = []
    pred_poses = []
    count = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            for i in range(inputs.size(0)):
                images.append(inputs[i].cpu())
                gt_poses.append(labels[i].cpu().numpy())
                pred_poses.append(outputs[i].cpu().numpy())
                count += 1
                if count >= num_samples:
                    break
            if count >= num_samples:
                break

    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    for i, ax in enumerate(axes):
        # Denormalize image for visualization
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        img_denorm = inv_normalize(images[i]).clamp(0, 1)
        np_img = img_denorm.permute(1, 2, 0).numpy()
        ax.imshow(np_img)
        ax.axis('off')
        ax.set_title(f"GT: {np.round(gt_poses[i],2)}\nPred: {np.round(pred_poses[i],2)}", fontsize=8)
    pred_path = os.path.join(VIS_DIR, "sample_predictions.png")
    plt.tight_lamet()
    plt.savefig(pred_path)
    plt.close()
    print(f"Saved sample predictions visualization to {pred_path}")

def train():
    print("Starting training process...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = PoseDataset("C:/Users/Peter Zeng/Desktop/resnetdata/rawimageswithpose/", transform)
    
    # Visualize dataset samples and pose distribution
    visualize_dataset_samples(dataset)
    visualize_pose_distribution(dataset)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"Split dataset into {len(train_set)} training samples and {len(val_set)} validation samples.")

    num_workers = 0 if os.name == 'nt' else 4
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=num_workers)

    model = PoseRegressor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Visualize model architecture
    visualize_model_architecture(model, device)

    os.makedirs("C:/Users/Peter Zeng/Desktop/morevisualizations/resnetdatawitha/step2outcome/models", exist_ok=True)
    os.makedirs("C:/Users/Peter Zeng/Desktop/morevisualizations/resnetdatawitha/step2outcome/logs", exist_ok=True)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(5):
        print(f"Epoch {epoch+1} starting...")
        model.train()
        running_loss = 0.0
        batch_count = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"Processed {batch_count} batches in epoch {epoch+1}...")
        train_loss = running_loss / len(train_set)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
        val_loss /= len(val_set)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1} completed. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "C:/Users/Peter Zeng/Desktop/morevisualizations/resnetdatawitha/step2outcome/models/best_model.pth")
            print("Best model updated and saved.")

        # Save intermediate loss curve for the epoch
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend()
        epoch_curve_path = os.path.join(VIS_DIR, f"training_curve_epoch_{epoch+1}.png")
        plt.savefig(epoch_curve_path)
        plt.close()
        print(f"Saved training curve for epoch {epoch+1} to {epoch_curve_path}")

    # Final training loss curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    final_curve_path = os.path.join(VIS_DIR, "final_training_curve.png")
    plt.savefig(final_curve_path)
    plt.close()
    print(f"Saved final training loss curve visualization to {final_curve_path}")

    # Visualize predictions on validation data
    visualize_predictions(model, val_loader, device)
    print("Training process completed.")

if __name__ == "__main__":
    train()
