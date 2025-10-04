import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models, datasets
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')

from config import *
from data.data_loader import prepare_data
from data.data_utils import display_images, plot_class_distribution
from models.custom_cnn import BrainTumorCNN
from models.resnet_model import get_resnet18_model, setup_resnet18_optimizer, setup_resnet18_finetuning
from utils.train_utils import train_model, plot_training_history
from utils.eval_utils import test_model, visualize_predictions

def main():
    print("Starting Brain Tumor Classification Pipeline...")
    
    print("\n1. Data Exploration:")
    categories = os.listdir(base_directory+'/'+train)
    print(f"Categories: {categories}")
    
    plot_class_distribution(train)
    display_images(train)
    
    plot_class_distribution(test)
    display_images(test)
    
    print("\n2. Preparing Data...")
    train_loader, val_loader, test_loader = prepare_data()
    
    print(f"\n3. Training Custom CNN Model...")
    model1 = BrainTumorCNN(num_classes=4).to(device) 
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.0007)
    
    print("Starting training...")
    history1 = train_model(model1, train_loader, val_loader, criterion, optimizer1, num_epochs=60)
    
    plot_training_history(history1, model_name="CustomCNN")
    
    print("\nLoading best model for testing...")
    model1.load_state_dict(torch.load('models/best_brain_tumor_model.pth'))
    test_model(model1, test_loader)
    visualize_predictions(model1, test_loader, 8)
    
    print(f"\n4. Training ResNet18 Model...")
    model2 = get_resnet18_model(num_classes=4, device=device)
    optimizer2 = setup_resnet18_optimizer(model2, lr=0.001)
    
    print("Starting training...")
    history2 = train_model(model2, train_loader, val_loader, criterion, optimizer2, num_epochs=60, name='resnet18')
    
    plot_training_history(history2, model_name="ResNet18")
    
    print("\nLoading best model for testing...")
    model2.load_state_dict(torch.load('models/best_brain_tumor_resnet18.pth'))
    test_model(model2, test_loader)
    visualize_predictions(model2, test_loader, 8)
    
    print(f"\n5. Fine-tuning ResNet18...")
    optimizer3 = setup_resnet18_finetuning(model2, lr=1e-4)
    
    print("Starting fine-tuning...")
    history3 = train_model(model2, train_loader, val_loader, criterion, optimizer3, num_epochs=30, name='resnet18_finetuned')
    
    plot_training_history(history3, model_name="ResNet18-Finetuned")
    
    print("\nLoading best fine-tuned model for testing...")
    model2.load_state_dict(torch.load('models/best_brain_tumor_resnet18_finetuned.pth'))
    test_model(model2, test_loader)
    visualize_predictions(model2, test_loader, 8)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()