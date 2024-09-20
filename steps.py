# preprocess.ipynb ve segmentasyon.ipynb
# fe.ipynb
# transformers.ipynb



"""
https://www.kaggle.com/code/bahoho/notebooke40127fc25/edit

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load and preprocess data
data = pd.read_csv('/kaggle/input/tf-bcg/segmented_data_statistics.csv')
data['Label'] = data['Filename'].apply(lambda x: 1 if 'H' in x else 0)

# Select 80 hypertensive and 20 normal samples
hypertensive_data = data[data['Label'] == 1].sample(n=200, random_state=42)
normal_data = data[data['Label'] == 0].sample(n=800, random_state=42)
data = pd.concat([hypertensive_data, normal_data])

X = data.drop(columns=['Filename', 'Label'])
y = data['Label']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)

# Define Transformer model
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=128)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = x.mean(dim=1)   # Pool over the sequence
        x = self.fc(x)
        return x

# Calculate class weights
class_weights = torch.tensor([1.0 / np.mean(y_train.numpy() == 0), 1.0 / np.mean(y_train.numpy() == 1)], dtype=torch.float32)

# Instantiate and train the model
model = TransformerClassifier(input_dim=X_train.shape[1], num_classes=2)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def train_model(model, X_train, y_train, X_test, y_test, criterion, optimizer, scheduler, epochs=20):
    best_accuracy = 0
    patience = 5
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            test_output = model(X_test)
            _, predicted = torch.max(test_output, 1)
            accuracy = accuracy_score(y_test.numpy(), predicted.numpy())

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                epochs_no_improve = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print('Early stopping')
                    break

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_output = model(X_test)
        _, predicted = torch.max(test_output, 1)
        y_test_np = y_test.numpy()
        predicted_np = predicted.numpy()

        accuracy = accuracy_score(y_test_np, predicted_np)
        f1 = f1_score(y_test_np, predicted_np, average='weighted')
        conf_matrix = confusion_matrix(y_test_np, predicted_np)

        # Calculate additional metrics
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Sensitivity (Recall): {sensitivity:.4f}')
        print(f'Specificity: {specificity:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print('Confusion Matrix:')
        print(conf_matrix)
        print('Classification Report:')
        print(classification_report(y_test_np, predicted_np, target_names=['Normal', 'Hypertensive']))

# Train the model
train_model(model, X_train, y_train, X_test, y_test, criterion, optimizer, scheduler, epochs=20)

"""