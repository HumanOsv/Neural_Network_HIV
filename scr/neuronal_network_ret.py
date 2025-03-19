
#X = entrada(:,1:3);    % Primeras 3 columnas como entradas
#T = entrada(:,4);      % Cuarta columna como target

#import torch
#import torch.nn as nn
#import pandas as pd
#import numpy as np
#import time
#from scipy.stats import pearsonr
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
#from sklearn.model_selection import train_test_split
#import joblib


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

# Definición de la red neuronal

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=3, hidden_size=39, output_size=1):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
       
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

#class NeuralNetwork(nn.Module):
#    def __init__(self, input_size=3, hidden_size1=32, hidden_size2=16, output_size=1):
#        super(NeuralNetwork, self).__init__()
#        self.layer1 = nn.Linear(input_size, hidden_size1)
#        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
#        self.layer3 = nn.Linear(hidden_size2, output_size)
#        self.relu = nn.ReLU()
        
#    def forward(self, x):
#        x = self.relu(self.layer1(x))
#        x = self.relu(self.layer2(x))
#        x = self.layer3(x)
#        return x

#class NeuralNetwork(nn.Module):
#    def __init__(self, input_size=3, hidden_sizes=[64, 32, 16, 8], output_size=1):
#        super(NeuralNetwork, self).__init__()
#        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
#        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
#        self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
#        self.layer4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
#        self.layer5 = nn.Linear(hidden_sizes[3], output_size)
#        self.relu = nn.ReLU()
        
#    def forward(self, x):
#        x = self.relu(self.layer1(x))
#        x = self.relu(self.layer2(x))
#        x = self.relu(self.layer3(x))
#        x = self.relu(self.layer4(x))
#        x = self.layer5(x)
#        return x

def load_data(file_path):
    """Carga y preprocesa los datos desde un archivo Excel"""
    try:
        # Leer el archivo Excel
        df = pd.read_excel(file_path)
        print(f"Datos cargados correctamente. Forma: {df.shape}")
        
        # Separar features (X) y target (y)
        X = df.iloc[:, :3].values  # Tres primeras columnas como entrada
        y = df.iloc[:, 3].values   # Cuarta columna como salida
        
        # Normalizar los datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Convertir a tensores de PyTorch
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
        
        return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
                X_train, y_train, X_test, y_test, scaler)
    
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None

def train_model(model, X_train, y_train, X_test, y_test, epochs=1000, lr=0.001):
    """Entrena el modelo y registra las métricas durante el entrenamiento"""
    # Definir función de pérdida y optimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Listas para almacenar métricas durante el entrenamiento
    train_losses = []
    test_losses = []
    correlations = []
    
    # Entrenamiento del modelo
    best_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        # Modo entrenamiento
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass y optimización
        loss.backward()
        optimizer.step()
        
        # Evaluar en conjunto de prueba
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            
            # Calcular correlación
            pred_np = test_outputs.numpy().flatten()
            true_np = y_test.numpy().flatten()
            correlation = np.corrcoef(pred_np, true_np)[0, 1]
        
        # Guardar métricas
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        correlations.append(correlation)
        
        # Guardar el mejor modelo
        if test_loss < best_loss:
            best_loss = test_loss
            best_model = model.state_dict().copy()
        
        # Imprimir progreso cada 100 épocas
        if (epoch + 1) % 100 == 0:
            print(f'Época [{epoch+1}/{epochs}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Test Loss: {test_loss.item():.4f}, '
                  f'Correlación: {correlation:.4f}')
    
    # Cargar el mejor modelo
    model.load_state_dict(best_model)
    
    return model, train_losses, test_losses, correlations

def save_model(model, path='best_model.pth'):
    """Guarda el modelo entrenado"""
    torch.save(model.state_dict(), path)
    print(f"Modelo guardado en {path}")

def load_model(path='best_model.pth', input_size=3):
    """Carga un modelo guardado"""
    model = NeuralNetwork(input_size=input_size)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_cases(model, new_data, scaler=None):
    """Realiza predicciones con nuevos datos"""
    model.eval()
    with torch.no_grad():
        # Normalizar si se proporciona un scaler
        if scaler is not None:
            new_data = scaler.transform(new_data)
        
        new_data_tensor = torch.FloatTensor(new_data)
        predictions = model(new_data_tensor)
    
    return predictions.numpy()

def plot_training_metrics(train_losses, test_losses, correlations):
    """Visualiza las métricas durante el entrenamiento"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Gráfico de pérdida
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Pérdida Entrenamiento')
    plt.plot(epochs, test_losses, 'r-', label='Pérdida Prueba')
    plt.title('Pérdida durante Entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (MSE)')
    plt.legend()
    
    # Gráfico de correlación
    plt.subplot(1, 3, 3)
    plt.plot(epochs, correlations, 'g-')
    plt.title('Correlación durante Entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Correlación')
    
    plt.tight_layout()
    plt.show()

def main():
    """Función principal que ejecuta todo el proceso"""
    # 1. Cargar datos
    print("Cargando datos...")
    data = load_data('datos-entrada.xlsx')
    if data is None:
        return
    
    X_train, y_train, X_test, y_test, X_train_np, y_train_np, X_test_np, y_test_np, scaler = data
    
    # 2. Crear modelo
    print("\nCreando modelo de red neuronal...")
    input_size = X_train.shape[1]
    model = NeuralNetwork(input_size=input_size)
    print(model)
    
    # 3. Entrenar modelo
    print("\nEntrenando modelo...")
    trained_model, train_losses, test_losses, correlations = train_model(
        model, X_train, y_train, X_test, y_test, epochs=1000
    )
        
    # 5. Guardar modelo
    print("\nGuardando el modelo...")
    save_model(trained_model)
    
    # 6. Visualizar métricas de entrenamiento
    print("\nVisualizando métricas de entrenamiento...")
    plot_training_metrics(train_losses, test_losses, correlations)
    
    # 7. Cargar modelo y hacer predicciones
    print("\nCargando modelo guardado y realizando predicciones...")
    loaded_model = load_model()
    
    # Ejemplo de predicción con datos nuevos
    new_data = np.array([[10, 19, 46]])  # Ejemplo de datos de entrada
    predicted_cases = predict_cases(loaded_model, new_data, scaler)
    print("\nPredicción para nuevos datos:")
    print(f"Total casos predichos: {predicted_cases[0][0]:.4f}")

if __name__ == "__main__":
    main()