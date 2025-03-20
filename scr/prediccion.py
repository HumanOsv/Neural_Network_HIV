import torch
import numpy as np
import pandas as pd
import sys
import os
from sklearn.preprocessing import StandardScaler

# Define la clase NeuralNetwork (debe ser idéntica a la usada durante el entrenamiento)
class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size=3, hidden_size=39, output_size=1):
        super(NeuralNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()
       
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

def load_model(path='best_model.pth'):
    """Carga un modelo guardado"""
    model = NeuralNetwork(input_size=3)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

def predict_values(input_values, model_path='best_model.pth', scaler=None):
    """
    Realiza predicciones con nuevos datos
    
    Args:
        input_values: Lista o array de filas con 3 valores cada una
        model_path: Ruta al modelo guardado
        scaler: Opcional, el scaler usado durante el entrenamiento
        
    Returns:
        Array con las predicciones
    """
    # Convertir entrada a numpy array si no lo es
    if not isinstance(input_values, np.ndarray):
        input_values = np.array(input_values)
    
    # Asegurar que tenga la forma correcta
    if input_values.ndim == 1 and len(input_values) == 3:
        input_values = input_values.reshape(1, -1)
    
    # Cargar el modelo
    model = load_model(model_path)
    
    # Realizar predicciones
    model.eval()
    with torch.no_grad():
        # Normalizar si se proporciona un scaler
        if scaler is not None:
            input_values = scaler.transform(input_values)
        
        # Convertir a tensor
        input_tensor = torch.FloatTensor(input_values)
        
        # Realizar predicción
        predictions = model(input_tensor)
    
    return predictions.numpy()

def process_input_file(input_file):
    """
    Procesa un archivo CSV de entrada y genera predicciones
    
    Args:
        input_file: Ruta al archivo CSV con los datos de entrada
        
    Returns:
        DataFrame con los datos de entrada y las predicciones
    """
    # Leer el archivo de entrada
    df = pd.read_csv(input_file)
    
    # Extraer los valores de entrada
    input_values = df[['valor1', 'valor2', 'valor3']].values
    
    # Realizar predicciones
    predictions = predict_values(input_values)
    
    # Añadir las predicciones al DataFrame
    df['prediccion'] = predictions
    
    return df

def process_input_data(input_data):
    """
    Procesa datos de entrada en formato de lista y genera predicciones
    
    Args:
        input_data: Lista de listas con los valores de entrada
        
    Returns:
        DataFrame con los datos de entrada y las predicciones
    """
    # Convertir a numpy array
    input_array = np.array(input_data)
    
    # Realizar predicciones
    predictions = predict_values(input_array)
    
    # Crear DataFrame con los datos de entrada y las predicciones
    df = pd.DataFrame(input_array, columns=['valor1', 'valor2', 'valor3'])
    df['prediccion'] = predictions
    
    return df

def save_predictions(df, output_file='resultados_prediccion.csv'):
    """
    Guarda las predicciones en un archivo CSV
    
    Args:
        df: DataFrame con los datos y predicciones
        output_file: Ruta del archivo de salida
    """
    df.to_csv(output_file, index=False)
    print(f"Predicciones guardadas en {output_file}")
    return output_file

def main():
    """Función principal para procesamiento por línea de comandos"""
    if len(sys.argv) < 2:
        print("Uso: python prediccion.py [archivo_entrada.csv o 'valor1 valor2 valor3']")
        return
    
    # Verificar si el argumento es un archivo o valores directos
    input_arg = sys.argv[1]
    
    if os.path.isfile(input_arg):
        # Procesar archivo
        df = process_input_file(input_arg)
    else:
        # Procesar valores directos
        try:
            # Intentar procesar todos los argumentos como valores
            values = []
            for arg in sys.argv[1:]:
                # Dividir cada argumento por espacios
                arg_values = [float(val) for val in arg.split()]
                if len(arg_values) == 3:
                    values.append(arg_values)
                else:
                    # Si no tiene 3 valores, tratar cada argumento como un valor individual
                    for val in arg_values:
                        values.append(float(val))
            
            # Si tenemos una lista plana de valores, reorganizarla en grupos de 3
            if len(values) > 0 and not isinstance(values[0], list):
                if len(values) % 3 != 0:
                    print("Error: El número de valores debe ser múltiplo de 3")
                    return
                values = [values[i:i+3] for i in range(0, len(values), 3)]
            
            df = process_input_data(values)
        except ValueError:
            print("Error: Los valores deben ser numéricos")
            return
    
    # Guardar predicciones
    output_file = save_predictions(df)
    
    # Mostrar resultados
    print("\nResultados de predicción:")
    for i, row in df.iterrows():
        input_vals = row[['valor1', 'valor2', 'valor3']].values
        pred = row['prediccion']
        print(f"Entrada {i+1} {input_vals}: {pred:.4f}")
    
    return output_file

if __name__ == "__main__":
    main()
