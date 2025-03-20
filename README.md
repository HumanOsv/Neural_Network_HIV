# Monitoreo y predicción de la epidemia de sida en chile empleando redes neuronales

Cuando una persona tiene interés en buscar información acerca del SIDA, suele acudir a buscadores en internet, como Google, empleando términos de búsqueda apropiados, tales como: “test de VIH”, “Triterapia”, “Diagnóstico y tratamiento de SIDA”, etc. Considerando que el interés de búsqueda puede estar relacionado por alguna sospecha o certeza acerca del padecimiento de la enfermedad, creemos que es posible monitorear y predecir las incidencias y muertes por SIDA en Chile empleando patrones de datos de tendencias de búsqueda en Google. Contando con esa hipótesis, el objetivo de nuestro proyecto es desarrollar redes neuronales artificiales que, a partir de datos de tendencias de búsqueda en Google, puedan monitorear y predecir las incidencias y muertes de SIDA en el país.

Más ampliamente, mediante políticas de salud basadas en información epidemiológica adecuada, se intenta disminuir la letalidad y la propagación de la epidemia de VIH. La información, proveniente de los centros de salud de todo el país, debe ser actualizada y estar prontamente disponible para ser usada en modelos estadísticos predictivos. Sin embargo, la tardanza o la falta de toda la información necesaria pueden afectar el desempeño de esas técnicas. Considerando que en la red de internet hay gran cantidad de datos en tiempo real, en condición de big data, asociados a las conductas de búsquedas de los usuarios sobre temas de diagnóstico, síntomas y tratamientos de enfermedades, esta información se puede emplear en conjunto con métodos de redes neuronales artificiales para establecer sistemas de inteligencia artificial que monitoreen epidemias de distinto tipo, como la de SIDA. Este proyecto propone la implementación de aquellas técnicas, adaptándolas a la realidad y necesidades de nuestro país en el mejoramiento de los sistemas de control epidemiológico de incidencia y muertes por VIH/SIDA.


*FONIS-SA22I0129-2000*


# Red Neuronal para Predicción con PyTorch

Este código implementa una red neuronal feedforward para un problema de regresión con 3 variables de entrada y 1 variable de salida.

## Arquitectura de la Red Neuronal

La red neuronal implementada es un perceptrón multicapa (MLP) con 2 capas:

```python
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
```

- **Capa de entrada**: Recibe 3 características (input_size=3)
- **Capa oculta**: Contiene 39 neuronas (hidden_size=39) con activación ReLU
- **Capa de salida**: Produce 1 valor de salida (output_size=1) sin función de activación (para regresión)

## Proceso de Entrenamiento

El entrenamiento se realiza en la función `train_model()` con los siguientes componentes:

1. **Función de pérdida**: Error Cuadrático Medio (MSE)
2. **Optimizador**: Adam con tasa de aprendizaje de 0.001
3. **Número de épocas**: 1000
4. **Métricas de seguimiento**:
   - Pérdida en entrenamiento
   - Pérdida en prueba
   - Correlación entre predicciones y valores reales

Durante el entrenamiento:
- Se guarda el modelo con mejor rendimiento en el conjunto de prueba
- Se imprime el progreso cada 100 épocas
- Se registran las métricas para visualización posterior

```python
def train_model(model, X_train, y_train, X_test, y_test, epochs=1000, lr=0.001):
    # Definir función de pérdida y optimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Listas para almacenar métricas
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
```

## Preprocesamiento de Datos

Antes del entrenamiento, los datos se preparan en la función `load_data()`:

1. Lectura del archivo Excel 'datos-entrada.xlsx'
2. Separación de características (3 primeras columnas) y variable objetivo (4ª columna)
3. Normalización de las características usando StandardScaler
4. División en conjuntos de entrenamiento (80%) y prueba (20%)
5. Conversión a tensores de PyTorch

## Visualización y Evaluación

Después del entrenamiento, se visualizan las métricas con gráficos de:
- Pérdida de entrenamiento y prueba vs. épocas
- Correlación vs. épocas

## Uso del Modelo

El modelo entrenado se guarda y puede cargarse para realizar predicciones con nuevos datos:

```python
# Guardar modelo
save_model(trained_model)

# Cargar modelo
loaded_model = load_model()

# Hacer predicciones
new_data = np.array([[10, 19, 46]])
predicted_cases = predict_cases(loaded_model, new_data, scaler)
```

## Flujo de Ejecución

La función `main()` coordina todo el proceso:
1. Carga de datos
2. Creación del modelo
3. Entrenamiento
4. Guardado del modelo
5. Visualización de métricas
6. Carga del modelo y predicción con nuevos datos

# Ejecutar código: 

Para iniciar la aplicación, ejecute el siguiente comando en su terminal:

```
python main.py
```

Luego se desplegará una interfaz gráfica (GUI) con tres botones principales ubicados en el panel izquierdo:

## Datos VIH
Este botón permite cargar el documento "Notificaciones_VIH_2010_2019.xlsx" que contiene los datos históricos de casos de VIH. Al hacer clic:
- Se abrirá un explorador de archivos para seleccionar el documento Excel
- Una vez cargado, se procesarán los datos automáticamente
- Se mostrarán dos gráficos interactivos en el panel central: uno de casos totales y otro de casos anuales
- Puede hacer clic en cualquier gráfico para verlo en tamaño completo con opción de zoom

## Correr Red Neuronal
Este botón ejecuta el entrenamiento de la red neuronal para predecir casos de VIH. Al hacer clic:
- Aparecerá un mensaje informando que el proceso puede tardar varios minutos
- El sistema ejecutará el script "neuronal_network_ret.py" que entrena el modelo
- Al finalizar, se mostrará automáticamente un gráfico "nn_optim.png" con dos métricas clave:
  - La pérdida durante el entrenamiento (comparando datos de entrenamiento y prueba)
  - La correlación entre valores predichos y reales
- La ventana del gráfico es adaptable y permite hacer zoom con la rueda del ratón

## Predicción
Este botón abre una ventana para realizar predicciones con el modelo entrenado. Al hacer clic:
- Se abrirá una ventana de 400x300 píxeles
- Podrá introducir múltiples filas de datos, cada una con tres valores numéricos
- Cada fila representa un conjunto de parámetros para predecir un caso
- Al presionar "Iniciar Predicción", el sistema:
  - Validará que los datos ingresados sean correctos
  - Procesará los datos a través del modelo entrenado
  - Generará un archivo CSV con los resultados
  - Mostrará una nueva ventana con los datos de entrada y las predicciones correspondientes
