import os
import subprocess
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import tempfile
import csv

def execute_notificaciones_script(file_path):
    try:
        subprocess.run(["python", "notificaciones.py", file_path], check=True)
    except Exception as e:
        print(f"Error ejecutando script: {e}")

def load_and_execute():
    file_path = filedialog.askopenfilename(
        title="Seleccione archivo Excel",
        filetypes=(("Archivos Excel", "*.xlsx"), ("Todos los archivos", "*.*"))
    )
    if file_path:
        execute_notificaciones_script(file_path)
        display_generated_graphs()

def show_full_size_image(image_path):
    # Crear nueva ventana para imagen completa
    image_window = ctk.CTkToplevel()
    image_window.title("Gráfico con Zoom")
    image_window.attributes('-topmost', True)  # Mantener ventana siempre al frente
    
    # Variables para el zoom
    zoom_level = 1.0
    
    # Cargar imagen original
    img = Image.open(image_path)
    original_width, original_height = img.size
    
    def zoom(event):
        nonlocal zoom_level, photo
        
        # Ajustar nivel de zoom según dirección de la rueda
        if event.delta > 0:  # Zoom in
            zoom_level *= 1.1
        elif event.delta < 0:  # Zoom out
            zoom_level /= 1.1
        
        # Calcular nuevas dimensiones
        new_width = int(original_width * zoom_level)
        new_height = int(original_height * zoom_level)
        
        # Redimensionar imagen
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized_img)
        
        # Actualizar imagen en el label
        label.configure(image=photo)
        label.image = photo
    
    # Calcular tamaño medio para la imagen (50% del tamaño original)
    medium_width = int(original_width * 0.7)
    medium_height = int(original_height * 0.7)
    
    # Redimensionar imagen a tamaño medio
    medium_img = img.resize((medium_width, medium_height), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(medium_img)
    
    # Crear label para mostrar la imagen
    label = ctk.CTkLabel(image_window, image=photo)
    label.image = photo
    label.pack(padx=10, pady=10, fill="both", expand=True)
    
    # Vincular evento de rueda del ratón
    image_window.bind("<MouseWheel>", zoom)
    
    # Botón para cerrar
    close_button = ctk.CTkButton(image_window, text="Cerrar", command=image_window.destroy)
    close_button.pack(pady=10)
    
    # Centrar ventana y hacerla adaptable
    image_window.update_idletasks()
    screen_width = image_window.winfo_screenwidth()
    screen_height = image_window.winfo_screenheight()
    window_width = medium_width + 40
    window_height = medium_height + 80
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    image_window.geometry(f"{int(window_width)}x{int(window_height)}+{int(x)}+{int(y)}")
    
    # Hacer que la ventana sea redimensionable
    image_window.grid_rowconfigure(0, weight=1)
    image_window.grid_columnconfigure(0, weight=1)

def display_generated_graphs():
    try:
        # Limpiar frame central excepto el texto de descripción
        for widget in frame_central.winfo_children():
            if widget != texto_widget:
                widget.destroy()

        # Cargar las imágenes originales
        global total_cases_img, yearly_cases_img
        total_cases_img = Image.open("total_cases.png")
        yearly_cases_img = Image.open("yearly_cases.png")

        # Crear frame para gráficos
        graphs_frame = ctk.CTkFrame(frame_central)
        graphs_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Calcular tamaño medio para las imágenes (ajustado para que se vean bien)
        frame_width = frame_central.winfo_width() - 40
        frame_height = (frame_central.winfo_height() - 60) // 2
        
        # Asegurar un tamaño mínimo
        frame_width = max(frame_width, 400)
        frame_height = max(frame_height, 200)

        # Redimensionar imágenes
        resized_total = ImageTk.PhotoImage(total_cases_img.resize((frame_width, frame_height)))
        resized_yearly = ImageTk.PhotoImage(yearly_cases_img.resize((frame_width, frame_height)))

        # Crear labels para las imágenes
        global total_cases_label, yearly_cases_label
        total_cases_label = ctk.CTkLabel(graphs_frame, image=resized_total, cursor="hand2")
        total_cases_label.image = resized_total
        total_cases_label.pack(pady=10)
        
        # Agregar eventos de clic para cada gráfico
        total_cases_label.bind("<Button-1>", lambda e: show_full_size_image("total_cases.png"))
        
        yearly_cases_label = ctk.CTkLabel(graphs_frame, image=resized_yearly, cursor="hand2")
        yearly_cases_label.image = resized_yearly
        yearly_cases_label.pack(pady=10)
        yearly_cases_label.bind("<Button-1>", lambda e: show_full_size_image("yearly_cases.png"))

        # Función para redimensionar
        def resize_graphs(event):
            frame_width = graphs_frame.winfo_width()
            frame_height = graphs_frame.winfo_height()
            
            # Calcular nuevas dimensiones manteniendo la proporción
            new_width = frame_width - 40
            new_height = (frame_height - 60) // 2
            
            # Asegurar un tamaño mínimo
            new_width = max(new_width, 400)
            new_height = max(new_height, 200)
            
            # Redimensionar y actualizar imágenes
            resized_total = ImageTk.PhotoImage(total_cases_img.resize((new_width, new_height)))
            resized_yearly = ImageTk.PhotoImage(yearly_cases_img.resize((new_width, new_height)))
            
            total_cases_label.configure(image=resized_total)
            total_cases_label.image = resized_total
            
            yearly_cases_label.configure(image=resized_yearly)
            yearly_cases_label.image = resized_yearly

        # Vincular evento de redimensionamiento
        graphs_frame.bind("<Configure>", resize_graphs)

    except Exception as e:
        print(f"Error mostrando gráficos: {e}")

def run_neural_network():
    try:
        # Mostrar mensaje de espera
        messagebox.showinfo("Procesando", "La red neuronal está ejecutándose. Esto puede tardar unos minutos.")
        
        # Ejecutar el script de red neuronal
        subprocess.run(["python", "neuronal_network_ret.py"], check=True)
        
        # Verificar si se generó la imagen
        if os.path.exists("nn_optim.png"):
            # Mostrar la imagen generada
            show_full_size_image("nn_optim.png")
        else:
            messagebox.showerror("Error", "No se generó la imagen de resultados.")
    except Exception as e:
        messagebox.showerror("Error", f"Error al ejecutar la red neuronal: {e}")

def show_csv_results(csv_path):
    # Crear ventana para mostrar resultados
    result_window = ctk.CTkToplevel()
    result_window.title("Resultados de Predicción")
    result_window.geometry("600x400")
    result_window.attributes('-topmost', True)
    
    # Leer el CSV generado
    try:
        df = pd.read_csv(csv_path)
        
        # Crear frame para mostrar resultados
        frame = ctk.CTkFrame(result_window)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Título
        title_label = ctk.CTkLabel(frame, text="Resultados de Predicción", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Área de texto para mostrar los resultados
        result_text = ctk.CTkTextbox(frame, font=("Arial", 12))
        result_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Insertar los datos en formato tabular
        result_text.insert("1.0", df.to_string(index=False))
        result_text.configure(state="disabled")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error al cargar los resultados: {e}")

def open_prediction_window():
    pred_window = ctk.CTkToplevel()
    pred_window.geometry("400x300")
    pred_window.title("Predicción")
    pred_window.transient(app)
    pred_window.grab_set()
    pred_window.focus_set()
    pred_window.attributes('-topmost', True)  # Mantener ventana siempre al frente

    x = app.winfo_x() + (app.winfo_width() - 400) // 2
    y = app.winfo_y() + (app.winfo_height() - 300) // 2
    pred_window.geometry(f"400x300+{x}+{y}")

    frame_pred = ctk.CTkFrame(pred_window)
    frame_pred.pack(fill="both", expand=True, padx=10, pady=10)

    # Instrucciones
    instructions = ctk.CTkLabel(frame_pred, text="Ingrese valores para predicción (3 columnas por fila)")
    instructions.pack(pady=5)

    # Crear un frame para el área de texto
    text_frame = ctk.CTkFrame(frame_pred)
    text_frame.pack(fill="both", expand=True, padx=10, pady=5)

    # Área de texto para ingresar valores
    input_text = ctk.CTkTextbox(text_frame)
    input_text.pack(fill="both", expand=True, padx=5, pady=5)
    input_text.insert("1.0", "10 19 46\n20 30 50\n# Ingrese más filas según necesite, 3 valores por fila")

    # Función para procesar la predicción
    def start_prediction():
        try:
            # Obtener el texto y procesarlo
            text_content = input_text.get("1.0", "end-1c")
            lines = [line.strip() for line in text_content.split('\n') if line.strip() and not line.strip().startswith('#')]
            
            if not lines:
                messagebox.showerror("Error", "No se encontraron datos válidos para procesar.")
                return
            
            # Procesar cada línea para obtener los valores
            data_rows = []
            for i, line in enumerate(lines):
                values = line.split()
                if len(values) != 3:
                    messagebox.showerror("Error", f"La fila {i+1} no tiene 3 valores: {line}")
                    return
                
                try:
                    row_values = [float(val) for val in values]
                    data_rows.append(row_values)
                except ValueError:
                    messagebox.showerror("Error", f"Valores no numéricos en la fila {i+1}: {line}")
                    return
            
            # Crear archivo temporal con los datos de entrada
            temp_input_file = "input_data.csv"
            with open(temp_input_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['valor1', 'valor2', 'valor3'])
                for row in data_rows:
                    writer.writerow(row)
            
            # Ejecutar el script de predicción con el archivo de entrada
            messagebox.showinfo("Procesando", "Realizando predicciones. Esto puede tardar unos momentos.")
            subprocess.run(["python", "prediccion.py", temp_input_file], check=True)
            
            # Verificar si se generó el archivo de resultados
            results_file = "resultados_prediccion.csv"
            if os.path.exists(results_file):
                show_csv_results(results_file)
            else:
                messagebox.showerror("Error", "No se generó el archivo de resultados.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar los datos: {e}")

    # Botón para iniciar la predicción
    start_button = ctk.CTkButton(frame_pred, text="Iniciar Predicción", command=start_prediction)
    start_button.pack(pady=10)

app = ctk.CTk()
app.geometry("800x624")
app.title("FONIS SA22I0129")

# Frame izquierdo
frame_izq = ctk.CTkFrame(app, width=200)
frame_izq.grid(row=0, column=0, padx=10, pady=10, sticky="ns")

label_datos_iniciales = ctk.CTkLabel(frame_izq, text="Datos Iniciales", font=("Arial", 16))
label_datos_iniciales.pack(pady=10)

btn_datos_vih = ctk.CTkButton(frame_izq, text="Datos VIH", command=load_and_execute)
btn_datos_vih.pack(pady=10)

btn_config_rna = ctk.CTkButton(frame_izq, text="Correr Red Neuronal", command=run_neural_network)
btn_config_rna.pack(pady=10)

btn_correr_simulacion = ctk.CTkButton(frame_izq, text="Predicción", command=open_prediction_window)
btn_correr_simulacion.pack(pady=10)

# Frame central
frame_central = ctk.CTkFrame(app)
frame_central.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

app.grid_columnconfigure(1, weight=1)
app.grid_rowconfigure(0, weight=1)

texto_descripcion = "FONIS SA22I0129\n\nMonitoreo y predicción de la epidemia de sida en Chile empleando redes neuronales artificiales y datos de tendencias de búsqueda en Google."
texto_widget = ctk.CTkTextbox(frame_central, font=("Arial", 20), wrap="word")
texto_widget.pack(fill="both", padx=20, pady=20)
texto_widget.insert("1.0", texto_descripcion)
texto_widget.configure(state="disabled")

app.mainloop()
