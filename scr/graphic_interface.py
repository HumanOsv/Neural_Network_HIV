import os
import subprocess
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk

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
    
    # Configurar tamaño inicial
    photo = ImageTk.PhotoImage(img)
    
    # Crear label para mostrar la imagen
    label = ctk.CTkLabel(image_window, image=photo)
    label.image = photo
    label.pack(padx=10, pady=10)
    
    # Vincular evento de rueda del ratón
    image_window.bind("<MouseWheel>", zoom)
    
    # Botón para cerrar
    close_button = ctk.CTkButton(image_window, text="Cerrar", command=image_window.destroy)
    close_button.pack(pady=10)
    
    # Centrar ventana
    screen_width = image_window.winfo_screenwidth()
    screen_height = image_window.winfo_screenheight()
    x = (screen_width - original_width) // 2
    y = (screen_height - original_height) // 2
    image_window.geometry(f"+{x}+{y}")

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

        # Crear labels para las imágenes
        global total_cases_label, yearly_cases_label
        total_cases_label = ctk.CTkLabel(graphs_frame)
        total_cases_label.pack(pady=10)
        
        # Agregar eventos de clic para cada gráfico
        total_cases_label.bind("<Button-1>", lambda e: show_full_size_image("total_cases.png"))
        
        yearly_cases_label = ctk.CTkLabel(graphs_frame)
        yearly_cases_label.pack(pady=10)
        yearly_cases_label.bind("<Button-1>", lambda e: show_full_size_image("yearly_cases.png"))

        # Función para redimensionar
        def resize_graphs(event):
            frame_width = graphs_frame.winfo_width()
            frame_height = graphs_frame.winfo_height()
            
            # Calcular nuevas dimensiones manteniendo la proporción
            new_width = frame_width - 40
            new_height = (frame_height - 60) // 2
            
            # Redimensionar y actualizar imágenes
            resized_total = ImageTk.PhotoImage(total_cases_img.resize((new_width, new_height)))
            resized_yearly = ImageTk.PhotoImage(yearly_cases_img.resize((new_width, new_height)))
            
            total_cases_label.configure(image=resized_total)
            total_cases_label.image = resized_total
            
            yearly_cases_label.configure(image=resized_yearly)
            yearly_cases_label.image = resized_yearly

            # Cambiar el cursor para indicar que es clickeable
            total_cases_label.configure(cursor="hand2")
            yearly_cases_label.configure(cursor="hand2")

        # Vincular evento de redimensionamiento
        graphs_frame.bind("<Configure>", resize_graphs)

    except Exception as e:
        print(f"Error mostrando gráficos: {e}")	

def open_config_window():
    config_window = ctk.CTkToplevel()
    config_window.geometry("400x300")
    config_window.title("Configuración Red Neuronal")
    config_window.transient(app)
    config_window.grab_set()
    config_window.focus_set()

    x = app.winfo_x() + (app.winfo_width() - 400) // 2
    y = app.winfo_y() + (app.winfo_height() - 300) // 2
    config_window.geometry(f"400x300+{x}+{y}")

    frame_config = ctk.CTkFrame(config_window)
    frame_config.pack(fill="both", expand=True, padx=10, pady=10)

    cargar_datos_btn = ctk.CTkButton(frame_config, text="Cargar Datos")
    cargar_datos_btn.pack(pady=10)

    introducir_capas_label = ctk.CTkLabel(frame_config, text="Introducir Capas de la Red")
    introducir_capas_label.pack(pady=10)

    introducir_capas_entry = ctk.CTkEntry(frame_config)
    introducir_capas_entry.pack(pady=10)

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

btn_config_rna = ctk.CTkButton(frame_izq, text="Configuración Red Neuronal", command=open_config_window)
btn_config_rna.pack(pady=10)

btn_correr_simulacion = ctk.CTkButton(frame_izq, text="Correr Simulación")
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