import customtkinter as ctk
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
#import pandas as pd
import csv
import os.path
from tkinter import filedialog
import json
from ctktable import CTkTable

app = ctk.CTk()

app.geometry("1024x768")

frame_izq = ctk.CTkFrame(app, border_width=2)
frame_izq.grid(row=0, column=0, padx=5, pady=5)

label_f1=ctk.CTkLabel(frame_izq, text="Carga de Datos", fg_color="gray30", corner_radius=6)
label_f1.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

frame_der = ctk.CTkFrame(app, border_width=3, height=600)
frame_der.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

texto_intro = ctk.CTkTextbox(frame_der, width=600)
texto_intro.grid(row=0, column=0, padx=5, pady=10, sticky="ew")
texto_intro.insert("0.0", "Aquí irá una breve introducción de la App")

def graficar_datos(datos):
    pass

def salir_datos():
    frame_botones_datos.destroy()
    frame_der.destroy()


frame_botones_datos = ctk.CTkFrame(frame_der, fg_color="gray30")
graficar_datos = ctk.CTkButton(frame_botones_datos, text="Graficar", command=graficar_datos)
graficar_datos.grid(row=0, column=0, padx=10, pady=5)
salir_datos = ctk.CTkButton(frame_botones_datos, text="Salir", command=salir_datos)
salir_datos.grid(row=0, column=1, padx=10, pady=5)

def crea_tabla_datos(datos):
    tabla = CTkTable(frame_der, row=5, column=5, values=datos)
    return tabla

def opendatafiles():
    print("abre archivos de entrada")
    nombrearchivo = filedialog.askopenfilename(initialdir=".",
                                               title="Seleccione un archivo",
                                               filetypes=(("CSV", "*.csv"),
                                                          ("XLSX", "*.xlsx")))

def cerrar_app():
    print("Cerrando aplicación")
    app.destroy()

def open_data():
    print("abre archivos de entrada")
    nombrearchivo = filedialog.askopenfilename(initialdir=".",
                                               title="Seleccione un archivo",
                                               filetypes=(("CSV", "*.csv"),
                                                          ("XLSX", "*.xlsx")))
    archivo, extension = os.path.splitext(nombrearchivo)
    print("La extension es" + extension)
    x = []
    y = []
    if extension == ".csv":
        texto_intro.destroy()       # ESTO HAY QUE REVISARLO (MEJORARLO)
        print("CSV")
        with open(nombrearchivo) as csvfile:
            datos_csv = csv.reader(csvfile, delimiter = ';')
            datos = [fila for fila in datos_csv]
            tabla = crea_tabla_datos(datos)
            tabla.grid(row=1, column=0)
            frame_botones_datos.grid(row=0, column=0)

            """
            for row in datos:
                #print(', '.join(row))
                x.append(row[0])
                y.append(row[2])
            #fig = plt.figure(10,5)
            X = x[1:]
            Y = y[1:]
            xmax = max(X)
            ymax = max(Y)
            print(xmax + " "+ymax)
            """
        
            
           

def open_rna():
    print("Cargando datos de RNA")
    nombrearchivo = filedialog.askopenfilename(initialdir=".",
                                               title="Seleccione un archivo",
                                               filetypes=(('JSON', '*.json'),))

        


opendf = ctk.CTkButton(frame_izq, text="Cargar archivos de entrada", command=opendatafiles)
opendf.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

opendata = ctk.CTkButton(frame_izq, text="Cargar datos de simulación", command=open_data)
opendata.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

salir = ctk.CTkButton(frame_izq, text="Cerrar Aplicación", command=cerrar_app, fg_color="red")
salir.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
#salir.place(relx=0.5, rely=0.8, anchor=ctk.CENTER)




frame_rna = ctk.CTkFrame(app, border_width=2)

frame_rna.grid(row=1, column=0, padx=5, pady=10)
label_rna=ctk.CTkLabel(frame_rna, text="Opciones de RNA", fg_color="gray30", corner_radius=6)
label_rna.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

openrna = ctk.CTkButton(frame_rna, text="Cargar Red neuronal", command=open_rna)
openrna.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
#openrna.place(relx=0.5, rely=0.2, anchor=ctk.CENTER)

app.mainloop()


"""
def open_data():
    print("abre archivos de entrada")
    nombrearchivo = filedialog.askopenfilename(initialdir=".",
                                               title="Seleccione un archivo",
                                               filetypes=(("CSV", "*.csv"),
                                                          ("XLSX", "*.xlsx")))
    archivo, extension = os.path.splitext(nombrearchivo)
    print("La extension es" + extension)
    x = []
    y = []
    if extension == ".csv":
        print("CSV")
        with open(nombrearchivo) as csvfile:
            datos = csv.reader(csvfile, delimiter = ';')
            for row in datos:
                #print(', '.join(row))
                x.append(row[0])
                y.append(row[2])
            #fig = plt.figure(10,5)
            X = x[1:]
            Y = y[1:]
            xmax = max(X)
            ymax = max(Y)
            print(xmax + " "+ymax)
            
            fig, ax = plt.subplots()

            canvas = FigureCanvasTkAgg(fig, master = frame_der)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            ax.scatter(x[1:],y[1:])
            ax.set_xlim(0, float(xmax)+5.0)
            ax.set_ylim(0, float(ymax)+10.0)
            canvas.draw()
"""