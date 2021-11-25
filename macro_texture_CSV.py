# -*- coding: utf-8 -*-
import os
import csv
import errno
import anisotropia as an
# =============================================================================
# Este script utiliza la biblioteca acoustical_parameters para calcular el 
# parámetro T20 en bandas de tercio de octava de todas las mediciones de la
# cámara reverberante, vacía y con muestra. Luego, organiza los parámetros 
# calculados en dos archivos .csv para su posterior análisis.
# =============================================================================

b = 1
os.chdir("DATA")

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred
            
silentremove("ETx.csv")
silentremove("Tx.csv")
silentremove("DBM.csv")


filedir = "../WAV"

counter = 0
for filename in os.listdir(filedir):
    print(filename)
    ETx, Tx, DBM = an.texture(filedir+"/"+filename, b)
    ETx.insert(0, filename)
    Tx.insert(0, filename)
    DBM.insert(0, filename)
    
    with open("ETx.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(ETx)

    with open("Tx.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(Tx)

    with open("DBM.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(DBM)

