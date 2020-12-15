import cv2
import os
import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
os.listdir('../sputnik/') # даем программе доступ к определенной папке для упрощения ввода

if __name__ == '__main__':
    #открываем тиф файлы в виде матрицы
    img1 = rasterio.open('../sputnik/LE07_L1TP_201024_20010113_20170208_01_T1_B3.tif')
    img2= rasterio.open('../sputnik/LE07_L1TP_201024_20010113_20170208_01_T1_B4.tif')
    #переводим целочисленные данные в данные с плавающей точкой для точности подсчета
    red=img1.read(1).astype('float64')
    nir=img2.read(1).astype('float64')
    #расчет индекса ndvi
    ndvi=np.where(nir+red==0., 0, (nir-red)/(nir+red))
    #задаем размер изображения
    fig = plt.figure(figsize=(10, 8))
    #задаем палитру цветов для гео снимка
    tk = sns.heatmap(ndvi, cmap="gist_earth", vmin=-1, vmax=1, xticklabels=False, yticklabels=False)
    #название окна
    tk.set_title('tepl karta')

    tk.set_axis_off()
    plt.tight_layout()
    plt.show()