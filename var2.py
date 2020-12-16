import os
import rasterio
import cv2
from rasterio import plot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

os.listdir('../sputnik/')  # даем программе доступ к определенной папке для упрощения ввода


def show_result(result):
    screen_res = 1280, 720
    scale_width = screen_res[0] / result.shape[1]
    scale_height = screen_res[1] / result.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(result.shape[1] * scale)
    window_height = int(result.shape[0] * scale)

    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt', window_width, window_height)

    cv2.imshow('dst_rt', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ndvi_ppros(red1, nir1):
    ras = red1 - nir1
    sum = red1 + nir1
    ndvi = ras / sum
    show_result(ndvi)


if __name__ == '__main__':
    a = cv2.imread("LE07_L1TP_201024_20010113_20170208_01_T1_B3.tif")
    b = cv2.imread("LE07_L1TP_201024_20010113_20170208_01_T1_B4.tif")
    # ndvi_ppros(a,b)

    # открываем тиф файлы в виде матриц

    img1 = rasterio.open('../sputnik/LE07_L1TP_201024_20010113_20170208_01_T1_B3.tif')
    img2 = rasterio.open('../sputnik/LE07_L1TP_201024_20010113_20170208_01_T1_B4.tif')
    # переводим целочисленные данные в данные с плавающей точкой для точности подсчета

    red = img1.read(1).astype('float64')
    nir = img2.read(1).astype('float64')

    # расчет индекса ndvi

    ndvi = np.where(nir + red == 0., 0, (nir - red) / (nir + red))
    # задаем размер изображения

    fig = plt.figure(figsize=(10, 8))

    # накладываем на снимок палитру "gist_earth"

    tk = sns.heatmap(ndvi, cmap="gist_earth", vmin=-1, vmax=1, xticklabels=False, yticklabels=False)

    # название окна

    tk.set_title('tepl karta')

    tk.set_axis_off()
    plt.tight_layout()
    plt.show()
