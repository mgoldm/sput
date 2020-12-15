import os
import cv2
import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
os.listdir('../sputnik/')

def get_coord(word):
    inp = open('LE07_L1TP_201024_20010113_20170208_01_T1_MTL.txt').readlines()
    for i in iter(inp):
        if word in i:
            original = i[4:]
            return float(original.replace(word + " = ", ""))



def liner_to_center(p1, p2, kof):
    l = kof / (1 - kof)
    if kof <= 0.5:
        xm = (p1[0] + l * p2[0]) / (1 + l)
        ym = (p1[1] + l * p2[1]) / (1 + l)
        M = xm, ym
        return M
    else:
        l = 1 / l
        xm = (p2[0] + l * p1[0]) / (1 + l)
        ym = (p2[1] + l * p1[1]) / (1 + l)
        M = xm, ym
        return M

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

if __name__ == '__main__':
    a=int(input('Введите номер лабораторной работы'))
    if a==1:
        E = 0, 1200  # погрешность
        k1 = 1896, 15
        k2 = 8063, 1699
        k3 = 6203, 7550
        k4 = 39, 5863

        CORNER_UL_LAT_PRODUCT = get_coord('CORNER_UL_LAT_PRODUCT')
        CORNER_UL_LON_PRODUCT = get_coord('CORNER_UL_LON_PRODUCT')
        CORNER_UR_LAT_PRODUCT = get_coord('CORNER_UR_LAT_PRODUCT')
        CORNER_UR_LON_PRODUCT = get_coord('CORNER_UR_LON_PRODUCT')
        CORNER_LL_LAT_PRODUCT = get_coord('CORNER_LL_LAT_PRODUCT')
        CORNER_LL_LON_PRODUCT = get_coord('CORNER_LL_LON_PRODUCT')
        CORNER_LR_LAT_PRODUCT = get_coord('CORNER_LR_LAT_PRODUCT')
        CORNER_LR_LON_PRODUCT = get_coord('CORNER_LR_LON_PRODUCT')
        TOWNS_COORDS = 51.50853, -0.1257400

        delt_lat = ((CORNER_UL_LAT_PRODUCT + CORNER_UR_LAT_PRODUCT) / 2 - (CORNER_LL_LAT_PRODUCT + CORNER_LR_LAT_PRODUCT) / 2)
        delt_lon = abs((CORNER_UL_LON_PRODUCT + CORNER_UR_LON_PRODUCT) / 2 - (CORNER_LL_LON_PRODUCT + CORNER_LR_LON_PRODUCT) / 2)
        print('Разница:', delt_lat, delt_lon)

        z_x = (((CORNER_UL_LON_PRODUCT - (CORNER_LL_LON_PRODUCT)) / 2) - (TOWNS_COORDS[1])) / delt_lon
        z_y = (TOWNS_COORDS[0] - (CORNER_LR_LAT_PRODUCT + CORNER_LL_LAT_PRODUCT) / 2) / delt_lat
        print('Z: ', z_x, z_y)
        line1 = liner_to_center(k1, k2, z_x)
        line2 = liner_to_center(k4, k1, z_y)
        print(line1, line2)

        img = cv2.imread("LE07_L1TP_201024_20010113_20170208_01_T1_B2.tif")
        img = cv2.line(img, (k1[0], k1[1]), (int(line1[0]), int(line1[1])), (255, 0, 0), 10)
        img = cv2.line(img, (k4[0], k4[1]), (int(line2[0]), int(line2[1])), (255, 0, 0), 10)
        img = cv2.circle(img, (int(line1[0]), int(line2[1] + E[1])), 500, (255, 0, 0), thickness=10)
        show_result(img)
    elif a==2:
        # открываем тиф файлы в виде матрицы
        img1 = rasterio.open('../sputnik/LE07_L1TP_201024_20010113_20170208_01_T1_B3.tif')
        img2 = rasterio.open('../sputnik/LE07_L1TP_201024_20010113_20170208_01_T1_B4.tif')
        # переводим целочисленные данные в данные с плавающей точкой для точности подсчета
        red = img1.read(1).astype('float64')
        nir = img2.read(1).astype('float64')
        # расчет индекса ndvi
        ndvi = np.where(nir + red == 0., 0, (nir - red) / (nir + red))
        # задаем размер изображения
        fig = plt.figure(figsize=(10, 8))
        # задаем палитру цветов для гео снимка
        tk = sns.heatmap(ndvi, cmap="gist_earth", vmin=-1, vmax=1, xticklabels=False, yticklabels=False)
        # название окна
        tk.set_title('tepl karta')

        tk.set_axis_off()
        plt.tight_layout()
        plt.show()


