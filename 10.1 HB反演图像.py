import numpy as np
import pandas as pd
from tkinter import filedialog
import tkinter as tk
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
import seaborn as sns
from osgeo import gdal
import os
import numpy as np
import pandas as pd
from tkinter import filedialog
import tkinter as tk
import time
from sklearn.linear_model import LinearRegression

class RSP:
    #def __init__(self,hh):  self.hh = hh

    #读图像文件
    def get_input_path_by_window(self,name):
        # creat a form
        root = tk.Tk()
        root.withdraw()
        # access path
        file_path = filedialog.askopenfilename(title='输入待处理影像(tif格式)'+'--'+str(name))
        return file_path

    """读取影像文件"""
    def read_img(self,filename):
        dataset=gdal.Open(filename)       #打开文件
        im_width = dataset.RasterXSize    #栅格矩阵的列数
        im_height = dataset.RasterYSize   #栅格矩阵的行数
        im_geotrans = dataset.GetGeoTransform()  #仿射矩阵
        im_proj = dataset.GetProjection() #地图投影信息
        im_data = dataset.ReadAsArray(0,0,im_width,im_height) #将数据写成数组，对应栅格矩阵
        del dataset
        return im_proj,im_geotrans,im_data

    def get_output_path_by_window(self):
        # creat a form
        root = tk.Tk()
        root.withdraw()
        # access path
        file_path = filedialog.askdirectory(title='save results')
        return file_path

    #写出文件，以写成tif为例
    def write_img(self,filename,im_proj,im_geotrans,im_data):
        #gdal数据类型包括
        #gdal.GDT_Byte,
        #gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        #gdal.GDT_Float32, gdal.GDT_Float64

        #判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        #判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1,im_data.shape
        #创建文件
        driver = gdal.GetDriverByName("GTiff")            #数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

        dataset.SetGeoTransform(im_geotrans)              #写入仿射变换参数
        dataset.SetProjection(im_proj)                    #写入投影

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  #写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(im_data[i])

        del dataset

def read_data_integration_click_path_2(name_s):
    # creat a form
    root = tk.Tk()
    root.withdraw()
    # access path
    file_path = filedialog.askopenfilename(title='输入' + str(name_s) + '参数文件')
    # reading dara
    data = pd.read_excel(file_path)
    col_name_ = np.array(data.columns)
    # the words in the upper left corner of the data
    sort_ = col_name_[0]
    # Target value, such as copper content
    target_value_ = np.array(data[sort_])
    # The number of rows, the number of samples
    sample_number_ = np.shape(target_value_)[0]
    features_ = data.drop(sort_, axis=1)
    # Column name, wavelength
    features_names_ = list(features_.columns)
    # spectral values
    features_ = np.array(features_)
    # Number of columns, such as the number of bands
    wavelength_number_ = np.shape(features_)[1]
    return col_name_, sort_, target_value_, sample_number_, features_, features_names_, wavelength_number_
def read_data_integration_click_path_1(name_s):
    # creat a form
    root = tk.Tk()
    root.withdraw()
    # access path
    file_path = filedialog.askopenfilename(title='输入' + str(name_s) + '光谱文件')
    # reading dara
    data = pd.read_excel(file_path)
    col_name_ = np.array(data.columns)
    # the words in the upper left corner of the data
    sort_ = col_name_[0]
    # Target value, such as copper content
    target_value_ = np.array(data[sort_])
    # The number of rows, the number of samples
    sample_number_ = np.shape(target_value_)[0]
    features_ = data.drop(sort_, axis=1)
    # Column name, wavelength
    features_names_ = list(features_.columns)
    # spectral values
    features_ = np.array(features_)
    # Number of columns, such as the number of bands
    wavelength_number_ = np.shape(features_)[1]
    return col_name_, sort_, target_value_, sample_number_, features_, features_names_, wavelength_number_
def read_data_integration_click_path(name_s):
    # creat a form
    root = tk.Tk()
    root.withdraw()
    # access path
    file_path = filedialog.askopenfilename(title='输入' + str(name_s) + '化验实测值文件')
    # reading dara
    data = pd.read_excel(file_path)
    col_name_ = np.array(data.columns)
    # the words in the upper left corner of the data
    sort_ = col_name_[0]
    # Target value, such as copper content
    target_value_ = np.array(data[sort_])
    # The number of rows, the number of samples
    sample_number_ = np.shape(target_value_)[0]
    features_ = data.drop(sort_, axis=1)
    # Column name, wavelength
    features_names_ = list(features_.columns)
    # spectral values
    features_ = np.array(features_)
    # Number of columns, such as the number of bands
    wavelength_number_ = np.shape(features_)[1]
    return col_name_, sort_, target_value_, sample_number_, features_, features_names_, wavelength_number_
def index_E(s,bm_value,bn_value):
    index_value_0 = bm_value + bn_value
    index_value_1 = bm_value - bn_value
    index_value_2 = bm_value * bn_value
    index_value_3 = bm_value / bn_value
    index_value_4 = (bm_value - bn_value)/(bm_value + bn_value)

    if s == 0:
        return index_value_0
    elif s ==1:
        return index_value_1
    elif s ==2:
        return index_value_2
    elif s ==3:
        return index_value_3
    elif s ==4:
        return index_value_4


if __name__ == '__main__':
    time_start = time.time()  # 开始时间
    """
    config = {
        "font.family": 'serif',
        "font.size": 10,#全图字号
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    }
    rcParams.update(config)
    # 图中字体参数
        """
    sample_type_name = np.array(["土壤", "植被", "水体"])#"土壤", "植被", "水体"
    """
    y_label_name =  np.array([['pH','Cr','Ni','Cu','Zn','As','Cd','Hg','Pb'],
                              ['SPAD', 'Cr', 'Ni', 'Cu', 'Zn', 'As', 'Cd', 'Hg', 'Pb'],
                              ['pH','BOD$\mathregular{_5}$','SS','CODcr','NO$\mathregular{^2}$$\mathregular{^-}$','TH','F$\mathregular{^-}$','SO$\mathregular{_4}$$\mathregular{^2}$$\mathregular{^-}$','AAAAA']])
    """

    run = RSP()

    """分别基于土、植被、水的影像，各做多次反演"""
    for i in range(0, 3):  # soil\vegetation\water
        name_s = sample_type_name[i]

        """写入rs data"""
        input_path = run.get_input_path_by_window(name_s)
        proj, geotrans, data = run.read_img(input_path)

        """写入par file"""
        col_name_par, sort_par, target_value_par, sample_number_par, features_par, features_names_par, wavelength_number_par = read_data_integration_click_path_2(
            name_s)

        """写出image path"""
        #output_path = run.get_output_path_by_window()#写出路径

        """基于一种影像，多次反演"""
        for ii in range(0,target_value_par.shape[0]):  # cu\pb\cr\cod........\....\........

            """影像前期处理"""
            data[data == 0] = 'nan'  # 数值置换
            #data = np.nan_to_num(data)  # nan换0
            data = data.astype(np.float)  # 转数据格式

            """波段组合数据准备"""
            Bm = int(features_par[ii, 0])#波段号
            Bn = int(features_par[ii, 1])#波段号

            bm_value= data[Bm-1]#波段值
            bn_value= data[Bn-1]#波段值

            s = features_par[ii,6]#构型代号

            index_value=index_E(s,bm_value,bn_value)#波段组合值 #X

            """反演参数准备"""
            k = features_par[ii,4]
            b =features_par[ii,5]
            y = k*index_value + b

            """小于0的换成固定值"""
            y[y < 0] = 0.000001  # 数值置换

            """写出，背景值为0,用于下一步计算"""
            y = np.nan_to_num(y)  # nan换0
            #outfile_name = output_path + str(i)+"."+str(ii)+str(sample_type_name[i])+"的"+str(target_value_par[ii])+"__GB"+'.tif'  #
            outfile_name = "S:/YE/"+ str(i)+"."+str(ii)+str(sample_type_name[i])+"的"+str(target_value_par[ii])+"_BV is zero_for next"+"__YE"+'.tif'  #
            #print("out path", outfile_name)
            run.write_img(outfile_name, proj, geotrans, y)  # 写为ndvi图像

            """写出，背景值为nan，用于出图"""
            y_s = y
            y_s[y_s == 0] = 'nan'  # 数值置换
            #outfile_name = output_path + str(i)+"."+str(ii)+str(sample_type_name[i])+"的"+str(target_value_par[ii])+"__GB"+'.tif'  #
            outfile_name = "S:/YE/"+ str(i)+"."+str(ii)+str(sample_type_name[i])+"的"+str(target_value_par[ii])+"_BV is nan_for plot"+"__YE"+'.tif'  #
            #print("out path", outfile_name)
            run.write_img(outfile_name, proj, geotrans, y_s)  # 写为ndvi图像

    time_end = time.time()
    print('totally cost:', (time_end - time_start) // 3600, 'h', ((time_end - time_start) % 3600) // 60, 'min',
          ((time_end - time_start) % 3600) % 60, 's')
