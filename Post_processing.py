# -*- coding: utf-8 -*-


import os
from PIL import Image
import cv2
import numpy as np
from scipy import stats
from tqdm import tqdm
from scipy.signal import savgol_filter
import time
import tifffile as tiff
from osgeo import gdal
from multiprocessing import Pool, cpu_count
import math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def calculate_entropy(img):
    histogram = np.histogram(img[:, :, 0].flatten(), 256, [0, 255])
    prob = histogram[0] / (img.shape[0] * img.shape[1])
    entropy = -np.sum(prob * np.log2(prob + 1e-5))

    histogram1 = np.histogram(img[:, :, 1].flatten(), 256, [0, 255])
    prob = histogram1[0] / (img.shape[0] * img.shape[1])
    entropy += -np.sum(prob * np.log2(prob + 1e-5))

    histogram2 = np.histogram(img[:, :, 2].flatten(), 256, [0, 255])
    prob = histogram2[0] / (img.shape[0] * img.shape[1])
    entropy += -np.sum(prob * np.log2(prob + 1e-5))
    return entropy


def STD_select(image_path):
    signal_unique = 0
    signal_entropy = 0
    img_name_list = []
    image_index = []
    # for image in os.listdir(image_path):
    for image in tqdm(os.listdir(image_path), desc='从深度学习结果中筛选最优影像'):
        _, endwith = os.path.splitext(image)
        if endwith != '.tif':
            continue
        img_path = os.path.join(image_path, image)
        img_hist = cv2.imread(img_path)
        unique_list = []
        total_unique = 0
        for i in range(3):
            img_8band = img_hist[:, :, i:i + 1]
            # todo 灰度值个数
            unique_values = np.unique(img_8band)
            unique_list.append(len(unique_values))
            total_unique += len(unique_values)
        if total_unique > signal_unique or total_unique >= 760:
            # img_name_list.append(image)
            # if signal_unique < 760:
            # signal_unique = total_unique
            img_name_list.clear()
            img_name_list.append(image)
        elif total_unique == signal_unique:
            img_name_list.append(image)
    if len(img_name_list) > 1:
        # for index in range(len(img_name_list)):
        signal_entropy = 0
        for index in tqdm(range(len(img_name_list))):
            entropy_image = os.path.join(image_path, img_name_list[index])
            img_hist = cv2.imread(entropy_image)
            img_entropy = calculate_entropy(img_hist)
            if img_entropy > signal_entropy:
                signal_unique = img_entropy
                image_index.clear()
                image_index.append(img_name_list[index])
            elif img_entropy == signal_entropy:
                image_index.append(img_name_list[index])
        print("熵：{}, 灰度值总数：{}".format(signal_entropy, signal_unique))
        print("----------------------------------------------------------------------------------------\n")
        for i in range(len(image_index)):
            print("图名：{}".format(image_index[i]))
        return img_name_list
    elif len(img_name_list) == 1:
        entropy_image = os.path.join(image_path, img_name_list[0])
        img_hist = cv2.imread(entropy_image)
        img_entropy = calculate_entropy(img_hist)
        print("熵：{}, 灰度值总数：{}".format(img_entropy, signal_unique))
        print("----------------------------------------------------------------------------------------\n")
        print("图名：{}".format(img_name_list[0]))
        return img_name_list
    else:
        print("代码写错了！！！！！！！！！！！！！！！")


def sMax(img):
    max_val = np.max(img)
    return int(max_val)


def Unit16_number(image_path):
    max = 0
    for image in tqdm(os.listdir(image_path), desc='16位影像数量'):
        _, endwith = os.path.splitext(image)
        if endwith != '.tif':
            continue
        max += 1
    return max


def Unit8_number(image_path):
    max = 0
    for image in tqdm(os.listdir(image_path), desc='8位影像数量'):
        _, endwith = os.path.splitext(image)
        if endwith != '.tif':
            continue
        max += 1
    return max


def Unit16_max_select(image_path):
    maxArray = []
    for image in tqdm(os.listdir(image_path), desc='16位影像处理_求最大值'):
        _, endwith = os.path.splitext(image)
        if endwith != '.tif':
            continue
        img_path = os.path.join(image_path, image)
        ds = gdal.OpenShared(img_path)
        for i in range(1, 4):
            bandArr = ds.GetRasterBand(i).ReadAsArray().astype(np.uint16)
            # 追加最大值
            maxArray.append(np.max(bandArr))
        ds = None
    return np.max(maxArray)


def Unit8_max_select(image_path):
    maxArray = []
    for image in tqdm(os.listdir(image_path), desc='从8位影像中筛选最大值'):
        _, endwith = os.path.splitext(image)
        if endwith != '.tif':
            continue
        img_path = os.path.join(image_path, image)

        ds = gdal.OpenShared(img_path)
        for i in range(1, 4):
            bandArr = ds.GetRasterBand(i).ReadAsArray().astype(np.uint8)
            # 追加最大值
            maxArray.append(np.max(bandArr))
        ds = None
    return np.max(maxArray)


def process_image_batch(args):
    """处理一批图像的直方图计算"""
    image_paths, max_value = args
    batch_hist = np.zeros((3, max_value))

    for img_path in image_paths:
        # 读取并处理图像（切换到gdal读取更合适）
        img = tiff.imread(img_path)[:, :, :3].astype(np.uint16)

        # 计算每个通道的直方图
        for i in range(3):
            hist = np.histogram(img[:, :, i], bins=max_value, range=(0, max_value))[0]
            batch_hist[i] += hist

    return batch_hist


def Unit16_imhist(image_path, max_value, batch_size=100):
    # 获取所有.tif文件路径
    image_files = [
        os.path.join(image_path, f)
        for f in os.listdir(image_path)
        if f.endswith('.tif')
    ]

    # 将文件列表分成批次
    total_files = len(image_files)
    num_batches = math.ceil(total_files / batch_size)
    batches = [
        image_files[i * batch_size:(i + 1) * batch_size]
        for i in range(num_batches)
    ]

    # 准备进程池和参数
    n_cores = max(1, cpu_count() - 1)  # 保留一个核心给系统
    pool = Pool(processes=n_cores)
    print(f'= {n_cores} cores)')
    # 准备批次参数
    args_list = [(batch, max_value) for batch in batches]

    # 使用tqdm显示进度
    results = list(tqdm(
        pool.imap(process_image_batch, args_list),
        total=len(args_list),
        desc=f'并行处理直方图统计 ({n_cores} cores)'
    ))

    # 关闭进程池
    pool.close()
    pool.join()

    # 合并所有批次结果
    imhist_16 = np.sum(results, axis=0, dtype=np.uint32)

    return imhist_16


def Unit8_imhist(image_path, max_value):
    imhist_8 = np.zeros((3, max_value))
    for image in tqdm(os.listdir(image_path), desc='从8位影像中统计直方图'):
        _, endwith = os.path.splitext(image)
        if endwith != '.tif':
            continue
        img_path = os.path.join(image_path, image)

        ds = gdal.OpenShared(img_path)
        red_channel1 = ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
        green_channel1 = ds.GetRasterBand(2).ReadAsArray().astype(np.uint8)
        blue_channel1 = ds.GetRasterBand(3).ReadAsArray().astype(np.uint8)
        ds = None

        red_histogram = np.histogram(red_channel1.flatten(), max_value, [0, max_value - 1])
        green_histogram = np.histogram(green_channel1.flatten(), max_value, [0, max_value - 1])
        blue_histogram = np.histogram(blue_channel1.flatten(), max_value, [0, max_value - 1])

        imhist_8[0, :] += red_histogram[0]
        imhist_8[1, :] += green_histogram[0]
        imhist_8[2, :] += blue_histogram[0]

    return imhist_8


def Th_16(na, pixel_num, im_tgt_max):
    B = np.zeros((3, im_tgt_max))  # todo 256--->max
    Sa = np.zeros((3, im_tgt_max))  # todo 256--->max

    threshold = np.zeros(3)
    for c in range(3):
        index = np.argsort(na[c, :])  # np.argsort：返回排序索引
        B[c, :] = np.sort(na[c, :])
        Sa[c, 0] = B[c, 0]
        for n in range(1, im_tgt_max):
            Sa[c, n] = Sa[c, n - 1] + B[c, n]

        number = 0
        while (Sa[c, number] < pixel_num / 200):  # 小于总数量的二十分之一
            number += 1
        threshold[c] = na[c, index[number]]
    return threshold


def Th_8(na, pixel_num):
    B = np.zeros((3, 256))
    Sa = np.zeros((3, 256))
    threshold = np.zeros(3)

    for c in range(3):
        index = np.argsort(na[c, :])  # np.argsort：返回排序索引
        B[c, :] = np.sort(na[c, :])
        Sa[c, 0] = B[c, 0]
        for n in range(1, 256):
            Sa[c, n] = Sa[c, n - 1] + B[c, n]
        number = 0
        while (Sa[c, number] < pixel_num / 20):  # 小于总数量的二十分之一
            number += 1
        threshold[c] = na[c, index[number]]  # 像素值出现的频率
    return threshold


def HHM(na, nb, na_max, nb_max, number):
    # W, H = im_src.size
    W = 512
    H = 512

    pixel_num = H * W * number
    # im_tgt = im_tgt.resize([W, H])
    MAX = na_max

    x = [i for i in range(MAX)]

    Sa = np.zeros((3, na_max))
    # Sa = [[0 for j in range(MAX)] for i in range(3)]

    # nb = [[0 for j in range(256)] for i in range(3)]
    Sb = np.zeros((3, nb_max))

    m_down_trunc = 0.001 * pixel_num  # // 下方截断比例
    m_up_trunc = pixel_num - m_down_trunc  # // 上方截断比例

    threshold_high_index = np.zeros(3)
    threshold_low_index = np.zeros(3)

    Sa[:, 0] = na[:, 0]  # 累加直方图
    Sb[:, 0] = nb[:, 0]
    for n in range(1, na_max):
        Sa[:, n] = Sa[:, n - 1] + na[:, n]
        for m in range(3):
            if Sa[m, n] > m_down_trunc and threshold_low_index[m] == 0:
                threshold_low_index[m] = n
            if Sa[m, n] > m_up_trunc and threshold_high_index[m] == 0:
                threshold_high_index[m] = n
    for n in range(1, nb_max):
        Sb[:, n] = Sb[:, n - 1] + nb[:, n]
    start1 = time.time()
    # build map
    mapp = np.array([x, x, x]).astype(float)  # mapp-->(3, MAX)
    index = np.ones((3, MAX))  # 此处为imtgt的索引，所以256--->MAX

    for c in range(3):
        gradient = np.zeros((c + 1, MAX + 1))
        srcMax = 254  # np.max(np.array(im_src)[:, :, c])+1  # 最大值加一  0~255 所以加一
        srcMin = 0  # np.min(np.array(im_src)[:, :, c])+1  # 最小值加一

        for a in range(MAX):
            b = 0
            while Sa[c, a] > Sb[c, b]:  # 以Sa[c, a]（16）为基准，寻找Sb[c, b]（8）对应值
                b += 1
                if b > srcMax:  # 如果Sa[c, a]对应Sb[c, b]最大值
                    b = srcMax
                    break
            if b < srcMin:  # 如果Sa[c, a]对应Sb[c, b]最小值
                b = srcMin
            mapp[c, a] = b + 1  # mapp[c,a]存储对应值

        mapp[c, 0] = srcMin  # todo ?
        mapp[c, MAX - 1] = srcMax
        index[c, 0] = 1
        index[c, MAX - 1] = 1


    gradient[:, 1:MAX] = index[:, 1:MAX] - index[:, 0:MAX - 1]
    gradient[:, 0] = 0
    print("全局映射", time.time() - start1)
    region = np.zeros((3, MAX, 2)).astype(int)
    XX = np.zeros((3, MAX, 2)).astype(int)
    YY = np.zeros((3, MAX, 2)).astype(int)
    start2 = time.time()

    for c in range(3):
        n_re = -1
        for a in range(MAX):
            if gradient[c, a] == -1:
                n_re += 1
                region[c, n_re, 0] = a
            elif gradient[c, a] == 1:
                region[c, n_re, 1] = a - 1
        for num in range(n_re + 1):
            XX[c, num, 0] = region[c, num, 0] - 1
            YY[c, num, 0] = mapp[c, XX[c, num, 0]]
            XX[c, num, 1] = region[c, num, 1] + 1
            YY[c, num, 1] = mapp[c, XX[c, num, 1]]

            p = np.polyfit(XX[c, num, :], YY[c, num, :], 1)
            for a in range(region[c, num, 0], region[c, num, 1] + 1):
                mapp[c, a] = np.polyval(p, a)
    print("局部映射：", time.time() - start2)
    start3 = time.time()
    x = [i for i in range(MAX)]
    mapp[0, :] = savgol_filter(mapp[0, x], 15, 1)
    mapp[1, :] = savgol_filter(mapp[1, x], 15, 1)
    mapp[2, :] = savgol_filter(mapp[2, x], 15, 1)
    # for c in range(3):
    #     x = [i for i in range(image_16_max_list[c])]
    #     mapp[c, :] = savgol_filter(mapp[0, x], 15, 1)

    print("平滑直方图", time.time() - start3)
    return mapp


def PA(mapp, input):
    w = input.shape[1]
    h = input.shape[0]
    im = np.array(input)
    for c in range(3):
        im[0:h, 0:w, c] = mapp[c, im[0:h, 0:w, c]] - 1

    im1 = im.astype(np.uint8)

    return Image.fromarray(im1)


def LutProcess(lutmap, src_filePath, dst_filePath):
    """
    进行图像映射

    :param lutmap: 映射表
    :param src_filePath: 待映射的16位图像路径
    :param dst_filePath: 映射后的8位图像路径

    """
    # 打开输出的16位图像
    src_ds = gdal.OpenShared(src_filePath)
    driver = gdal.GetDriverByName('GTiff')
    # 创建输出图像
    dst_ds = driver.Create(dst_filePath, src_ds.RasterXSize, src_ds.RasterYSize, 3, gdal.GDT_Byte)
    dst_ds.SetProjection(src_ds.GetProjection())
    dst_ds.SetGeoTransform(src_ds.GetGeoTransform())

    for i in range(1, 4):
        bandarr = src_ds.GetRasterBand(i).ReadAsArray().astype(np.uint16)
        bandarr = np.take(lutmap[i - 1], bandarr)
        dst_ds.GetRasterBand(i).WriteArray(bandarr.astype(np.uint8))
    src_ds = None
    dst_ds = None

import psutil
import os

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # 返回MB


if __name__ == '__main__':

    # 1.设置路径
    file_image_16_part = r''  # 16位小图图像文件夹
    image_16 = r"" # 待映射的16位大图
    output_save = r""  # 本地保存的路径
    file_image_deeplearning = r''  # 深度学习结果文件夹

    start = time.time()
    mem_before = get_memory_usage_mb()
    image_16_num = Unit16_number(file_image_16_part)
    image_8_num = Unit8_number(file_image_deeplearning)
    if image_16_num != image_8_num:
        print("数据不对等！！！！！！！！！！！！！！！")

    image_16_max = Unit16_max_select(file_image_16_part) + 1
    imhist_16 = Unit16_imhist(file_image_16_part, image_16_max)

    imhist_8 = Unit8_imhist(file_image_deeplearning, 256)

    print('开始计算映射')
    mapp = HHM(imhist_16, imhist_8, image_16_max, 256, image_16_num)

    print(mapp.shape)
    save_file = os.path.join(os.path.dirname(output_save), 'mapping.npy')
    np.save(save_file, mapp)
    print('开始读取图像')
    LutProcess(mapp, image_16, output_save)
    mem_after = get_memory_usage_mb()
    mem_diff = mem_after - mem_before

    print(f"Memory usage increased by: {mem_diff:.2f} MB")
    print('一共时间耗费: ', time.time() - start)
