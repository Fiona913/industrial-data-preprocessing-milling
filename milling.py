# 注意：需先下载NASA铣削数据集（mill.mat），放在当前项目根目录
# 数据集下载地址：https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 数据列含义：
# case	       实验案例编号（区分不同的铣削工艺方案，比如不同的刀具类型、设备型号）
# run	       实验运行编号（同case下的多次重复实验，用于验证数据稳定性）
# VB	       铣削刀片磨损量（核心标签，单位 mm，是预测的目标，对应 “刀具剩余寿命” 场景）
# time	       实验运行时间（记录从加工开始到当前时刻的时长，用于分析 “磨损随时间的变化规律”）
# DOC	       切削深度（工艺参数，决定刀具吃刀量，是影响磨损的关键因素之一）
# feed	       进工艺参数，刀具每分钟移动的距离，直接影响加工效率和刀具负荷）
# material	   加工材料（比如钢材、铝材，不同材料的硬度会导致刀具磨损差异）
# smcAC	       交流主轴电机电流（传感器数据，反映主轴负载，负载高则磨损快）
# smcDC	       直流主轴电机电流（传感器数据，补充主轴负载的监测维度）
# vib_table	   工作台振动（x/y/z 轴振动传感器数据，振动大说明刀具或设备有异常磨损）
# vib_spindle  主轴振动（传感器数据，主轴是刀具的直接载体，振动异常是磨损的核心信号）
# AE_table	   工作台声发射信号（声发射是材料损伤的声学信号，用于早期故障预警）
# AE_spindle   主轴声发射信号（同理，主轴的声发射信号直接反映刀具磨损状态）


# ---------------------- Step1：数据加载与读取 ----------------------
# 一、读取数据（不是csv形式，不能直接用numpy读取，用scipy.io）
# 读取mat文件
mat_data = scipy.io.loadmat('./mill.mat') #需要替换成自己存储的路径
# 生成的 mat_data 是一个字典,字典的键就是下方输出的“变量名（keys()）”
# 查看文件中的变量名
print("数据中的变量名：", mat_data.keys())
# mat_data['mill']中的mill就是其中的一个键
mill_data = mat_data['mill']

# 二、查看数据基本信息
print("数据形状：", mill_data.shape)
# print("前5行数据：\n", mill_data[:5])
print("结构体的所有字段名：", mill_data[0:1].dtype.names)
# print("各列缺失值数量：", np.isnan(mill_data).sum(axis=0))  # 查看每列缺失值（工业数据常见问题）

# 三、拆解.mat文件的复杂结构
all_samples=[]
for i in range(167):
    # 把每一列数据取出来（一行一列，所以就相当于每格的数据）
    cell=mill_data[0,i]

    # 部分单元格中字段无数据，会导致索引越界，所以要做一个判断：
    # hasattr(cell, 'dtype')是判断cell中是否有'dtype'属性（一般VB、case都有dtype属性）
    # if hasattr(cell, 'dtype') and cell.dtype.names is not None and 'case' in cell.dtype.names:

    case=cell['case'][0, 0].astype(float)  # 提取后转成float，和其他特征类型对齐
    run=cell['run'][0, 0].astype(float)
    VB = cell['VB'][0, 0]
    time = cell['time'][0, 0]
    DOC = cell['DOC'][0, 0]
    feed = cell['feed'][0, 0]
    material = cell['material'][0, 0]
    # 以下数据取平均值，把 “一串数据” 变成 “一个特征值”,降低处理难度（以后可在这个地方做优化）
    smcAC = cell['smcAC'].mean()
    smcDC = cell['smcDC'].mean()
    vib_table = cell['vib_table'].mean()
    vib_spindle = cell['vib_spindle'].mean()
    AE_table = cell['AE_table'].mean()
    AE_spindle = cell['AE_spindle'].mean()

    sample=[DOC, feed, smcAC, smcDC, vib_table, vib_spindle, AE_table, AE_spindle,VB]
    all_samples.append(sample)
data=np.array(all_samples)
# 确认数据的输出和格式没问题
# print(data.shape)
# print("前5行数据：\n", data[:5])


# ---------------------- Step2：数据清洗 ----------------------
print("进行数据清洗前的数据形状：", data.shape)
# 一、处理缺失值（np.nan，统一用均值填充）
print("\n开始处理缺失值...")
missing_count = np.isnan(data).sum(axis=0)  # np.isnan判断是否为空，sum(axis=0)按列统计
print("各列缺失值数量（对应8个特征+VB标签）：", missing_count)
# 填充VB缺失值
data[:,8]=np.where(np.isnan(data[:,8]),np.nanmean(data[:,8]),data[:,8])
missing_count_new = np.isnan(data).sum(axis=0)
print("缺失值填充后各列缺失值数量：", missing_count_new)

# 二、处理异常值
print("\n开始处理异常值...")
# 1.先记录原始样本数，方便后续对比剔除了多少异常值
original_sample_num = len(data)
print("异常值处理前的样本数：",original_sample_num)
# 2.计算每列特征列的均值（mean）和标准差（std）
for i in range(8):
    dataing=data[:, i]
    mean=np.mean(data[:,i])
    std=np.std(data[:,i])
    # 3σ法则：只保留 均值- 3*标准差 ≤ 数据 ≤ 均值+ 3*标准差 的样本
    # 生成布尔索引：符合条件的样本=True，异常样本=False
    mask=np.abs(dataing-mean) <= 3*std
    # 筛选有效样本：只保留True的行，剔除False的异常行（这里之后可优化，异常值换成均值）
    data=data[mask]
remove_sample_num = len(data)
print("异常值处理后的样本数：",remove_sample_num)
print("剔除的样本数：",original_sample_num-remove_sample_num)

# 三、剔除冗余数据（如果有无效样本，比如 VB=0 且传感器数据全为0的“空实验”）
print("\n开始处理冗余值...")
data = data[data[:,8] > 0.001]  # 保留磨损量 VB>0.001的有效样本（避免全零无效数据）
num=len(data)
print("冗余值处理后的样本数：",num)
print("剔除的样本数：",remove_sample_num-num)
print("\n进行数据清洗后的数据形状：", data.shape)


# ---------------------- Step3：数据标准化 ----------------------
print("\n开始进行数据标准化...")
feature=data[:, :8] #前8个为特征
target=data[:, 8]   #目标VB
#标准化
mean=np.mean(feature,axis=0)
std=np.std(feature,axis=0)
feature_norm=(feature-mean)/(std+1e-8)
print("进行数据标准化后的数据形状：",feature_norm.shape)
print("标准化后前5行数据：\n", feature_norm[:5])
print("标准化后各特征均值（≈0）：\n", np.round(np.mean(feature_norm, axis=0), 3))
print("标准化后各特征标准差（≈1）：\n", np.round(np.std(feature_norm, axis=0), 3))


# ---------------------- Step4：数据可视化----------------------
#设置中文显示
rcParams['font.family']="SimHei"
# 特征名称（用于图表标注）
feature_names = ['切削深度(DOC)', '进给量(feed)', '交流主轴电流(smcAC)', '直流主轴电流(smcDC)',
                 '工作台振动(vib_table)', '主轴振动(vib_spindle)', '工作台声发射(AE_table)', '主轴声发射(AE_spindle)']
#一、VB磨损量的分布直方图
plt.figure(figsize=(10, 6))
plt.hist(target, bins=15, color='blue', alpha=0.4, density=True)
plt.title('铣削刀片磨损量(VB)分布', fontsize=14, fontweight='bold')
plt.xlabel('磨损量(VB/mm)', fontsize=12)
plt.ylabel('样本数量（频数）', fontsize=12)
plt.tight_layout()
plt.savefig('VB_distribution_count.png', dpi=300)
plt.show()


# ---------------------- Step5：保存处理后的数据----------------------
np.savez(
    'milling_preprocessed_data.npz',
    features=feature_norm,  #标准化后的特征
    labels=target,          #VB
    # 备注特征名，方便后续分析
    feature_names=['DOC','feed','smcAC','smcDC','vib_table','vib_spindle','AE_table','AE_spindle','VB']
)

print("\n数据预处理完成")
print("\n数据预处理与可视化完成！")
print("\n最终特征形状：",feature_norm.shape)
print("最终目标VB形状：",target.shape)
