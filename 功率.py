import pandas as pd
import matplotlib.pyplot as plt

# 步骤一：加载 Excel 文件到 DataFrame 中
file_path = r'C:\Users\20586\Desktop\新建 Microsoft Excel 工作表 (2).xlsx'  # 替换为实际路径
df = pd.read_excel(file_path, sheet_name='Sheet1')  # 指定工作表名称或索引号

# 假设我们要绘制第二列作为 X 轴，第三列作为 Y 轴
x_data = df.iloc[:, 1].values  # 获取第二列的数据
y_data = df.iloc[:, 2].values  # 获取第三列的数据

# 步骤二：创建折线图
plt.figure(figsize=(10, 6))  # 设置图形大小
plt.plot(x_data, y_data, marker='o', linestyle='-', color='b', label='Data Line')

# 添加标题和标签
plt.title('Line Chart from Excel Data')
plt.xlabel('X-Axis Label')
plt.ylabel('Y-Axis Label')

# 显示网格以及图例
plt.grid(True)
plt.legend()

# 展示最终结果
plt.show()