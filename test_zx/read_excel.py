import pandas as pd
import ast

def read_excel():
    # 读取Excel文件的某一列
    filename = '../数据集情况汇总 - 副本.xlsx'
    sheetname = 'Sheet1'
    columnname = '类别名称'
    # df = pd.read_excel(filename, sheet_name=sheetname, usecols=[columnname],header=1)
    df = pd.read_excel(filename, sheet_name=sheetname, usecols=[columnname])
    # 删除缺失值
    df = df.dropna()
    # print(df)

    # 将列转换为列表
    list_data = df[columnname].tolist()
    list_data = [ast.literal_eval(elem) for elem in list_data]

    # #读取每一行
    # for line in list_data:
    #     print(line)
    return list_data