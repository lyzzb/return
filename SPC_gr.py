# coding=utf-8
import numpy as np
import streamlit as st
import re
import  pandas as pd
from collections import defaultdict
from graphviz import Digraph
#streamlit run SPC_text.py
def dfs(graph, start, path=[]):
    path = path + [start]
    #print(path)  # 打印当前路径，你可以选择保存到列表或其他地方
    if not graph[start]:  # 检查是否还有下一个节点
        return path  # 返回当前路径（当没有下一个节点时）
    paths = []
    for node in graph[start]:
        if node not in path:
            newpath = dfs(graph, node, path)  # 递归调用，并更新路径
            if newpath:  # 如果返回了路径（即下面还有节点），则保存这条路径
                paths.append(newpath)
    return paths  # 返回所有路径（当有下一个节点时）
def get_pin_name(x):
    tt = x.split('-')
    index = 0
    result = ""
    for i in range(len(tt)):
        if len(tt[i]) >= 8:
            index = i
    for i in range(0, index + 1):
        result = result + tt[i] + "-"
    result = result[:-1]
    return result
@st.cache_data
def get_data(file_path,sheet_name,header):
    dff1 = pd.read_excel(file_path, engine="openpyxl", sheet_name=sheet_name,header=header)
    return dff1
def get_lims_data(uploaded_files):
    lims_list = []
    for uploaded_file2 in uploaded_files:
        df = get_data(uploaded_file2, sheet_name=0, header=3)
        del_list = []
        for j in range(5, len(df.columns)):
            if str(df.iloc[5, j]) == "nan":
                del_list.append(j)
        df = df.drop(df.columns[del_list], axis=1)
        df = df.drop(df.index[6])
        df.rename(columns={"Unnamed: 0": "NO", 'Unnamed: 1': '批号', 'Unnamed: 2': '批次号', 'Unnamed: 3': '送检日期'},
                  inplace=True)
        test_range_index = df.iloc[5, 5:].index.tolist()
        test_range = df.iloc[5, 5:].tolist()
        df = df.iloc[6:]
        df['批号'] = df['批号'].fillna(method='ffill')
        df['批次号'] = df['批次号'].fillna(method='ffill')
        df['NO'] = df['NO'].fillna(method='ffill')
        df = df.dropna()
        df['送检日期'] = df['送检日期'].fillna(method='ffill')
        df = df.drop(df[df['批号'].str.contains('FQ')].index)
        df = df.drop(df[df['批号'].str.contains('FC')].index)
        df = df.drop(df[df['批号'].str.contains('BAO')].index)
        df = df.drop(df[df['批号'].str.contains('bao')].index)
        df = df.drop(df[df['批号'].str.contains('无内超')].index)
        df = df.drop(df[df['批号'].str.contains('g')].index)
        df['contains_suffix'] = df['批号'].apply(lambda x: "G" in x[-6:])
        df = df.drop(df[df['contains_suffix'] == True].index)
        df = df.drop(df[df['批号'].str.contains("异常")].index)
        df = df.drop(df[df['批号'].str.contains("纤维")].index)
        df = df.drop(df[df['批号'].str.contains('油系')].index)
        df = df.drop(df[df['批号'].str.contains('JS')].index)
        df = df.drop(df[df['批号'].str.contains('石墨化')].index)
        df = df.drop(df[df['批号'].str.contains('YF')].index)
        df = df.drop(df[df['批号'].str.contains('SSW')].index)
        df = df.drop(df[df['批号'].str.contains('粉压')].index)
        df = df.drop(df[df['批号'].str.len() < 8].index)
        df = df.drop(df[df['批次号'].str.contains('FQ')].index)
        df = df.drop(df[df['批次号'].str.contains('FC')].index)
        df = df.drop(df[df['批次号'].str.contains('BAO')].index)
        df = df.drop(df[df['批次号'].str.contains('bao')].index)
        df['品名'] = df['批号'].apply(get_pin_name)
        df = df.reindex(columns=[df.columns[-1], *df.columns[:-1]])
        df = df.drop("contains_suffix", axis=1)
        df = df.drop("分析项目", axis=1)
        df = df.drop("NO", axis=1)
        df = df.drop("批次号", axis=1)
        dg = df.groupby('品名')
        for name, dg1 in dg:
            lims_list.append(dg1)
    return lims_list
start_node = st.text_input('请输入要追溯的批号', '批号')
st.write('当前输入的批号是', start_node)
uploaded_file = st.file_uploader("请上传总表")
#检测数据

uploaded_files = st.file_uploader("请上传检测数据", accept_multiple_files=True)
uploaded_files=["D:\limsdownload\M12B.xlsx","D:\limsdownload\\COA00401_原始数据台账_deerta_Ver1.0.1 (1).xlsx"]
check_lims_data = st.checkbox('检测数据追溯')
st.write("没查到可能是1：数据质量不行.2.真的没有对应批次.3.标点符号(括号、空格)")
output_data=[]
df2 = pd.DataFrame()
if uploaded_file!=None :
#处理数据
 df=get_data(uploaded_file,"Sheet1",header=0)
 graph = defaultdict(list)
 input_list = df['投入批号'].tolist()
 output_list = df['输出批号'].tolist()
 for  i in range(len(input_list)):
    input_list[i]=str(input_list[i]).replace(" ","")
 for i in range(len(output_list)):
    output_list[i] = str(output_list[i]).replace(" ", "")
 for i in range(len(input_list)):
    graph[output_list[i]].append(input_list[i])
 paths = dfs(graph, start_node.replace(" ",""))
 st.write(start_node)
 paths=str(paths)
 data=paths.split("],")
 df_out = pd.DataFrame(columns=['input', 'output'])
 for m in range(len(data)):
     tt=data[m]
     tt = tt.replace("[", "").replace("]", "").replace("'", "").replace(" ","")
     tt = tt.split(",")
     for j in range(1,len(tt)):
         append_list=[tt[j-1],tt[j]]
         df_out.loc[len( df_out)] = append_list
     output_data.append(tt)
 for lst in output_data:
    df2 = df2._append(pd.DataFrame(lst).transpose())
 st.write(df2)
 df_out=df_out.drop_duplicates()
#重量、检测项、
 source=df_out["input"].tolist()
 target=df_out["output"].tolist()
 all_list=source+target
 all_list=list1 = list(dict.fromkeys(all_list))
 col1,col2=st.columns(2)
 with col1:
     dot = Digraph()
     for k in range(len(source)):
         dot.node(source[k])
         dot.node(target[k])
         dot.edge(source[k], target[k])
     st.graphviz_chart(dot)
 with col2:
  if len(uploaded_files)>0 and check_lims_data==True:
      lims_df = get_lims_data(uploaded_files)
      for i in range(len(all_list)):
          temp_df1=pd.DataFrame()
          for lims in lims_df:
              temp_df=lims[lims["品名"]==all_list[i]]
              if temp_df.empty!=True:
                  temp_df1=temp_df
          st.write(temp_df1)









