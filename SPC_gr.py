# coding=utf-8
import networkx as nx
import numpy as np
import streamlit as st
import re
import  pandas as pd
from collections import defaultdict
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

@st.cache_data
def get_data(file_path,sheet_name,header):
    dff1 = pd.read_excel(file_path, engine="openpyxl", sheet_name=sheet_name,header=header)
    return dff1

start_node = st.text_input('请输入要追溯的批号', '批号')
st.write('当前输入的批号是', start_node)

uploaded_file = st.file_uploader("请上传总表")
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
 for m in range(len(data)):
     tt=data[m]
     tt = tt.replace("[", "").replace("]", "").replace("'", "").replace(" ","")
     tt = tt.split(",")
     output_data.append(tt)
 for lst in output_data:
    df2 = df2._append(pd.DataFrame(lst).transpose())
 st.write(df2)







