# coding=utf-8
import numpy as np
import streamlit as st
import re
import  pandas as pd
from collections import defaultdict
#streamlit run SPC_text.py
def dfs(graph, start, path=[]):
    path = path + [start]
    print(path)  # 打印当前路径，你可以选择保存到列表或其他地方
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
def clean_original_data(x):
 try:
  if str(x)=="nan":
        return ""
  else:
    tt = str(x).split("-")
    temp_index = 0
    string = ""
    for m in range(len(tt)):
        if (len(tt[m]) >= 8) and "/" not in tt[m]:
            temp_index = m + 1
    for j in range(temp_index):
        string = string + tt[j] + "-"
    string = string[:-1]
    return string
 except Exception as e:
     st.write(e,x)
def clean_df_half(df,product,ID):
 df_name=df[product].tolist()
 df_ID=df[ID].tolist()
 for i in range(len(df_name)):
    if str(df_ID[i]).startswith(str(df_name[i])):
        continue
    else:
        df_ID[i]=str(df_name[i])+"-"+str(df_ID[i])
 df[ID]=df_ID
 return df
@st.cache_data
def get_df1(uploaded_files):
 df_original = pd.DataFrame()
 df_mix = pd.DataFrame()
 df_liq = pd.DataFrame()
 df_710_ = pd.DataFrame()
 df_sf = pd.DataFrame()
 df_fusion = pd.DataFrame()
 df_lsf = pd.DataFrame()
 df_wsf = pd.DataFrame()
 df_gtl = pd.DataFrame()
 df_gtlsf = pd.DataFrame()
 df_tanhua = pd.DataFrame()
 df_hzy = pd.DataFrame()
 for file_path in uploaded_files:
    df_original1 = get_data(file_path, "原材料", header=0)
    df_mix1 = get_data(file_path, "混料", header=1)
    df_liq1 = get_data(file_path, "沥青", header=1)
    df_710_1 = get_data(file_path, "制粉710", header=1)
    df_sf1 = get_data(file_path, "筛分", header=1)
    df_fusion1 = get_data(file_path, "融合", header=1)
    df_lsf1 = get_data(file_path, "立包", header=1)
    df_wsf1 = get_data(file_path, "卧包", header=1)
    df_gtl1 = get_data(file_path, "滚筒炉", header=1)
    df_gtlsf1 = get_data(file_path, "GTL筛分", header=1)
    df_tanhua1 = get_data(file_path, "碳化", header=1)
    df_hzy1 = get_data(file_path, "回转窑", header=1)
    df_original = pd.concat((df_original, df_original1))
    df_mix = pd.concat((df_mix, df_mix1))
    df_liq = pd.concat((df_liq, df_liq1))
    df_710_ = pd.concat((df_710_, df_710_1))
    df_sf = pd.concat((df_sf, df_sf1))
    df_fusion = pd.concat((df_fusion, df_fusion1))
    df_lsf = pd.concat((df_lsf, df_lsf1))
    df_wsf = pd.concat((df_wsf, df_wsf1))
    df_gtl = pd.concat((df_gtl, df_gtl1))
    df_gtlsf = pd.concat((df_gtlsf, df_gtlsf1))
    df_tanhua = pd.concat((df_tanhua, df_tanhua1))
    df_hzy = pd.concat((df_hzy, df_hzy1))
 return df_original,df_mix,df_liq,df_710_,df_sf,df_fusion,df_lsf,df_wsf,df_gtl,df_gtlsf,df_tanhua,df_hzy
@st.cache_data
def get_milling_df(uploaded_files):
    df_rank = pd.DataFrame()
    df_bangxiao = pd.DataFrame()
    df_ori_smash = pd.DataFrame()
    df_R_D_mix = pd.DataFrame()
    df_Coulter_mix = pd.DataFrame()
    for milling_file in uploaded_files:
        df_rank1 = get_data(milling_file, "分级", header=1)
        df_bangxiao1 = get_data(milling_file, "棒销磨", header=1)
        df_ori_smash1 = get_data(milling_file, "生料粉碎", header=1)
        df_R_D_mix1 = get_data(milling_file, "研发料混料", header=1)
        df_Coulter_mix1 = get_data(milling_file, "犁刀混料", header=1)
        df_rank  = pd.concat((df_rank, df_rank1))
        df_bangxiao = pd.concat((df_bangxiao, df_bangxiao1))
        df_ori_smash = pd.concat((df_ori_smash, df_ori_smash1))
        df_R_D_mix = pd.concat((df_R_D_mix, df_R_D_mix1))
        df_Coulter_mix = pd.concat((df_Coulter_mix, df_Coulter_mix1))
    return df_rank,df_bangxiao,df_ori_smash,df_R_D_mix,df_Coulter_mix
@st.cache_data
def get_half_df(uploaded_files):
    df_half = pd.DataFrame()
    for file_path in uploaded_files:
        df_half1 = get_data(file_path, "Sheet1", header=1)
        #df_half1 = clean_df_half(df_half1, "品名", "批号")
        df_half = pd.concat((df_half, df_half1))
    return df_half
def get_mix_df(uploaded_files):
    df_mix_screen = pd.DataFrame()
    for file_path in uploaded_files:
        df_mix_screen1 = get_data(file_path, "23年", header=0)
        df_mix_screen = pd.concat((df_mix_screen, df_mix_screen1))
        #df_mix_screen = clean_df_half(df_mix_screen, "原料品名", "原料批号")
    return df_mix_screen
def get_df(df,df_original,input1,output1):
 df_original = df_original.rename(columns={input1: '投入批号'})
 df_original = df_original.rename(columns={output1: '输出批号'})
 df = pd.concat([df,df_original],ignore_index=True)
 return df

uploaded_file = st.file_uploader("请上传生料表", accept_multiple_files=True)
uploaded_files2 = st.file_uploader("请上传混料筛分表", accept_multiple_files=True)
uploaded_files3 = st.file_uploader("请上传半成品表", accept_multiple_files=True)
uploaded_files4 = st.file_uploader("请上传制粉表", accept_multiple_files=True)

start_node = st.text_input('请输入要追溯的批号', '批号')
st.write('当前输入的批号是', start_node)

#获取数据
df_original,df_mix,df_liq,df_710_,df_sf,df_fusion,df_lsf,df_wsf,df_gtl,df_gtlsf,df_tanhua,df_hzy=get_df1(uploaded_file)
df_mix_screen=get_mix_df(uploaded_files2)
df_half=get_half_df(uploaded_files3)
df_rank,df_bangxiao,df_ori_smash,df_R_D_mix,df_Coulter_mix=get_milling_df(uploaded_files4)
if len(uploaded_file)>0 and len(uploaded_files2)>0 and len(uploaded_files3)>0 and len(uploaded_files4)>0:
#处理数据
 df_mix['出料批号']=df_mix['出料批号'].apply(clean_original_data)
 df_liq['出料批号']=df_liq['出料批号'].apply(clean_original_data)
 df_710_['B料批号']=df_710_['B料批号'].apply(clean_original_data)
 df_sf['A料批号']=df_sf['A料批号'].apply(clean_original_data)
 df_fusion['生产批号']=df_fusion['生产批号'].apply(clean_original_data)
 df_lsf['A料批号']=df_lsf['A料批号'].apply(clean_original_data)
 df_wsf['A料批号']=df_wsf['A料批号'].apply(clean_original_data)
 df_gtl['A料批号']=df_gtl['A料批号'].apply(clean_original_data)
 df_gtlsf['A料批号']=df_gtlsf['A料批号'].apply(clean_original_data)
#空值处理
 df_mix_screen= clean_df_half(df_mix_screen , "原料品名", "原料批号")
 df_half = clean_df_half(df_half, "品名", "批号")
#确定输入和输出
 df=pd.DataFrame(columns=['品名','投入批号', '输出批号'])
 df_original =df_original [['品名','投料批号','生产批号']]
 df_original = df_original.rename(columns={'品号': '品名'})
 df_original=df_original.drop_duplicates()
 df=get_df(df,df_original,"投料批号",'生产批号')
 df_mix =df_mix [['品名','原料1批号',"原料2批号",'出料批号']]
 df_mix=df_mix.drop_duplicates()
 df_mix1=df_mix[['品名','原料1批号','出料批号']]
 df_mix2=df_mix[['品名','原料2批号','出料批号']]
 df=get_df(df,df_mix1,"原料1批号",'出料批号')
 df=get_df(df,df_mix2,"原料2批号",'出料批号')

 df_liq =df_liq [['品号','原料批号','出料批号']]
 df_liq = df_liq.rename(columns={'品号': '品名'})
 df_liq=df_liq.drop_duplicates()
 df=get_df(df,df_liq,"原料批号",'出料批号')
 df_710_=df_710_[['品名','A料批号','B料批号',"发货批号"]]
 df_710_=df_710_.drop_duplicates()
 df_710_1=df_710_[['品名','A料批号','B料批号']]
 df_710_2=df_710_[['品名','B料批号','发货批号']]
 df=get_df(df,df_710_1,"A料批号",'B料批号')
 df=get_df(df,df_710_2,"B料批号",'发货批号')
 df_sf=df_sf[['品名','A料批号','B料批号']]
 df_sf=df_sf.drop_duplicates()
 df=get_df(df,df_sf,"A料批号",'B料批号')
#融合车间
 df_fusion=df_fusion[['品名','投料批号','生产批号',"发货批号"]]
 df_fusion=df_fusion.drop_duplicates()
 df_fusion1=df_fusion[['品名','投料批号','生产批号']]
 df_fusion2=df_fusion[['品名','生产批号','发货批号']]
 df=get_df(df,df_fusion1,"投料批号",'生产批号')
 df=get_df(df,df_fusion2,"生产批号",'发货批号')

 df_lsf=df_lsf[['品名','原料1批号','原料2批号','A料批号',"B料批号"]]
 df_lsf=df_lsf.drop_duplicates()
 df_lsf1=df_lsf[['品名','原料1批号','B料批号']]
 df_lsf2=df_lsf[['品名','原料2批号','B料批号']]
 df_lsf3=df_lsf[['品名','A料批号','B料批号']]
 df=get_df(df,df_lsf1,"原料1批号",'B料批号')
 df=get_df(df,df_lsf2,"原料2批号",'B料批号')
 df=get_df(df,df_lsf3,"A料批号",'B料批号')

 df_wsf=df_wsf[['品名','原料1批号','原料2批号','A料批号',"B料批号"]]
 df_wsf=df_wsf.drop_duplicates()
 df_wsf1=df_wsf[['品名','原料1批号','B料批号']]
 df_wsf2=df_wsf[['品名','原料2批号','B料批号']]
 df_wsf3=df_wsf[['品名','A料批号','B料批号']]
 df=get_df(df,df_wsf1,"原料1批号",'B料批号')
 df=get_df(df,df_wsf2,"原料2批号",'B料批号')
 df=get_df(df,df_wsf3,"A料批号",'B料批号')

 df_gtl=df_gtl[['品名','原料1批号','原料2批号','A料批号']]
 df_gtl=df_gtl.drop_duplicates()
 df_gtl1=df_gtl[['品名','原料1批号','A料批号']]
 df_gtl2=df_gtl[['品名','原料2批号','A料批号']]
 df=get_df(df,df_gtl1,"原料1批号",'A料批号')
 df=get_df(df,df_gtl2,"原料2批号",'A料批号')

 df_gtlsf=df_gtlsf[['品名','A料批号',"B料批号"]]
 df_gtlsf=df_gtlsf.drop_duplicates()
 df=get_df(df,df_gtlsf,"A料批号",'B料批号')

 df_tanhua=df_tanhua[['品名','原料批号','BC料批号']]
 df_tanhua=df_tanhua.drop_duplicates()
 df=get_df(df,df_tanhua,"原料批号",'BC料批号')

 df_hzy=df_hzy[['品名','原料批号','出料批号']]
 df_hzy=df_hzy.drop_duplicates()
 df=get_df(df,df_hzy,"原料批号",'出料批号')
#混料筛分
 df_mix_screen=df_mix_screen[['品名','原料批号','产品批号']]
 df_mix_screen=df_mix_screen.drop_duplicates()
 df=get_df(df,df_mix_screen,"原料批号",'产品批号')
#半成品到货表
 df_half=df_half[['品名','批号','B料批号(回货批号）']]
 df_half=df_half.drop_duplicates()
 df=get_df(df,df_half,"批号",'B料批号(回货批号）')

#制粉
 #分级、生料粉碎、棒消、研发料混料、力道混料
 df_rank,df_bangxiao,df_ori_smash,df_R_D_mix,df_Coulter_mix=get_milling_df(uploaded_files4)

 df_rank=df_rank[['品名','投料批号','F2出料批号']]
 df_rank=df_rank.drop_duplicates()
 df=get_df(df,df_rank,'投料批号','F2出料批号')

 df_ori_smash=df_ori_smash[['品名','投料批号','B料批号',"发货批号"]]
 df_ori_smash=df_ori_smash.drop_duplicates()
 df_ori_smash1=df_ori_smash[['品名','投料批号','B料批号']]
 df_ori_smash2=df_ori_smash[['品名','B料批号','发货批号']]
 df=get_df(df,df_ori_smash1,"投料批号",'B料批号')
 df=get_df(df,df_ori_smash2,"B料批号",'发货批号')
#总的数据处理
 df =df.dropna(how="all")
 df = df.dropna(subset=['投入批号'])
 df =df.drop_duplicates()
 graph = defaultdict(list)
 input_list = df['投入批号'].tolist()
 output_list = df['输出批号'].tolist()
 for i in range(len(input_list)):
    graph[output_list[i]].append(input_list[i])
 paths = dfs(graph, start_node)
 st.write(start_node)
 st.write('原始数据')
 st.write(paths)
 st.write('修改后的数据')
 for i in range(len(paths)):
  tt=paths[i]
  tt=str(tt)
  list_temp = []
  tt = tt.replace("[", "").replace("]", "").replace("'", "")
  tt=tt.split(",")
  for j in range(len(tt)):
     list_temp.append(tt[j].replace(" ",""))
  list_temp = list(dict.fromkeys(list_temp))
  st.write(list_temp)