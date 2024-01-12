# coding=utf-8
import numpy as np
import streamlit as st
import re
import  pandas as pd
#streamlit run SPC.py
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
        if len(tt[m]) >= 8:
            temp_index = m + 1
    for j in range(temp_index):
        string = string + tt[j] + "-"
    string = string[:-1]
    return string
 except Exception as e:
     st.write(e,x)
def get_SK1_M13(send_ID,df_fusion,df_gtlsf,df_gtl,df_mix,df_liq):
#A料，B料
 data = {
    '发货批号': [send_ID],
 }
 df = pd.DataFrame(data)
#融合车间
 df = pd.merge(df_fusion, df, on='发货批号')
 df=df.loc[:,['生产批号', '投料批号',"发货批号"]]
 df= df.drop_duplicates()
#GTL筛分和GTL车间，会出现是SF和非SF的情况
 df=pd.merge (df_gtlsf, df, left_on= 'B料批号', right_on= '投料批号')
 df=df.loc[:,['A料批号', 'B料批号','生产批号',"发货批号"]]
 df= df.drop_duplicates()
 df = pd.merge(df_gtl, df, on='A料批号')
 df=df.loc[:,['原料1批号','A料批号', 'B料批号','生产批号',"发货批号"]]
 df= df.drop_duplicates()
#混料车间
 df=pd.merge (df_mix, df, left_on= '出料批号', right_on= '原料1批号')
 df=df.loc[:,['原料1批号_x','原料2批号','出料批号','A料批号', 'B料批号','生产批号',"发货批号"]]
 df= df.drop_duplicates()
#XM-5
 df=pd.merge (df_liq, df, left_on= '出料批号', right_on= '原料2批号')
 df=df.loc[:,['原料批号','出料批号_x','原料1批号_x','原料2批号','出料批号_y','A料批号', 'B料批号','生产批号',"发货批号"]]
 df= df.drop_duplicates()
 return df
def get_SK1_M23(send_ID,df_710_,df_gtl,df_mix,df_fusion):
#A料，B料
 data = {
    '发货批号': [send_ID],
 }
 df = pd.DataFrame(data)
#710车间
 df = pd.merge(df_710_, df, on='发货批号')
 df=df.loc[:,['A料批号', 'B料批号',"发货批号"]]
 df= df.drop_duplicates()
 #GTL车间
 df = pd.merge(df_gtl, df, on='A料批号')
 df=df.loc[:,["原料1批号","原料2批号",'A料批号', 'B料批号',"发货批号"]]
 df= df.drop_duplicates()
#混料车间
 df=pd.merge (df_mix, df, left_on= '出料批号', right_on= '原料1批号')
 df=df.loc[:,["原料1批号_x","原料2批号_x",'出料批号','A料批号', 'B料批号',"发货批号"]]
 df= df.drop_duplicates()
 df=df.ffill()
 df=pd.merge (df_liq, df, left_on= '出料批号', right_on= '原料2批号_x')
 df=df.loc[:,['原料批号','出料批号_x','原料1批号_x','出料批号_y','A料批号', 'B料批号',"发货批号"]]
 df= df.drop_duplicates()
#R23
 df_fusion=df_fusion[df_fusion["品名"]=="R23"]
 dg=df_fusion.groupby('生产批号')
 X13_list=[]
 D11_list=[]
 send_list=[]
 for name, dg1 in dg:
    X13=dg1[dg1["投料批号"].str.contains("X13")]
    X13=X13.drop_duplicates()
    X13=X13['投料批号'].tolist()[0]
    D11 = dg1[dg1["投料批号"].str.contains("D11")]
    D11 = D11.drop_duplicates()
    D11  = D11['投料批号'].tolist()[0]
    send =dg1["发货批号"].tolist()[0]
    X13_list.append(X13)
    D11_list.append(D11)
    send_list.append(send)
 data = {
    'X13': X13_list,
    'D11': D11_list,
    '原料1批号_x': send_list
 }
 df_fusion = pd.DataFrame(data)
 df = pd.merge(df_fusion, df, on='原料1批号_x')
 df= df.drop_duplicates()
 return df
def get_LK_T(send_ID,df_fusion,df_gtlsf,df_gtl,df_mix,df_liq):
 data = {
    '发货批号': [send_ID],
 }
 df = pd.DataFrame(data)
#融合车间
 df = pd.merge(df_fusion, df, on='发货批号')
 df=df.loc[:,['投料批号', '生产批号',"发货批号"]]
 df= df.drop_duplicates()
#GTL筛分车间和GTL
 df=pd.merge (df_gtlsf, df, left_on= 'B料批号', right_on= '投料批号')
 df=df.loc[:,['A料批号', 'B料批号','生产批号',"发货批号"]]
 df= df.drop_duplicates()
 df = pd.merge(df_gtl, df, on='A料批号')
 df=df.loc[:,['原料1批号','A料批号', 'B料批号','生产批号',"发货批号"]]
 df= df.drop_duplicates()
#混料车间
 df=pd.merge (df_mix, df, left_on= '出料批号', right_on= '原料1批号')
 df=df.loc[:,["原料1批号_x","原料2批号",'出料批号','A料批号', 'B料批号','生产批号',"发货批号"]]
 df= df.drop_duplicates()
 df=df.ffill()
 df=pd.merge (df_liq, df, left_on= '出料批号', right_on= '原料2批号')
 df=df.loc[:,['原料批号','出料批号_x','原料1批号_x','出料批号_y','A料批号', 'B料批号','生产批号',"发货批号"]]
 df= df.drop_duplicates()
 return df
def get_LK_P(send_ID,df_fusion,df_gtlsf,df_gtl,df_mix):
 data = {
    '发货批号': [send_ID],
 }
 df = pd.DataFrame(data)
#融合车间，可能会存在一个是SF，一个是A的情况
 df = pd.merge(df_fusion, df, on='发货批号')
 df=df.loc[:,['投料批号', '生产批号',"发货批号"]]
 df_all= df.drop_duplicates()
#GTL筛分车间
 df=pd.merge (df_gtlsf, df_all, left_on= 'B料批号', right_on= '投料批号')
 df=df.loc[:,['A料批号', 'B料批号','生产批号',"发货批号"]]
 df= df.drop_duplicates()
#GTL车间
 df = pd.merge(df_gtl, df, on='A料批号')
 df=df.loc[:,['原料1批号','A料批号', 'B料批号','生产批号',"发货批号"]]
 df= df.drop_duplicates()
#不包含SF
 mask = df_all['投料批号'].str.contains('SF') == False
 df_no_SF= df_all[mask]
 if df_no_SF.empty==False:
  df_no_SF=pd.merge (df_gtl, df_no_SF, left_on= 'A料批号', right_on= '投料批号')
  df_no_SF=df_no_SF.loc[:,['原料1批号','A料批号', '生产批号',"发货批号"]]
  df_no_SF= df_no_SF.drop_duplicates()
  df = pd.concat([df, df_no_SF])
#混料
 df=pd.merge (df_mix, df, left_on= '出料批号', right_on= '原料1批号')
 df=df.loc[:,["原料1批号_x","原料2批号",'出料批号','A料批号', 'B料批号','生产批号',"发货批号"]]
 df= df.drop_duplicates()
 df = df.ffill()
#沥青
 df=pd.merge (df_liq, df, left_on= '出料批号', right_on= '原料2批号')
 df=df.loc[:,['原料批号','出料批号_x','原料1批号_x','出料批号_y','A料批号', 'B料批号','生产批号',"发货批号"]]
 df= df.drop_duplicates()
 return df
def get_AP5(send_ID,df_lsf,df_wsf):
 data = {
    'B料批号': [send_ID],
 }
 df = pd.DataFrame(data)
#立包
 df_l = pd.merge(df_lsf, df, on='B料批号')
 df_l=df_l.loc[:,['原料1批号','原料2批号','A料批号' ,"B料批号"]]
 df_l= df_l.drop_duplicates()
 df_l = df_l.ffill()
#卧包
 df_w = pd.merge(df_wsf, df, on='B料批号')
 df_w=df_w.loc[:,[ '原料1批号','原料2批号','A料批号',"B料批号"]]
 df_w= df_w.drop_duplicates()
 df_w=df_w.ffill()
 df= pd.concat([df_l , df_w])
 df = df.drop_duplicates()
 #如果有A3-R5H则需要在混料表里面进行体现
 A3_R5H1=df['原料1批号'].tolist()
 A3_R5H_boolean=False
 A3_R5H2_boolean=True
 A3_R5H2=df['原料2批号'].tolist()
 #如果是A3-R5H
 for A3_R5H in A3_R5H1:
     if "A3-R5H" in str(A3_R5H):
         A3_R5H_boolean=True
 if A3_R5H_boolean==True:
  df1 = pd.merge (df_mix, df, left_on= '出料批号', right_on= '原料1批号')
  df1 = df1.loc[:, ['原料1批号_x', '原料2批号_x','出料批号','原料2批号_y', 'A料批号', "B料批号"]]
  df1 = df1.drop_duplicates()
  df1= pd.merge(df_liq, df1, left_on='出料批号', right_on='原料2批号_x')
  df1 = df1.loc[:, ['原料批号', '出料批号_x', '原料1批号_x', '出料批号_y', '原料2批号_y', 'A料批号', 'B料批号']]
  df1 = df1.drop_duplicates()
 for A3_R5H in A3_R5H2:
     if "A3-R5H" in str(A3_R5H):
         A3_R5H2_boolean=True
 if A3_R5H2_boolean==True:
       df2 = pd.merge(df_mix, df, left_on='出料批号', right_on='原料2批号')
       df2 = df2.loc[:, ['原料1批号_x', '原料2批号_x', '出料批号', '原料1批号_y', 'A料批号', "B料批号"]]
       df2 = df2.drop_duplicates()
       df2 = pd.merge(df_liq, df2, left_on='出料批号', right_on='原料2批号_x')
       df2 = df2.loc[:, ['原料批号', '出料批号_x', '原料1批号_x', '出料批号_y', '原料1批号_y','A料批号', 'B料批号']]
       df2 = df2.drop_duplicates()

 return df,df1,df2
def get_PV_6(send_ID,df_tanhua,df_fusion,df_mix,df_liq):
 data = {
        'BC料批号': [send_ID],
 }
 df = pd.DataFrame(data)
#碳化车间
 df = pd.merge(df_tanhua, df, on='BC料批号')
 df = df.loc[:, [ '原料批号', "BC料批号"]]
 df = df.drop_duplicates()
#融合车间,分两种情况一种是PVH-6，一种是GHMG-C3和D11
 PVH_list=[]
 GHMG_C3_list=[]
 D11_list=[]
 send_list=[]
 send_list_PVH=[]
 df_fusion=df_fusion[df_fusion["品名"]=="PVA-6"]
 dg = df_fusion.groupby('生产批号')
 for name, dg1 in dg:
    send = dg1["发货批号"].tolist()[0]
    PVH = dg1[dg1["投料批号"].str.contains("PVH-6")]
    PVH = PVH.drop_duplicates()
    PVH = PVH['投料批号'].tolist()
    if len(PVH) != 0:
        PVH_list.append(PVH[0])
        send_list_PVH.append(send)
    GHMG = dg1[dg1["投料批号"].str.contains("GHMG-C3")]
    GHMG = GHMG.drop_duplicates()
    GHMG = GHMG['投料批号'].tolist()
    if len(GHMG)!=0:
        GHMG_C3_list.append(GHMG[0])
        send_list.append(send)
    D11 = dg1[dg1["投料批号"].str.contains("D11")]
    D11 = D11.drop_duplicates()
    D11 = D11['投料批号'].tolist()
    if len(D11) != 0:
        D11_list.append(D11[0])
#两个的
 data1 = {
    'GHMG-C3': GHMG_C3_list,
    'D11': D11_list,
    'PVH': [" "]*len(D11_list),
    '原料批号': send_list
  }
 df_fusion1 = pd.DataFrame(data1)
 df2= pd.merge(df_fusion1, df, on='原料批号')
 df2 = df2.loc[:, ['GHMG-C3','D11','原料批号', "BC料批号"]]
 df2= df2.drop_duplicates()
#一个的
 data = {
    'GHMG-C3': [" "]*len(PVH_list),
    'D11': [" "]*len(PVH_list),
    'PVH': PVH_list,
    '原料批号': send_list_PVH
  }
 df_fusion = pd.DataFrame(data)
 df3 = pd.merge(df_fusion, df, on='原料批号')
 df3 = df3.loc[:, ['PVH','GHMG-C3','D11','原料批号', "BC料批号"]]
 df3= df3.drop_duplicates()
#混料车间
 df3=pd.merge (df_mix, df3, left_on= '出料批号', right_on= 'PVH')
 df3=df3.loc[:,["原料1批号","原料2批号",'出料批号','PVH', '原料批号',"BC料批号"]]
 df3= df3.drop_duplicates()
 df3=df3.ffill()
#XH-3
 df3=pd.merge (df_liq, df3, left_on= '出料批号', right_on= '原料2批号')
 df3=df3.loc[:,['原料批号_x','出料批号_x','原料1批号','PVH', '原料批号_y',"BC料批号"]]
 df3= df3.drop_duplicates()
 return df2,df3
def get_EP7_H(send_ID,df_tanhua,df_fusion):
 data = {
        'BC料批号': [send_ID],
  }
 df = pd.DataFrame(data)
#碳化车间
 df = pd.merge(df_tanhua, df, on='BC料批号')
 df = df.loc[:, [ '原料批号', "BC料批号"]]
 df = df.drop_duplicates()
 df_fusion=df_fusion[df_fusion["品名"]=="EP7T-HB"]
 dg=df_fusion.groupby('发货批号')
 EP7T_list=[]
 D11_list=[]
 send_list=[]
 for name, dg1 in dg:
    df_t=dg1['投料批号'].tolist()
    for i in range(len(df_t)):
        if (i+1)%2==0:
            D11_list.append(df_t[i])
            send = dg1["发货批号"].tolist()[0]
            send_list.append(send)
        else:
            EP7T_list.append(df_t[i])
 data = {
    'EP7T': EP7T_list,
    'D11': D11_list,
    '原料批号': send_list
  }
 df_fusion = pd.DataFrame(data)
 df = pd.merge(df_fusion, df, on='原料批号')
 df= df.drop_duplicates()
 return df
def get_DFG(send_ID,df_710_,df_wsf,df_mix,df_liq):
 data = {
        '发货批号': [send_ID],
  }
 df = pd.DataFrame(data)
#碳化车间
 df = pd.merge(df_710_, df, on='发货批号')
 df = df.loc[:, [ 'A料批号','B料批号', "发货批号"]]
 df = df.drop_duplicates()
#卧式
 df = pd.merge(df_wsf, df, on='A料批号')
 df = df.loc[:, ['原料1批号', 'A料批号','B料批号_y', "发货批号"]]
 df = df.drop_duplicates()
#混料
 df=pd.merge (df_mix, df, left_on= '出料批号', right_on= '原料1批号')
 df = df.loc[:, ['原料1批号_x','原料2批号','出料批号','原料1批号_y', 'A料批号','B料批号_y', "发货批号"]]
 df = df.drop_duplicates()
 df=pd.merge (df_liq, df, left_on= '出料批号', right_on= '原料2批号')
 df=df.loc[:,['原料批号','出料批号_x','原料1批号_x','原料1批号_y','A料批号', 'B料批号_y',"发货批号"]]
 df= df.drop_duplicates()
 return df
def get_RG_S(send_ID,df_wsf,df_mix,df_liq):
 data = {
        'B料批号': [send_ID],
  }
 df = pd.DataFrame(data)
#碳化车间
 df = pd.merge(df_wsf, df, on='B料批号')
 df = df.loc[:, ['原料1批号', 'A料批号','B料批号']]
 df = df.drop_duplicates()
 df=pd.merge (df_mix, df, left_on= '出料批号', right_on= '原料1批号')
 df=df.loc[:,['原料1批号_x','原料2批号','出料批号','原料1批号_y','A料批号', 'B料批号']]
 df= df.drop_duplicates()
 df=pd.merge (df_liq, df, left_on= '出料批号', right_on= '原料2批号')
 df=df.loc[:,['原料批号','出料批号_x','原料1批号_x','出料批号_y','A料批号', 'B料批号']]
 df= df.drop_duplicates()
 return df
def get_CG09(send_ID,df_710_,df_wsf,df_mix):
 data = {
        '发货批号': [send_ID],
  }
 df = pd.DataFrame(data)
#碳化车间
 df1 = pd.merge(df_710_, df, on='发货批号')
 df1 = df1.loc[:, ['A料批号', 'B料批号','发货批号']]
 df1 = df1.drop_duplicates()
#卧包
 df2=pd.merge (df_wsf, df, left_on= 'B料批号', right_on= '发货批号')
 df2 = df2.loc[:, ['原料1批号','A料批号', 'B料批号']]
 df2 = df2.drop_duplicates()
 df2 = pd.merge(df_mix, df2, left_on='出料批号', right_on='原料1批号')
 df2 = df2.loc[:, ['原料1批号_x','原料2批号', '出料批号','A料批号', 'B料批号']]
 df2 = df2.drop_duplicates()
 df2 = pd.merge(df_liq, df2, left_on='出料批号', right_on='原料2批号')
 df2 = df2.loc[:, ['原料批号', '出料批号_x', '原料1批号_x', '出料批号_y', 'A料批号', 'B料批号']]
 df2 = df2.drop_duplicates()
 return df1,df2
#QCG-X4,G6B但写的是G6A
def get_QCG_X4(send_ID,df_tanhua,df_gtl,df_mix,df_liq):
 data = {
        'BC料批号': [send_ID],
  }
 df = pd.DataFrame(data)
#碳化车间
 df = pd.merge(df_tanhua, df, on='BC料批号')
 df = df.loc[:, ['原料批号','BC料批号']]
 df = df.drop_duplicates()
 #GTL筛分
 df = pd.merge(df_gtlsf, df, left_on='B料批号', right_on='原料批号')
 df = df.loc[:, [ 'A料批号','原料批号','BC料批号']]
 df = df.drop_duplicates()
#GTL车间
 df = pd.merge(df_gtl, df, on='A料批号')
 df = df.loc[:, ['原料1批号', 'A料批号','原料批号','BC料批号']]
 df = df.drop_duplicates()

 df = pd.merge(df_mix, df, left_on='出料批号', right_on='原料1批号')
 df = df.loc[:, ['原料1批号_x', '原料2批号', '出料批号',  'A料批号', '原料批号','BC料批号']]
 df = df.drop_duplicates()
 df = df.ffill()

 df=pd.merge (df_liq, df, left_on= '出料批号', right_on= '原料2批号')
 df=df.loc[:,['原料批号_x','出料批号_x','原料1批号_x','出料批号_y','A料批号','原料批号_y', 'BC料批号']]
 df= df.drop_duplicates()

 return df
def get_CSA_3E(send_ID,df_tanhua,df_fusion,df_mix,df_liq):
 data = {
        'BC料批号': [send_ID],
  }
 df = pd.DataFrame(data)
#碳化车间
 df = pd.merge(df_tanhua, df, on='BC料批号')
 df = df.loc[:, ['原料批号','BC料批号']]
 df = df.drop_duplicates()
#融合车间
 df = pd.merge(df_fusion, df, left_on='发货批号', right_on='原料批号')
 df = df.loc[:, ['投料批号','原料批号','BC料批号']]
 df = df.drop_duplicates()
#混料车间
 df = pd.merge(df_mix, df, left_on='出料批号', right_on='投料批号')
 df = df.loc[:, ['原料1批号','原料2批号','投料批号','原料批号','BC料批号']]
 df = df.drop_duplicates()
 df=pd.merge (df_liq, df, left_on= '出料批号', right_on= '原料2批号')
 df=df.loc[:,['原料批号_x','出料批号','原料1批号','投料批号','原料批号_y', 'BC料批号']]
 df= df.drop_duplicates()
 return df
def get_LA1(send_ID,df_sf,df_lsf,df_mix,df_liq):
#LA1
 data = {
        'B料批号': [send_ID],
  }
 df = pd.DataFrame(data)
#碳化车间
 df = pd.merge(df_sf, df, on='B料批号')
 df = df.loc[:, ['A料批号','B料批号']]
 df = df.drop_duplicates()
 df = pd.merge(df_lsf, df, on='A料批号')
 df = df.loc[:, ['原料1批号','原料2批号','A料批号','B料批号_x']]
 df = df.drop_duplicates()
 df=df.ffill()
 df2 = pd.merge(df_mix, df, left_on='出料批号', right_on='原料1批号')
 df2 = df2.loc[:, ['原料1批号_x','原料2批号_x', '出料批号',  '原料2批号_y','A料批号', 'B料批号_x']]
 df2 = df2.drop_duplicates()
 df2=pd.merge (df_liq, df2, left_on= '出料批号', right_on= '原料2批号_x')
 df2=df2.loc[:,['原料批号','出料批号_x','原料1批号_x','出料批号_y','原料2批号_y', 'A料批号', 'B料批号_x']]
 df2= df2.drop_duplicates()
#说明有第二个H6
 df3 = pd.merge(df_mix, df2, left_on='出料批号', right_on='原料2批号_y')
 df3 = df3.loc[:, ['原料1批号', '原料2批号','出料批号', '原料批号', '出料批号_x', '原料1批号_x','出料批号_y', 'A料批号', 'B料批号_x']]
 df3 = df3.drop_duplicates()
 df3.rename(columns={"出料批号_x": "出料批号1","出料批号_y": "出料批号2"},
              inplace=True)
 df3 = pd.merge(df_liq, df3, left_on='出料批号', right_on='原料2批号')
 df3 = df3.loc[:, [ '原料批号_y','出料批号1','原料1批号_x','出料批号2', '原料批号_x', '出料批号_x', '原料1批号', '出料批号_y','A料批号', 'B料批号_x']]
 df3 = df3.drop_duplicates()
 mask = df2.isnull().any(axis=1) # 找到含有NaN的行
 df2 = df2[mask]
 return df2,df3
def get_LN1(send_ID,df_710_,df_wsf,df_mix,df_liq):
 data = {
        '发货批号': [send_ID],
  }
 df = pd.DataFrame(data)
#710车间
 df = pd.merge(df_710_, df, on='发货批号')
 df = df.loc[:, ['A料批号','B料批号','发货批号']]
 df = df.drop_duplicates()
#卧包
 df = pd.merge(df_wsf, df, on='A料批号')
 df = df.loc[:, ['原料1批号','原料2批号','A料批号','B料批号_y','发货批号']]
 df = df.drop_duplicates()
#混料
 df2 = pd.merge(df_mix, df, left_on='出料批号', right_on='原料1批号')
 df2 = df2.loc[:, ['原料1批号_x','原料2批号_x', '出料批号',  '原料2批号_y','A料批号', 'B料批号_y',"发货批号"]]
 df2 = df2.drop_duplicates()
 df2=pd.merge (df_liq, df2, left_on= '出料批号', right_on= '原料2批号_x')
 df2=df2.loc[:,['原料批号','出料批号_x','原料1批号_x','出料批号_y','原料2批号_y', 'A料批号', 'B料批号_y',"发货批号"]]
 df2= df2.drop_duplicates()
 df3 = pd.merge(df_mix, df2, left_on='出料批号', right_on='原料2批号_y')
 df3 = df3.loc[:, ['原料1批号', '原料2批号','出料批号', '原料批号', '出料批号_x', '原料1批号_x','出料批号_y', 'A料批号', 'B料批号_y',"发货批号"]]
 df3 = df3.drop_duplicates()
 df3=df3.ffill()
 df3.rename(columns={"出料批号_x": "出料批号1","出料批号_y": "出料批号2"},
              inplace=True)
 df3 = pd.merge(df_liq, df3, left_on='出料批号', right_on='原料2批号')
 df3 = df3.loc[:, [ '原料批号_y','出料批号1','原料1批号_x','出料批号2', '原料批号_x', '出料批号_x', '原料1批号', '出料批号_y','A料批号', 'B料批号_y',"发货批号"]]
 df3 = df3.drop_duplicates()
 mask = df2.isnull().any(axis=1) # 找到含有NaN的行
 df2 = df2[mask]
 return df2,df3
#上传
uploaded_files = st.file_uploader("请上传生料车间表", accept_multiple_files=True)
@st.cache_data
def get_df(uploaded_files):
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
    df_mix['出料批号']=df_mix['出料批号'].apply(clean_original_data)
    df_liq['出料批号']=df_liq['出料批号'].apply(clean_original_data)
    df_710_['B料批号']=df_710_['B料批号'].apply(clean_original_data)
    df_sf['A料批号']=df_sf['A料批号'].apply(clean_original_data)
    df_fusion['生产批号']=df_fusion['生产批号'].apply(clean_original_data)
    df_lsf['A料批号']=df_lsf['A料批号'].apply(clean_original_data)
    df_wsf['A料批号']=df_wsf['A料批号'].apply(clean_original_data)
    df_gtl['A料批号']=df_gtl['A料批号'].apply(clean_original_data)
    df_gtlsf['A料批号']=df_gtlsf['A料批号'].apply(clean_original_data)
 return df_original,df_mix,df_liq,df_710_,df_sf,df_fusion,df_lsf,df_wsf,df_gtl,df_gtlsf,df_tanhua,df_hzy
df_original,df_mix,df_liq,df_710_,df_sf,df_fusion,df_lsf,df_wsf,df_gtl,df_gtlsf,df_tanhua,df_hzy=get_df(uploaded_files)
option = st.selectbox(
    '请选择你要追溯的产品',
    ('SK1', 'LK-T', 'LK-P',"AP5","PV-6","EP7-H","DFG","RG-S","CP09","CSA-3G","CSA-3E","LA1立包版本","LN1"))
st.write('你选择的是:', option)
send_ID = st.text_input('批号', '请输入相关批号')
if len(uploaded_files)>0:
 if option =="SK1" and send_ID!="请输入相关批号":
    df1=get_SK1_M13(send_ID,df_fusion,df_gtlsf,df_gtl,df_mix,df_liq)
    df2=get_SK1_M23(send_ID,df_710_,df_gtl,df_mix,df_fusion)
    st.write(df1,df2)
 elif option =="LK-T" and send_ID!="请输入相关批号":
    df = get_LK_T(send_ID,df_fusion,df_gtlsf,df_gtl,df_mix,df_liq)
    st.write(df)
 elif option =="LK-P" and send_ID!="请输入相关批号":
    df = get_LK_P(send_ID,df_fusion,df_gtlsf,df_gtl,df_mix)
    st.write(df)
 elif option == "AP5" and send_ID!="请输入相关批号":
    df,df1,df2=get_AP5(send_ID,df_lsf,df_wsf)
    st.write(df)
    st.write(df1)
    st.write(df2)
 elif option == "PV-6" and send_ID!="请输入相关批号":
    df2,df3=get_PV_6(send_ID,df_tanhua,df_fusion,df_mix,df_liq)
    st.write(df2, df3)
 elif option == "EP7-H" and send_ID!="请输入相关批号":
    df=get_EP7_H(send_ID,df_tanhua,df_fusion)
    st.write(df)
 elif option == "DFG" and send_ID!="请输入相关批号":
    df=get_DFG(send_ID, df_710_, df_wsf, df_mix, df_liq)
    st.write(df)
 elif option == "RG-S" and send_ID!="请输入相关批号":
    df=get_RG_S(send_ID, df_wsf, df_mix, df_liq)
    st.write(df)
 elif option == "CP09" and send_ID!="请输入相关批号":
    df1,df2=get_CG09(send_ID,df_710_,df_wsf,df_mix)
    st.write(df1)
    st.write(df2)
 elif option == "CSA-3G" and send_ID!="请输入相关批号":
    df=get_QCG_X4(send_ID, df_tanhua, df_gtl, df_mix, df_liq)
    st.write(df)
 elif option == "CSA-3E" and send_ID!="请输入相关批号":
    df=get_CSA_3E(send_ID, df_tanhua, df_fusion, df_mix, df_liq)
    st.write(df)
 elif option == "LA1立包版本" and send_ID!="请输入相关批号":
    df2,df3=get_LA1(send_ID,df_sf,df_lsf,df_mix,df_liq)
    st.write(df2, df3)
 elif option == "LN1" and send_ID!="请输入相关批号":
    df2,df3=get_LN1(send_ID, df_710_, df_wsf, df_mix, df_liq)
    st.write("看第二个")
    st.write(df2, df3)