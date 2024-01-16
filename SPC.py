# coding=utf-8
import numpy as np
import streamlit as st
import re
import  pandas as pd
#streamlit run SPC-ori.py
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
#可能会出现不存在的现象，所以需要设定数量
def get_SK1_M13(send_ID,df_half,df_fusion,df_gtlsf,df_gtl,df_mix,df_liq,n):
 # 半成品到货表
 if n>=1:
  data = {
         '批号': [send_ID.replace(" ", "")],
     }
  df = pd.DataFrame(data)
  df = pd.merge(df_half, df, on='批号')
  df=df.loc[:,[ 'B料批号(回货批号）',"批号"]]
  df= df.drop_duplicates()
  df.rename(columns={"B料批号(回货批号）": "发货批号"},
              inplace=True)
#融合车间
 if n >= 2:
  df = pd.merge(df_fusion, df, on='发货批号')
  df=df.loc[:,['生产批号', '投料批号',"发货批号",'批号']]
  df= df.drop_duplicates()
 if n >= 3:
#GTL筛分和GTL车间，会出现是SF和非SF的情况
  df=pd.merge (df_gtlsf, df, left_on= 'B料批号', right_on= '投料批号')
  df=df.loc[:,['A料批号', 'B料批号','生产批号',"发货批号",'批号']]
  df= df.drop_duplicates()
 if n>=4:
  df = pd.merge(df_gtl, df, on='A料批号')
  df=df.loc[:,['原料1批号','A料批号', 'B料批号','生产批号',"发货批号",'批号']]
  df= df.drop_duplicates()
 if n>=5:
#混料车间
  df=pd.merge (df_mix, df, left_on= '出料批号', right_on= '原料1批号')
  df=df.loc[:,['原料1批号_x','原料2批号','出料批号','A料批号', 'B料批号','生产批号',"发货批号",'批号']]
  df= df.drop_duplicates()
 if n>=6:
#XM-5
  df=pd.merge (df_liq, df, left_on= '出料批号', right_on= '原料2批号')
  df=df.loc[:,['原料批号','出料批号_x','原料1批号_x','出料批号_y','A料批号', 'B料批号','生产批号',"发货批号",'批号']]
  df= df.drop_duplicates()
 return df
def get_SK1_M23(send_ID,df_half,df_710_,df_gtl,df_mix,df_fusion,n):
#A料，B料
 if n>=1:
  data = {
         '批号': [send_ID.replace(" ", "")],
     }
  df = pd.DataFrame(data)
  df = pd.merge(df_half, df, on='批号')
  df=df.loc[:,[ 'B料批号(回货批号）',"批号"]]
  df= df.drop_duplicates()
  df.rename(columns={"B料批号(回货批号）": "发货批号"},
              inplace=True)
 if n>=2:
#710车间
  df = pd.merge(df_710_, df, on='发货批号')
  df=df.loc[:,['A料批号', 'B料批号',"发货批号","批号"]]
  df= df.drop_duplicates()
 if n>=3:
 #GTL车间
  df = pd.merge(df_gtl, df, on='A料批号')
  df=df.loc[:,["原料1批号","原料2批号",'A料批号', 'B料批号',"发货批号","批号"]]
  df= df.drop_duplicates()
 if n>=4:
#混料车间
  df=pd.merge (df_mix, df, left_on= '出料批号', right_on= '原料1批号')
  df=df.loc[:,["原料1批号_x","原料2批号_x",'出料批号','A料批号', 'B料批号',"发货批号","批号"]]
  df= df.drop_duplicates()
  df=df.ffill()
 if n>=5:
  df=pd.merge (df_liq, df, left_on= '出料批号', right_on= '原料2批号_x')
  df=df.loc[:,['原料批号','出料批号_x','原料1批号_x','出料批号_y','A料批号', 'B料批号',"发货批号","批号"]]
  df= df.drop_duplicates()
#R23
 if n>=6:
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
def get_QCG_D5(send_ID,df_tanhua,df_fusion,n):
#QCG-D5
 if n>=1:
  data = {
        'BC料批号': [send_ID],
    }
  df = pd.DataFrame(data)
  df = pd.merge(df_tanhua, df, on='BC料批号')
  df = df.loc[:, ['原料批号','BC料批号']]
  df = df.drop_duplicates()
 if n>=2:
#融合车间
  df_fusion=df_fusion[df_fusion["品名"]=="D5B"]
  dg=df_fusion.groupby('发货批号')
  TP9_list=[]
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
            TP9_list.append(df_t[i])
  data = {
    '其他产品':  TP9_list,
    'D11': D11_list,
    '原料批号': send_list
  }
  df_fusion = pd.DataFrame(data)
  df = pd.merge(df_fusion, df, on='原料批号')
  df= df.drop_duplicates()
 return df
def get_LK_T(send_ID,df_half,df_fusion,df_gtlsf,df_gtl,df_mix,df_liq,n):
 if n >= 1:
    data = {
            '批号': [send_ID.replace(" ", "")],
        }
    df = pd.DataFrame(data)
    df = pd.merge(df_half, df, on='批号')
    df = df.loc[:, ['B料批号(回货批号）', "批号"]]
    df = df.drop_duplicates()
    df.rename(columns={"B料批号(回货批号）": "发货批号"},
                  inplace=True)
 if n>=2:
#融合车间
  df = pd.merge(df_fusion, df, on='发货批号')
  df=df.loc[:,['投料批号', '生产批号',"发货批号", "批号"]]
  df= df.drop_duplicates()
 if n>=3:
#GTL筛分车间和GTL
  df=pd.merge (df_gtlsf, df, left_on= 'B料批号', right_on= '投料批号')
  df=df.loc[:,['A料批号', 'B料批号','生产批号',"发货批号", "批号"]]
  df= df.drop_duplicates()
 if n>=4:
  df = pd.merge(df_gtl, df, on='A料批号')
  df=df.loc[:,['原料1批号','A料批号', 'B料批号','生产批号',"发货批号", "批号"]]
  df= df.drop_duplicates()
 if n>=5:
#混料车间
  df=pd.merge (df_mix, df, left_on= '出料批号', right_on= '原料1批号')
  df=df.loc[:,["原料1批号_x","原料2批号",'出料批号','A料批号', 'B料批号','生产批号',"发货批号", "批号"]]
  df= df.drop_duplicates()
  df=df.ffill()
 if n>=6:
  df=pd.merge (df_liq, df, left_on= '出料批号', right_on= '原料2批号')
  df=df.loc[:,['原料批号','出料批号_x','原料1批号_x','出料批号_y','A料批号', 'B料批号','生产批号',"发货批号", "批号"]]
  df= df.drop_duplicates()
 return df
def get_LK_P(send_ID,df_half,df_fusion,df_gtlsf,df_gtl,df_mix,n):
 if n >= 1:
        data = {
            '批号': [send_ID.replace(" ", "")],
        }
        df = pd.DataFrame(data)
        df = pd.merge(df_half, df, on='批号')
        df = df.loc[:, ['B料批号(回货批号）', "批号"]]
        df = df.drop_duplicates()
        df.rename(columns={"B料批号(回货批号）": "发货批号"},
                  inplace=True)
 if n>=2:
#融合车间，可能会存在一个是SF，一个是A的情况
  df = pd.merge(df_fusion, df, on='发货批号')
  df=df.loc[:,['投料批号', '生产批号',"发货批号"]]
  df_all= df.drop_duplicates()
 if n>=3:
#GTL筛分车间
  df=pd.merge (df_gtlsf, df_all, left_on= 'B料批号', right_on= '投料批号')
  df=df.loc[:,['A料批号', 'B料批号','生产批号',"发货批号"]]
  df= df.drop_duplicates()
 if n>=4:
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
 if n>=5:
  df=pd.merge (df_mix, df, left_on= '出料批号', right_on= '原料1批号')
  df=df.loc[:,["原料1批号_x","原料2批号",'出料批号','A料批号', 'B料批号','生产批号',"发货批号"]]
  df= df.drop_duplicates()
  df = df.ffill()
#沥青
 if n>=6:
  df=pd.merge (df_liq, df, left_on= '出料批号', right_on= '原料2批号')
  df=df.loc[:,['原料批号','出料批号_x','原料1批号_x','出料批号_y','A料批号', 'B料批号','生产批号',"发货批号"]]
  df= df.drop_duplicates()
 return df
def get_CP5_M(send_ID,df_mix_screen,df_half,df_tanhua,df_fusion,df_original,n):
 if n>=1:
   data = {
        'BC料批号': [send_ID],
  }
   df = pd.DataFrame(data)
   df = pd.merge(df_tanhua, df, on='BC料批号')
   df = df.loc[:, ['原料批号','BC料批号']]
   df = df.drop_duplicates()
 if n>=2:
  df_fusion=df_fusion[df_fusion["品名"]=="CP5-HB"]
  dg=df_fusion.groupby('发货批号')
  TP9_list=[]
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
            TP9_list.append(df_t[i])
  data = {
    'CP5':  TP9_list,
    'D11': D11_list,
    '原料批号': send_list
   }
  df_fusion = pd.DataFrame(data)
  df = pd.merge(df_fusion, df, on='原料批号')
  df= df.drop_duplicates()
  # 在混料筛分表
     #BC料
 if n>=3:
     df=pd.merge (df_mix_screen, df, left_on= '产品批号', right_on= 'CP5')
     df = df.loc[:, ['原料批号_x', 'CP5', "D11", "原料批号_y","BC料批号"]]
     df = df.drop_duplicates()
 if n>=4:
     df = pd.merge(df_half, df, left_on='批号', right_on='原料批号_x')
     df = df.loc[:, ['B料批号(回货批号）','批号' ,'CP5', "D11", "原料批号_y", "BC料批号"]]
     df = df.drop_duplicates()
 if n>=5:
     df = pd.merge(df_original, df, left_on='生产批号', right_on='B料批号(回货批号）')
     df = df.loc[:, ['投料批号','B料批号(回货批号）', '批号', 'CP5', "D11", "原料批号_y", "BC料批号"]]
     df = df.drop_duplicates()
 return df
def get_AP5(send_ID,df_half,df_lsf,df_wsf,n):
 if n >= 1:
     data = {
         '批号': [send_ID.replace(" ", "")],
     }
     df = pd.DataFrame(data)
     df = pd.merge(df_half, df, on='批号')
     df = df.loc[:, ['B料批号(回货批号）', "批号"]]
     df = df.drop_duplicates()
     df.rename(columns={"B料批号(回货批号）": "发货批号"},
               inplace=True)
 return df
 if n>=2:
#立包,#卧包
  df_l=pd.merge (df_lsf, df, left_on= 'B料批号', right_on= '发货批号')
  df_l=df_l.loc[:,['原料1批号','原料2批号','A料批号' ,"B料批号","批号"]]
  df_l= df_l.drop_duplicates()
  df_l = df_l.ffill()
  df_w = pd.merge(df_wsf, df, left_on='B料批号', right_on='发货批号')
  df_w=df_w.loc[:,['原料1批号','原料2批号','A料批号' ,"B料批号","批号"]]
  df_w= df_w.drop_duplicates()
  df_w=df_w.ffill()
  df= pd.concat([df_l , df_w])
  df = df.drop_duplicates()
  return df
 if n>=3:
 #如果有A3-R5H则需要在混料表里面进行体现
  A3_R5H1=df['原料1批号'].tolist()
  A3_R5H_boolean=False
  A3_R5H2_boolean=False
  A3_R5H2=df['原料2批号'].tolist()
 #如果是A3-R5H
  for A3_R5H in A3_R5H1:
     if "A3-R5H" in str(A3_R5H):
         A3_R5H_boolean=True
  if A3_R5H_boolean==True:
   df1 = pd.merge (df_mix, df, left_on= '出料批号', right_on= '原料1批号')
   df1 = df1.loc[:, ['原料1批号_x', '原料2批号_x','出料批号','原料2批号_y', 'A料批号', "B料批号","批号"]]
   df1 = df1.drop_duplicates()
   df1= pd.merge(df_liq, df1, left_on='出料批号', right_on='原料2批号_x')
   df1 = df1.loc[:, ['原料批号', '出料批号_x', '原料1批号_x', '出料批号_y', '原料2批号_y', 'A料批号', 'B料批号',"批号"]]
   df1 = df1.drop_duplicates()
  for A3_R5H in A3_R5H2:
     if "A3-R5H" in str(A3_R5H):
         A3_R5H2_boolean=True
  if A3_R5H2_boolean==True:
       df2 = pd.merge(df_mix, df, left_on='出料批号', right_on='原料2批号')
       df2 = df2.loc[:, ['原料1批号_x', '原料2批号_x', '出料批号', '原料1批号_y', 'A料批号', "B料批号","批号"]]
       df2 = df2.drop_duplicates()
       df2 = pd.merge(df_liq, df2, left_on='出料批号', right_on='原料2批号_x')
       df2 = df2.loc[:, ['原料批号', '出料批号_x', '原料1批号_x', '出料批号_y', '原料1批号_y','A料批号', 'B料批号',"批号"]]
       df2 = df2.drop_duplicates()
  return df1,df2
def get_EP5T(send_ID,df_tanhua,df_fusion,df_half,n):
 if n>=1:
  data = {
        'BC料批号': [send_ID],
  }
  df = pd.DataFrame(data)
  df = pd.merge(df_tanhua, df, on='BC料批号')
  df = df.loc[:, ['原料批号','BC料批号']]
  df = df.drop_duplicates()
 if n>=2:
  df = pd.merge(df_fusion, df, left_on='发货批号', right_on='原料批号')
  df = df.loc[:, ['投料批号','原料批号', 'BC料批号']]
  df = df.drop_duplicates()
  # 对投料批号进行修正,可能会存在BUG
  df_D11=df [df ["投料批号"].str.contains("D11")]
  df_EP5 = df[df["投料批号"].str.contains("EP5")]
  df = pd.merge(df_D11, df_EP5, on='BC料批号')
  df = df.loc[:, ['投料批号_y','投料批号_x', '原料批号_x', 'BC料批号']]
 if n>=3:
     df=pd.merge (df_mix_screen, df, left_on= '产品批号', right_on= '投料批号_y')
     df = df.loc[:, ['原料批号','投料批号_y','投料批号_x', '原料批号_x', 'BC料批号']]
     df = df.drop_duplicates()
 if n>=4:
     df = pd.merge(df_half, df, left_on='批号', right_on='原料批号')
     df = df.loc[:, ['B料批号(回货批号）','原料批号','投料批号_y','投料批号_x', '原料批号_x', 'BC料批号']]
     df = df.drop_duplicates()
 return df
def get_PV_6(send_ID,df_tanhua,df_fusion,df_mix,df_liq,n):
 if n>=1:
   data = {
        'BC料批号': [send_ID],
  }
   df = pd.DataFrame(data)
#碳化车间
   df = pd.merge(df_tanhua, df, on='BC料批号')
   df = df.loc[:, [ '原料批号', "BC料批号"]]
   df = df.drop_duplicates()
 if n>=2:
#融合车间,分两种情况一种是PVH-6，一种是GHMG-C3和D11
  df = pd.merge(df_fusion, df, left_on='发货批号', right_on='原料批号')
  df = df.loc[:, ['投料批号', '原料批号', 'BC料批号']]
  df = df.drop_duplicates()
  df_h=df["投料批号"].tolist()
  test_boolean=False
  for test in df_h:
      if "PVH-6" in test:
          test_boolean=True
  if test_boolean==True:
      # 混料车间
      df = pd.merge(df_mix, df, left_on='出料批号', right_on='投料批号')
      df = df.loc[:, ["原料1批号", "原料2批号", '出料批号',  '原料批号', "BC料批号"]]
      df = df.drop_duplicates()
      df = df.ffill()
      # XH-3
      df = pd.merge(df_liq, df, left_on='出料批号', right_on='原料2批号')
      df = df.loc[:, ['原料批号_x', '出料批号_x', '原料1批号', '原料批号_y', "BC料批号"]]
      df = df.drop_duplicates()
 return df

def get_GKH(send_ID,df_mix_screen,df_half,df_tanhua,df_fusion,df_original,n):
    if n >= 1:
        data = {
            'BC料批号': [send_ID],
        }
        df = pd.DataFrame(data)
        # 碳化车间
        df = pd.merge(df_tanhua, df, on='BC料批号')
        df = df.loc[:, ['原料批号', "BC料批号"]]
        df = df.drop_duplicates()
    if n>=2:
        df_fusion=df_fusion[df_fusion["品名"]=="GK-HB"]
        df_t=df_fusion['投料批号'].tolist()
        df_f=df_fusion['发货批号'].tolist()
        GKBW_list = []
        GKC_list = []
        GR06_list = []
        D11_list = []
        send_list = []
        for i in range(len(df_t)):
            if (i+1)%3==1:
                GKBW_list.append(df_t[i])
                send_list.append(df_t[i])
            if (i+1)%3==2:
                GKC_list.append(df_t[i])
                GR06_list.append((df_t[i]))
            if (i+1)%3==0:
                D11_list.append(df_t[i])
        st.write(GKBW_list)
        st.write(send_list)
        st.write(GKC_list)
        st.write(GR06_list)
        st.write(D11_list)
        '''
        data = {
            'GKBW': GKBW_list,
            'GR0C': GKC_list,
            'GR06': GR06_list,
            'D11': D11_list,
            '原料批号': send_list}
        df_fusion = pd.DataFrame(data)
        df = pd.merge(df_fusion, df, on='原料批号')
        df = df.drop_duplicates()
        '''
    return df
def get_EP7_H(send_ID,df_mix_screen,df_half,df_tanhua,df_fusion,n):
 if n >= 1:
        data = {
            'BC料批号': [send_ID],
        }
        df = pd.DataFrame(data)
        # 碳化车间
        df = pd.merge(df_tanhua, df, on='BC料批号')
        df = df.loc[:, ['原料批号', "BC料批号"]]
        df = df.drop_duplicates()
 if n>=2:
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
 #混晒表
 if n>=3:
     df = pd.merge(df_mix_screen, df, left_on='产品批号', right_on='EP7T')
     df = df.loc[:, ['原料批号_x', 'EP7T','D11','原料批号_y', 'BC料批号']]
     df = df.drop_duplicates()
 if n>=4:
     df = pd.merge(df_half, df, left_on='B料批号(回货批号）', right_on='原料批号_x')
     df = df.loc[:, ['B料批号(回货批号）','原料批号', 'EP7T', 'D11', '原料批号_y', 'BC料批号']]
     df = df.drop_duplicates()
 return df
def get_DFG(send_ID,df_710_,df_wsf,df_mix,df_liq,n):
 if n>=1:
  data = {
        '发货批号': [send_ID],
  }
 df = pd.DataFrame(data)
#碳化车间
 if n>=2:
  df = pd.merge(df_710_, df, on='发货批号')
  df = df.loc[:, [ 'A料批号','B料批号', "发货批号"]]
  df = df.drop_duplicates()
#卧式
 if n>=3:
  df = pd.merge(df_wsf, df, on='A料批号')
  df = df.loc[:, ['原料1批号', 'A料批号','B料批号_y', "发货批号"]]
  df = df.drop_duplicates()
#混料
 if n>=4:
  df=pd.merge (df_mix, df, left_on= '出料批号', right_on= '原料1批号')
  df = df.loc[:, ['原料1批号_x','原料2批号','出料批号','原料1批号_y', 'A料批号','B料批号_y', "发货批号"]]
  df = df.drop_duplicates()
 if n>=5:
  df=pd.merge (df_liq, df, left_on= '出料批号', right_on= '原料2批号')
  df=df.loc[:,['原料批号','出料批号_x','原料1批号_x','原料1批号_y','A料批号', 'B料批号_y',"发货批号"]]
  df= df.drop_duplicates()
 return df
def get_RG_S(send_ID,df_wsf,df_mix,df_liq,n):
 if n>=1:
  data = {
        'B料批号': [send_ID],
   }
  df = pd.DataFrame(data)
 if n>=2:
  df = pd.merge(df_wsf, df, on='B料批号')
  df = df.loc[:, ['原料1批号', 'A料批号','B料批号']]
  df = df.drop_duplicates()
 if n>=3:
  df=pd.merge (df_mix, df, left_on= '出料批号', right_on= '原料1批号')
  df=df.loc[:,['原料1批号_x','原料2批号','出料批号','原料1批号_y','A料批号', 'B料批号']]
  df= df.drop_duplicates()
 if n>=4:
  df=pd.merge (df_liq, df, left_on= '出料批号', right_on= '原料2批号')
  df=df.loc[:,['原料批号','出料批号_x','原料1批号_x','出料批号_y','A料批号', 'B料批号']]
  df= df.drop_duplicates()
 return df

def get_RG_B(send_ID,df_gtlsf,df_gtl,df_mix,df_liq,df_wsf,n):
 if n >= 1:
        data = {
            '批号': [send_ID.replace(" ", "")],
        }
        df = pd.DataFrame(data)
        df = pd.merge(df_half, df, on='批号')
        df = df.loc[:, ['B料批号(回货批号）', "批号"]]
        df = df.drop_duplicates()
        df.rename(columns={"B料批号(回货批号）": "发货批号"},
                  inplace=True)
 if n>=2:
  df = pd.merge(df_gtlsf, df, left_on= 'B料批号', right_on= '发货批号')
  #df = df.loc[:, ['原料1批号', 'A料批号','B料批号']]
  df = df.drop_duplicates()
 if n>=3:
  df=pd.merge (df_gtl, df,on= 'A料批号')
  #df=df.loc[:,['原料1批号_x','原料2批号','出料批号','原料1批号_y','A料批号', 'B料批号']]
  df= df.drop_duplicates()
 if n>=4:
  df=pd.merge (df_mix, df,left_on= '出料批号', right_on= 'A料批号')
  #df=df.loc[:,['原料1批号_x','原料2批号','出料批号','原料1批号_y','A料批号', 'B料批号']]
  df= df.drop_duplicates()
 if n>=5:
  df=pd.merge (df_liq, df, left_on= '出料批号', right_on= '原料2批号')
  #df=df.loc[:,['原料批号','出料批号_x','原料1批号_x','出料批号_y','A料批号', 'B料批号']]
  df= df.drop_duplicates()
 return df
def get_CG09(send_ID,df_710_,df_wsf,df_mix,n):
 if n>=1:
  data = {
        '发货批号': [send_ID],
   }
  df = pd.DataFrame(data)
 if n>=2:
#碳化车间
  df = pd.merge(df_710_, df, on='发货批号')
  df = df.loc[:, ['A料批号', 'B料批号','发货批号']]
  df = df.drop_duplicates()
 if n>=3:
#卧包
  df=pd.merge (df_wsf, df, on= 'A料批号')
  df = df.loc[:, ['原料1批号','A料批号', 'B料批号_y','发货批号']]
  df = df.drop_duplicates()
 if n>=4:
  df = pd.merge(df_mix, df, left_on='出料批号', right_on='原料1批号')
  df = df.loc[:, ['原料1批号_x','原料2批号', '出料批号','A料批号', 'B料批号_y','发货批号']]
  df = df.drop_duplicates()
 if n>=5:
  df = pd.merge(df_liq, df, left_on='出料批号', right_on='原料2批号')
  df = df.loc[:, ['原料批号', '出料批号_x', '原料1批号_x', '出料批号_y', 'A料批号', 'B料批号_y','发货批号']]
  df = df.drop_duplicates()
 return df
#QCG-X4,G6B但写的是G6A
def get_QCG_X4(send_ID,df_tanhua,df_gtl,df_mix,df_liq,n):
 if n>=1:
  data = {
        'BC料批号': [send_ID],
  }
  df = pd.DataFrame(data)
#碳化车间
 if n>=2:
  df = pd.merge(df_tanhua, df, on='BC料批号')
  df = df.loc[:, ['原料批号','BC料批号']]
  df = df.drop_duplicates()
 #GTL筛分
 if n>=3:
  df = pd.merge(df_gtlsf, df, left_on='B料批号', right_on='原料批号')
  df = df.loc[:, [ 'A料批号','原料批号','BC料批号']]
  df = df.drop_duplicates()
#GTL车间
 if n>=4:
  df = pd.merge(df_gtl, df, on='A料批号')
  df = df.loc[:, ['原料1批号', 'A料批号','原料批号','BC料批号']]
  df = df.drop_duplicates()
 if n>=5:
  df = pd.merge(df_mix, df, left_on='出料批号', right_on='原料1批号')
  df = df.loc[:, ['原料1批号_x', '原料2批号', '出料批号',  'A料批号', '原料批号','BC料批号']]
  df = df.drop_duplicates()
  df = df.ffill()
 if n>=6:
  df=pd.merge (df_liq, df, left_on= '出料批号', right_on= '原料2批号')
  df=df.loc[:,['原料批号_x','出料批号_x','原料1批号_x','出料批号_y','A料批号','原料批号_y', 'BC料批号']]
  df= df.drop_duplicates()

 return df
def get_CSA_3E(send_ID,df_tanhua,df_fusion,df_mix,df_liq,n):
 if n>=1:
  data = {
        'BC料批号': [send_ID],
   }
  df = pd.DataFrame(data)
#碳化车间
 if n>=2:
  df = pd.merge(df_tanhua, df, on='BC料批号')
  df = df.loc[:, ['原料批号','BC料批号']]
  df = df.drop_duplicates()
#融合车间
 if n>=3:
  df = pd.merge(df_fusion, df, left_on='发货批号', right_on='原料批号')
  df = df.loc[:, ['投料批号','原料批号','BC料批号']]
  df = df.drop_duplicates()

#混料车间
 if n>=4:
  df = pd.merge(df_mix, df, left_on='出料批号', right_on='投料批号')
  df = df.loc[:, ['原料1批号','原料2批号','投料批号','原料批号','BC料批号']]
  df = df.drop_duplicates()
  df = df.ffill()
 if n>=5:
  df=pd.merge (df_liq, df, left_on= '出料批号', right_on= '原料2批号')
  df=df.loc[:,['原料批号_x','出料批号','原料1批号','投料批号','原料批号_y', 'BC料批号']]
  df= df.drop_duplicates()
 return df
def get_LA1(send_ID,df_sf,df_lsf,df_mix,df_liq,n):
#LA1
 if n>=1:
  data = {
        '发货批号': [send_ID],
  }
  df = pd.DataFrame(data)
 if n>=2:
  df = pd.merge(df_710_, df, on='发货批号')
  df = df.loc[:, ['A料批号', 'B料批号', '发货批号']]
  df = df.drop_duplicates()
  df = df.ffill()
 return df
def get_LN1(send_ID,df_710_,df_wsf,df_mix,df_liq,n):
 if n>=1:
  data = {
        '发货批号': [send_ID],
   }
  df = pd.DataFrame(data)
#710车间
 if n>=2:
  df = pd.merge(df_710_, df, on='发货批号')
  df = df.loc[:, ['A料批号','B料批号','发货批号']]
  df = df.drop_duplicates()
#卧包
 if n>=3:
  df = pd.merge(df_wsf, df, on='A料批号')
  df = df.loc[:, ['原料1批号','原料2批号','A料批号','B料批号_y','发货批号']]
  df = df.drop_duplicates()
#混料
 if n>=4:
  df = pd.merge(df_mix, df, left_on='出料批号', right_on='原料1批号')
  df = df.loc[:, ['原料1批号_x','原料2批号_x', '出料批号',  '原料2批号_y','A料批号', 'B料批号_y',"发货批号"]]
  df = df.drop_duplicates()
 if n>=5:
  df=pd.merge (df_liq, df, left_on= '出料批号', right_on= '原料2批号_x')
  df=df.loc[:,['原料批号','出料批号_x','原料1批号_x','出料批号_y','原料2批号_y', 'A料批号', 'B料批号_y',"发货批号"]]
  df= df.drop_duplicates()
 if n>=6:
  df = pd.merge(df_mix, df, left_on='出料批号', right_on='原料2批号_y')
  df = df.loc[:, ['原料批号', '出料批号_x', '原料1批号_x','出料批号_y','原料1批号', '原料2批号','出料批号', 'A料批号', 'B料批号_y',"发货批号"]]
  df = df.drop_duplicates()
  df=df.ffill()
  df.rename(columns={"出料批号_x": "出料批号1","出料批号_y": "出料批号2"},
              inplace=True)
 if n>=7:
  df = pd.merge(df_liq, df, left_on='出料批号', right_on='原料2批号')
  df = df.loc[:,['原料批号_y', '出料批号1', '原料1批号_x', '出料批号2','原料批号_x', '原料1批号', '原料2批号', '出料批号_y', 'A料批号','B料批号_y', "发货批号"]]
  df = df.drop_duplicates()
 return df

def get_T66BC(send_ID,df_half,df_710_,df_wsf,df_mix,df_liq,n):
    if n >= 1:
        data = {
            '批号': [send_ID.replace(" ", "")],
        }
        df = pd.DataFrame(data)
        df = pd.merge(df_half, df, on='批号')
        df = df.loc[:, ['B料批号(回货批号）', "批号"]]
        df = df.drop_duplicates()
        df.rename(columns={"B料批号(回货批号）": "发货批号"},
                  inplace=True)
    if n>=2:
        df = pd.merge(df_710_, df, on='发货批号')
        df = df.loc[:, ['A料批号', 'B料批号', "发货批号", "批号"]]
        df = df.drop_duplicates()
    if n>=3:
        df = pd.merge(df_wsf, df, on='A料批号')
        df = df.loc[:, ['原料1批号','原料2批号','A料批号', 'B料批号_y', "发货批号", "批号"]]
        df = df.drop_duplicates()
    if n>=4:
        #可能会存在2个
        df = pd.merge(df_mix, df, left_on='出料批号', right_on='原料1批号')
        df = df.loc[:, ['原料1批号_x','原料2批号_x','出料批号','原料2批号_y','A料批号', 'B料批号_y', "发货批号", "批号"]]
        df = df.drop_duplicates()
        df=df.ffill()
    if n >= 5:
        df = pd.merge(df_liq, df, left_on='出料批号', right_on='原料2批号_x')
        df = df.loc[:, ['原料批号','出料批号_x','原料1批号_x', '出料批号_y','原料2批号_y','A料批号', 'B料批号_y', "发货批号", "批号"]]
        df = df.drop_duplicates()
    if n>=6:
        df = pd.merge(df_mix, df, left_on='出料批号', right_on='原料2批号_y')
        df = df.loc[:, ['原料批号','出料批号_x','原料1批号_x', '出料批号_y','原料1批号','原料2批号','原料2批号_y','A料批号', 'B料批号_y', "发货批号", "批号"]]
        df = df.drop_duplicates()
        df = df.ffill()
    if n>=7:
        df = pd.merge(df_liq, df, left_on='出料批号', right_on='原料2批号')
        df = df.loc[:,
             ['原料批号_y', '出料批号_x', '原料1批号_x', '出料批号_y', '原料1批号', '原料批号_x','原料2批号', '原料2批号_y', 'A料批号',
              'B料批号_y', "发货批号", "批号"]]
        df = df.drop_duplicates()
    return df
def get_CS_8(send_ID,df_half,df_original,n):
    if n >= 1:
        data = {
            '批号': [send_ID.replace(" ", "")],
        }
        df = pd.DataFrame(data)
        df = pd.merge(df_half, df, on='批号')
        df = df.loc[:, ['B料批号(回货批号）', "批号"]]
        df = df.drop_duplicates()
        df.rename(columns={"B料批号(回货批号）": "发货批号"},
                  inplace=True)

    if n>=2:
            df = pd.merge(df_original, df, left_on='生产批号',right_on="发货批号")
            df = df.loc[:, ['投料批号', '生产批号', "发货批号", "批号"]]
            df = df.drop_duplicates()
    return df
#品名，批号
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
def get_mix_df(uploaded_files):
    df_mix_screen = pd.DataFrame()
    for file_path in uploaded_files:
        df_mix_screen1 = get_data(file_path, "23年", header=0)
        df_mix_screen = pd.concat((df_mix_screen, df_mix_screen1))
        df_mix_screen = clean_df_half(df_mix_screen, "原料品名", "原料批号")
    return df_mix_screen
def get_half_df(uploaded_files):
    df_half = pd.DataFrame()
    for file_path in uploaded_files:
        df_half1 = get_data(file_path, "Sheet1", header=1)
        df_half1 = clean_df_half(df_half1, "品名", "批号")
        df_half = pd.concat((df_half, df_half1))
    return df_half

def get_milling_df(uploaded_files):
    df_temp = pd.DataFrame()
    for milling_file in uploaded_files:
        df_rank = get_data(milling_file, "分级", header=1)
        df_bangxiao = get_data(milling_file, "棒销磨", header=1)
        df_ori_smash = get_data(milling_file, "生料粉碎", header=1)
        df_R_D_mix = get_data(milling_file, "研发料混料", header=1)
        df_Coulter_mix = get_data(milling_file, "犁刀混料", header=1)
        df_rank  = pd.concat((df_temp, df_rank1))
        df_bangxiao = pd.concat((df_temp, df_bangxiao1))
        df_ori_smash = pd.concat((df_temp, df_ori_smash1))
        df_R_D_mix = pd.concat((df_temp, df_R_D_mix1))
        df_Coulter_mix = pd.concat((df_temp, df_Coulter_mix1))
    return df_rank,df_bangxiao,df_ori_smash,df_R_D_mix,df_Coulter_mix
uploaded_files = st.file_uploader("请上传生料表", accept_multiple_files=True)
uploaded_files2 = st.file_uploader("请上传混料筛分表", accept_multiple_files=True)
uploaded_files3 = st.file_uploader("请上传半成品表", accept_multiple_files=True)
#uploaded_files4 = st.file_uploader("请上传制粉表", accept_multiple_files=True)
df_original,df_mix,df_liq,df_710_,df_sf,df_fusion,df_lsf,df_wsf,df_gtl,df_gtlsf,df_tanhua,df_hzy=get_df(uploaded_files)
df_mix_screen=get_mix_df(uploaded_files2)
df_half=get_half_df(uploaded_files3)

st.markdown("**混筛显示并查询**")
if len(uploaded_files2)>0:
 st.write(df_mix_screen)

st.markdown("**履历查询**")
option = st.selectbox(
    '请选择你要追溯的产品',
    ('SK1-BC', 'M23-BC','D5BC','M22BC',"CS-8", 'M12BC',"CP5-HBC","AP5-BC","EP5T-HBC","PVC-6-SQ","GKH-HBC",'T66BC',"EP7T-HBC","DFG","R(S)G-(B)BC","T7","AM12","G6BC","CAG-7MBC","LA1立包版本","LN1"))
st.write('你选择的是:', option)
number_before = st.selectbox(
    '请选择你要往前追溯几步',
    (1,2,3,4,5,6,7,8,9))
st.write('你选择的是:', option)
st.write('往前追溯:', number_before,"步")
send_ID = st.text_input('批号', '请输入相关批号')
if len(uploaded_files)>0:
 if option =="SK1-BC" and send_ID!="请输入相关批号":
    df=get_SK1_M13(send_ID,df_half,df_fusion,df_gtlsf,df_gtl,df_mix,df_liq,number_before)
    st.write(df)
 elif option =="M23-BC" and send_ID!="请输入相关批号":
    df=get_SK1_M23(send_ID,df_half,df_710_,df_gtl,df_mix,df_fusion,number_before)
    st.write(df)
 elif option =="CS-8" and send_ID!="请输入相关批号":
    df=get_CS_8(send_ID,df_half,df_original,number_before)
    st.write(df)
 elif option =="D5BC" and send_ID!="请输入相关批号":
    df=get_QCG_D5(send_ID,df_tanhua,df_fusion,number_before)
    st.write(df)
 elif option =="M22BC" and send_ID!="请输入相关批号":
    df = get_LK_T(send_ID,df_half,df_fusion,df_gtlsf,df_gtl,df_mix,df_liq,number_before)
    st.write(df)
 elif option =="M12BC" and send_ID!="请输入相关批号":
    df = get_LK_P(send_ID,df_half,df_fusion,df_gtlsf,df_gtl,df_mix,number_before)
    st.write(df)
 elif option == "CP5-HBC" and send_ID != "请输入相关批号":
    df=get_CP5_M(send_ID,df_mix_screen,df_half,df_tanhua,df_fusion,df_original,number_before)
    st.write(df)
 elif option == "AP5-BC" and send_ID!="请输入相关批号":
    if number_before<=2:
        df= get_AP5(send_ID, df_half, df_lsf, df_wsf, number_before)
        st.write(df)
    else:
      df1,df2 = get_AP5(send_ID, df_half, df_lsf, df_wsf, number_before)
      st.write(df1)
      st.write(df2)
 elif option == "EP5T-HBC" and send_ID != "请输入相关批号":
      df =  get_EP5T(send_ID,df_tanhua,df_fusion,df_half,number_before)
      st.write(df)
 elif option == "PVC-6-SQ" and send_ID!="请输入相关批号":
    df=get_PV_6(send_ID,df_tanhua,df_fusion,df_mix,df_liq,number_before)
    st.write(df)
 elif option == "GKH-HBC" and send_ID!="请输入相关批号":
    df=get_GKH(send_ID,df_mix_screen,df_half,df_tanhua,df_fusion,df_original,number_before)
    st.write(df)
 elif option == "T66BC" and send_ID!="请输入相关批号":
    df=get_T66BC(send_ID,df_half,df_710_,df_wsf,df_mix,df_liq,number_before)
    st.write(df)
 elif option == "EP7T-HBC" and send_ID!="请输入相关批号":
    df=get_EP7_H(send_ID,df_mix_screen,df_half,df_tanhua,df_fusion,number_before)
    st.write(df)
 elif option == "DFG" and send_ID!="请输入相关批号":
    df=get_DFG(send_ID, df_710_, df_wsf, df_mix, df_liq,number_before)
    st.write(df)
 elif option == "R(S)G-(B)BC" and send_ID != "请输入相关批号":
    st.write("有点问题")
 elif option == "T7" and send_ID!="请输入相关批号":
    df=get_RG_S(send_ID,df_wsf,df_mix,df_liq,number_before)
    st.write(df)
 elif option == "AM12" and send_ID!="请输入相关批号":
    df=get_CG09(send_ID,df_710_,df_wsf,df_mix,number_before)
    st.write(df)
 elif option == "G6BC" and send_ID!="请输入相关批号":
    df=get_QCG_X4(send_ID, df_tanhua, df_gtl, df_mix, df_liq,number_before)
    st.write(df)
 elif option == "CAG-7MBC" and send_ID!="请输入相关批号":
    df=get_CSA_3E(send_ID, df_tanhua, df_fusion, df_mix, df_liq,number_before)
    st.write(df)
 elif option == "LA1立包版本" and send_ID!="请输入相关批号":
    df=get_LA1(send_ID,df_sf,df_lsf,df_mix,df_liq,number_before)
    st.write(df)
 elif option == "M5B" and send_ID!="请输入相关批号":
    df=get_LN1(send_ID, df_710_, df_wsf, df_mix, df_liq,number_before)
    st.write(df)