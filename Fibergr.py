# coding=utf-8
import streamlit as st
import re
import  pandas as pd
#streamlit run Fiber.py
@st.cache_data
def get_data1(files,option):
  empty_df = pd.DataFrame()
  for file in files:
    df = pd.read_excel(file, engine="openpyxl", header=3)
    df.rename(columns={"Unnamed: 0": "NO", 'Unnamed: 1': '批号', 'Unnamed: 2': '批次号', 'Unnamed: 3': '送检日期'},
              inplace=True)
    df['批号'] = df['批号'].fillna(method='ffill')
    df['批次号'] = df['批次号'].fillna(method='ffill')
    df['NO'] = df['NO'].fillna(method='ffill')
    df = df.iloc[6:]
    df = df[['批号', option]]
    df = df.dropna(subset=[option])
    empty_df = pd.concat((empty_df, df), ignore_index=True)
  return empty_df

st.title("**纤维异物匹配**")
st.markdown("**有些数据匹配失败可能原因。1.批号格式不对，请到excel表中修改或下载后手动修改。2.确实不存在**")
uploaded_files = st.file_uploader("请上传投料数据", accept_multiple_files=True)
if len(uploaded_files) ==0:
    st.stop()
df1=get_data1(uploaded_files,"纤维异物（根）")
df1=df1.drop_duplicates(keep='first')
st.write("投料纤维异物情况")
st.write(df1)

file_path = st.file_uploader("请上传成品表")
if file_path==None:
    st.stop()
df= pd.read_excel(file_path, engine="openpyxl",header=0)
st.write("成品情况表")
st.write(df)
options = st.multiselect(
    '请选择匹配对象',
    df.columns.tolist(),
    )
for i in range(len(options)):
  st.write(options[i])
  df=pd.merge(df, df1,left_on=options[i],right_on="批号",how='left')
st.markdown("**结果有重复项**")
st.write(df)