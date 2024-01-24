# coding=utf-8
import  streamlit as st
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import  pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# 聚类
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
#streamlit run ceshi.py
#异常检测，无监督学习
def get_exception_data(input_list,df_index,df_value):
    my_set = list(set(input_list))
    result1 = [x for x in input_list if x == my_set[0]]
    result2 = [x for x in input_list if x == my_set[1]]
    exception_indices=[]
    exception_value = []
    if len(result1) >= len(result2):
        # 第二个是小的
        for i in range(len(input_list)):
            if input_list[i] == my_set[1]:
                exception_indices.append(df_index[i])
                exception_value.append(df_value[i])
        #第一个是小的
    else:
        for i in range(len(input_list)):
            if input_list[i] == my_set[0]:
                exception_indices.append(df_index[i])
                exception_value.append(df_value[i])
    return   exception_indices,exception_value
@st.cache_data
def get_data(file_path,sheet_name,header):
    dff1 = pd.read_excel(file_path, engine="openpyxl")
    return dff1

uploaded_file = st.file_uploader("上传xlsx文件")
if uploaded_file==None:
    st.stop()
df = pd.read_excel(uploaded_file, engine="openpyxl")
df=df.drop_duplicates()
#index
df_index=df.iloc[:,0]
#删除index
df=df.iloc[:,1:]
#处理数据
st.title('异常检测')
st.markdown("***需要上传文档，第一列为批号且文档不能出现空值***.")
options = st.multiselect(
    '选择指标:',
     df.columns.tolist())
if len(options)==0:
    st.stop()
df=df.loc[:, options]
data = df.values
tab1, tab2, tab3,tab4,tab5 = st.tabs(["LOF异常检测", "孤立森林检测", "Kmean异常检测","DBSCAN异常检测","高斯混合模型"])
with tab1:
   st.header("LOF异常检测")
   option1 = st.selectbox(
       'LOF请选择其中一个指标作图',
       tuple(options))
   st.write("你选择的是", option1)
   st.write("LOF参数设定")
   n_neighbors = st.slider('临近n_neighbors数', 2, 130, 20)
   contamination=st.slider('异常数据比例', 0.1, 0.5, 0.1)
   leaf_size = st.slider('子叶大小', 2, 130, 30)
   clf = LOF(n_neighbors=n_neighbors,contamination=contamination,leaf_size=leaf_size)
   res = clf.fit_predict(data)
   value1=df.loc[:, option1].tolist()
   exception_indices,exception_value = get_exception_data(res, df_index,value1)

   fig = go.Figure()
   # 绘制用户数折线图
   fig.add_trace(go.Scatter(x=df_index, y=df.loc[:, option1], name='趋势图', mode='lines'))
   # 绘制上下限、均值
   fig.add_trace(go.Scatter(x=df_index, y=[df.loc[:, option1].mean()] * len(df_index), name='均值',
                             mode='lines', line_color="red"))
   fig.add_trace(
       go.Scatter(x=exception_indices, y=exception_value, mode='markers', name='LOF异常点', marker_color='red'))
   st.plotly_chart(fig, theme="streamlit", use_container_width=True)
   st.write(exception_indices)
with tab2:
   st.header("孤立森林检测")
   option2 = st.selectbox(
       '孤立森林请选择其中一个指标作图',
       tuple(options))
   st.write("你选择的是", option2)
   n_estimators = st.slider('孤立森林生成的随机树数量', 2, 130, 100)
   contamination2=st.slider('孤立森林异常数据比例', 0.1, 0.5, 0.3)
   model = IsolationForest(n_estimators=n_estimators,
                           max_samples='auto',
                           contamination=contamination2
                          )
   # 训练模型
   model.fit(data)
   # 预测 decision_function 可以得出 异常评分
   df['scores'] = model.decision_function(data)
   #  predict() 函数 可以得到模型是否异常的判断，-1为异常，1为正常
   anomaly_list = model.predict(data)
   # 定义Iso异常点
   value2 = df.loc[:, option2].tolist()
   iso_exception_indices,iso_exception_value= get_exception_data(anomaly_list, df_index,value2)
   fig1 = go.Figure()
   # 绘制用户数折线图
   fig1.add_trace(go.Scatter(x=df_index, y=df.loc[:, option2], name='趋势图', mode='lines'))
   # 绘制上下限、均值
   fig1.add_trace(go.Scatter(x=df_index, y=[df.loc[:, option2].mean()] * len(df_index), name='均值',
                             mode='lines', line_color="red"))
   # 其他值
   fig1.add_trace(go.Scatter(x=iso_exception_indices, y=iso_exception_value, mode='markers', name='孤立森林异常点',
                             marker_color='red'))
   st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
   st.write(iso_exception_value)
with tab3:
    st.header("Kmean")
    option3 = st.selectbox(
        'Kmean请选择其中一个指标作图',
        tuple(options))
    st.write("你选择的是", option3)
    estimator = KMeans(n_clusters=2)  # 构造聚类器
    estimator.fit(data)  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    value3 = df.loc[:, option3].tolist()
    Kmean_exception_indices,Kmean_exception_value = get_exception_data(label_pred, df_index,value3)
    fig2 = go.Figure()
    # 绘制用户数折线图
    fig2.add_trace(go.Scatter(x=df_index, y=df.loc[:, option3], name='趋势图', mode='lines'))
    fig2.add_trace(go.Scatter(x=df_index, y=[df.loc[:, option3].mean()] * len(df_index), name='均值',
                              mode='lines', line_color="red"))
    # 其他值
    fig2.add_trace(go.Scatter(x=Kmean_exception_indices, y=Kmean_exception_value, mode='markers', name='K均值异常点',
                              marker_color='red'))
    st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
    st.write(Kmean_exception_value)
with tab4:
    st.header("DBSCAN")
    option4 = st.selectbox(
        'DBSCAN请选择其中一个指标作图',
        tuple(options))
    st.write("你选择的是", option4)
    min_samples = st.slider('最小样本数', 1, 50, 15)
    distance = st.selectbox(
        '距离公式',
        ('euclidean', 'manhattan', 'chebyshev'))
    leaf_size4 = st.slider('DBSCAN子叶大小', 1, 130, 30)
    y_pred = DBSCAN(metric=distance,min_samples=min_samples,leaf_size=leaf_size4).fit_predict(data)
    value4 = df.loc[:, option4].tolist()
    DBSCAN_exception_indices,DBSCAN_exception_value = get_exception_data(y_pred, df_index,value4)
    fig3 = go.Figure()
    # 绘制用户数折线图
    fig3.add_trace(go.Scatter(x=df_index, y=df.loc[:, option4], name='趋势图', mode='lines'))
    # 绘制上下限、均值
    fig3.add_trace(go.Scatter(x=df_index, y=[df.loc[:, option4].mean()] * len(df_index), name='均值',
                              mode='lines', line_color="red"))
    # 其他值
    fig3.add_trace(go.Scatter(x=DBSCAN_exception_indices, y=DBSCAN_exception_value, mode='markers', name='DBSCAN异常点',
                              marker_color='red'))
    st.plotly_chart(fig3, theme="streamlit", use_container_width=True)
    st.write(DBSCAN_exception_value)
with tab5:
    st.header("高斯混合模型")
    option5 = st.selectbox(
        '高斯混合模型请选择其中一个指标作图',
        tuple(options))
    st.write("你选择的是", option5)
    max_iter=st.slider('EM最大迭代次数', 1, 150, 100)
    gmm = GaussianMixture(n_components=2,max_iter=max_iter)
    gmm.fit(data)
    # 预测每个数据点的聚类标签
    labels = gmm.predict(data)
    # 创建折线图
    value5 = df.loc[:, option5].tolist()
    gmm_exception_indices,gmm_exception_value = get_exception_data(labels, df_index,value5)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df_index, y=df.loc[:, option5], name='趋势图', mode='lines'))
    # 绘制上下限、均值
    fig4.add_trace(go.Scatter(x=df_index, y=[df.loc[:, option5].mean()] * len(df_index), name='均值',
                              mode='lines', line_color="red"))

    fig4.add_trace(go.Scatter(x=gmm_exception_indices, y=gmm_exception_value, mode='markers', name='高斯混合异常点',
                              marker_color='red'))

    st.plotly_chart(fig4, theme="streamlit", use_container_width=True)
    st.write(gmm_exception_value)
