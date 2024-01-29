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
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import  numpy as np
from collections import Counter
from pyod.models.abod import ABOD
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
                all_wrong_list.append(df_index[i])
        #第一个是小的
    else:
        for i in range(len(input_list)):
            if input_list[i] == my_set[0]:
                exception_indices.append(df_index[i])
                exception_value.append(df_value[i])
                all_wrong_list.append(df_index[i])
    return   exception_indices,exception_value
@st.cache_data
def get_data(file_path,sheet_name,header):
    dff1 = pd.read_excel(file_path, engine="openpyxl")
    return dff1
uploaded_file = st.file_uploader("上传xlsx文件")
if uploaded_file==None:
    st.stop()
df = pd.read_excel(uploaded_file, engine="openpyxl")
#file_path="D:\回溯数据来源\\成品数据改.xlsx"
#df = pd.read_excel(file_path, engine="openpyxl")
df2 = pd.read_excel(uploaded_file, engine="openpyxl")
df=df.drop_duplicates()
#
all_wrong_list=[]
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
df=df.loc[:, options]
data = df.values
tab1, tab2, tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10 = st.tabs(["LOF", "孤立森林", "Kmean","DBSCAN","高斯混合模型","One-Class SVM","自编码器","随机投影","ABOD","总"])

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
   st.plotly_chart(fig1)
   st.write(iso_exception_indices)
with tab3:
    st.header("Kmean")
    option3 = st.selectbox(
        'Kmean请选择其中一个指标作图',
        tuple(options))
    st.write("你选择的是", option3)
    n_clusters = st.selectbox(
        '聚类数量',
        ('2', '3'))
    n_clusters=int(n_clusters)
    max_iter3 = st.slider('最大迭代数', 1, 500, 300)
    estimator = KMeans(n_clusters=n_clusters,max_iter=max_iter3)  # 构造聚类器
    estimator.fit(data)  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    if n_clusters==3:
     value3 = df.loc[:, option3]
     value3=value3.to_frame()
     value3['label']=label_pred.tolist()
     value3['index']=df_index
     counter = Counter(label_pred)
    # 找到出现次数最多的元素,然后找出不是他的元素就是异常值
     max_count_element = max(counter.keys(), key=lambda x: counter[x])
     value3=value3[value3['label']!= max_count_element]
     #批号
     Kmean_exception_indices=value3['index'].tolist()
     #值
     Kmean_exception_value=value3[option3].tolist()
    if n_clusters==2:
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
    st.write(Kmean_exception_indices)
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
    st.write(DBSCAN_exception_indices)
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
    st.write(gmm_exception_indices)
with tab6:
    st.header("One-Class SVM异常检测")
    option6 = st.selectbox(
        'One-Class SVM请选择其中一个指标作图',
        tuple(options))
    st.write("你选择的是", option6)
    st.write("0ne-Class SVM参数设定")
    nu = st.slider('训练误差', 0.0, 1.0, 0.5)
    clf2 = OneClassSVM( nu=nu)
    clf.fit(data)
    outliers2 = clf2.fit_predict(data)
    value6 = df.loc[:, option6].tolist()
    one_class_exception_indices, one_class_exception_value = get_exception_data(outliers2, df_index, value6)
    fig5= go.Figure()
    fig5.add_trace(go.Scatter(x=df_index, y=df.loc[:, option6], name='趋势图', mode='lines'))
    # 绘制上下限、均值
    fig5.add_trace(go.Scatter(x=df_index, y=[df.loc[:, option6].mean()] * len(df_index), name='均值',
                              mode='lines', line_color="red"))

    fig5.add_trace(go.Scatter(x=one_class_exception_indices, y=one_class_exception_value, mode='markers', name='One-Class SVM',
                              marker_color='red'))
    st.plotly_chart(fig5, theme="streamlit", use_container_width=True)
    st.write(one_class_exception_indices)
with tab7:
    st.header("自编码器异常检测")
    option7 = st.selectbox(
        '自编码器请选择其中一个指标作图',
        tuple(options))
    st.write("你选择的是", option7)
    st.write("自编码器参数设定")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    clf7 = PCA(n_components=1)
    data_pca = clf7.fit_transform(data_scaled)
    reconstructed_data = clf7.inverse_transform(data_pca)
    reconstruction_error = np.mean((data_scaled - reconstructed_data) ** 2, axis=1)
    threshold = np.percentile(reconstruction_error, 95)
    outliers7 = (reconstruction_error < threshold).astype(int)
    value7 = df.loc[:, option7].tolist()
    zbm_exception_indices, zbm_exception_value = get_exception_data(outliers7, df_index, value7)
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=df_index, y=df.loc[:, option7], name='趋势图', mode='lines'))
    # 绘制上下限、均值
    fig6.add_trace(go.Scatter(x=df_index, y=[df.loc[:, option7].mean()] * len(df_index), name='均值',
                              mode='lines', line_color="red"))
    fig6.add_trace(
        go.Scatter(x=zbm_exception_indices, y=zbm_exception_value, mode='markers', name='自编码器',
                   marker_color='red'))
    st.plotly_chart(fig6, theme="streamlit", use_container_width=True)
    st.write(zbm_exception_indices)
with tab8:
    st.header("随机投影异常检测")
    option8 = st.selectbox(
        '随机投影请选择其中一个指标作图',
        tuple(options))
    st.write("你选择的是", option8)
    st.write("随机投影参数设定")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    clf8 = PCA(n_components=1, random_state=0)
    data_pca = clf8.fit_transform(data_scaled)
    outliers8 = ((data_scaled - clf8.inverse_transform(data_pca)) ** 2).sum(axis=1)
    outliers8 = np.where(outliers8 > np.percentile(outliers8, 95), 1, 0)
    value8 = df.loc[:, option8].tolist()
    sjt_exception_indices, sjt_exception_value = get_exception_data(outliers8, df_index, value8)
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=df_index, y=df.loc[:, option8], name='趋势图', mode='lines'))
    # 绘制上下限、均值
    fig7.add_trace(go.Scatter(x=df_index, y=[df.loc[:, option8].mean()] * len(df_index), name='均值',
                              mode='lines', line_color="red"))
    fig7.add_trace(
        go.Scatter(x=sjt_exception_indices, y=sjt_exception_value, mode='markers', name='随机投影',
                   marker_color='red'))
    st.plotly_chart(fig7, theme="streamlit", use_container_width=True)
    st.write(sjt_exception_indices)
with tab9:
    option9 = st.selectbox(
        'ABOD请选择其中一个指标作图',
        tuple(options))
    st.write("你选择的是", option9)
    clf_name = 'ABOD'
    clf = ABOD()
    clf.fit(data)
    # 返回训练数据X_train上的异常标签和异常分值
    y_train_pred = clf.labels_  # 返回训练数据上的分类标签 (0: 正常值, 1: 异常值)
    y_train_scores = clf.decision_scores_  # 返回训练数据上的异常值 (分值越大越异常)
    value9 = df.loc[:, option9].tolist()
    ABOD_exception_indices, ABOD_exception_value = get_exception_data(y_train_pred, df_index, value9)
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=df_index, y=df.loc[:, option9], name='趋势图', mode='lines'))
    # 绘制上下限、均值
    fig7.add_trace(go.Scatter(x=df_index, y=[df.loc[:, option9].mean()] * len(df_index), name='均值',
                              mode='lines', line_color="red"))
    fig7.add_trace(
        go.Scatter(x=ABOD_exception_indices, y=ABOD_exception_value, mode='markers', name='ABOD',
                   marker_color='red'))
    st.plotly_chart(fig7, theme="streamlit", use_container_width=True)
    st.write(ABOD_exception_indices)
with tab10:
    counter = Counter(all_wrong_list)
    sorted_counter = sorted(filter(lambda x: x[1] >=5, counter.items()), key=lambda x: x[1], reverse=True)
    for key, value in sorted_counter:
        st.write(f'元素 {key} 出现的次数是 {value}')
