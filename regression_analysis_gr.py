# coding=utf-8
import streamlit as st
import  pandas as pd
import plotly.express as px
from scipy import stats
import statsmodels.api as sm
import  numpy as np
import seaborn as sns
from pycaret.datasets import get_data
from pycaret.classification import *
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.ticker as mticker
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import xgboost as xgb
from xgboost import plot_importance
import matplotlib
import plotly.figure_factory as ff
from scipy.special import gamma
from scipy.stats import norm
from statsmodels.stats.diagnostic import lilliefors
#streamlit run regression_analysis.py
from sklearn.inspection import PartialDependenceDisplay
def calculate_pooled_standard_deviation(groups):
    means = [np.mean(group) for group in groups]
    sizes = [len(group) for group in groups]
    sums = [sum((x - mean) ** 2 for x in group) for group, mean in zip(groups, means)]
    total_sum = sum(sums)
    degrees_of_freedom = sum(size - 1 for size in sizes)
    pooled_std_dev = np.sqrt(total_sum/degrees_of_freedom)
    c4 = np.sqrt(2/degrees_of_freedom)*gamma((degrees_of_freedom+1)/2)/gamma(degrees_of_freedom/2)
    pooled_std_dev_unbiased = pooled_std_dev/c4
    return pooled_std_dev_unbiased
def get_PPM(flat_data,size,USL,LSL):
    data = [flat_data[i:i + size] for i in range(0, len(flat_data), size)]
    mean = np.mean(flat_data)
    std = np.std(flat_data, ddof=1)
    N = len(flat_data)
    pool_sd = calculate_pooled_standard_deviation(data)
    percent_below_lsl = sum(1 for element in flat_data if element < LSL) / N
    percent_above_usl = sum(1 for element in flat_data if element > USL) / N
    data_zt = {'性能': ["PPM<规格下限"
        , "PPM大于规格上限", "合计PPM"],
               '观测': [percent_below_lsl, percent_above_usl,percent_below_lsl+ percent_above_usl ],
               '预测整体': [(norm.cdf(LSL, mean, std)) , (1 - norm.cdf(USL, mean, std)) ,
                            (norm.cdf(LSL, mean, std)) +(1 - norm.cdf(USL, mean, std)) ],
               '预测组内': [(1 - norm.cdf(USL, mean, pool_sd)) , (1 - norm.cdf(USL, mean, pool_sd)) ,(1 - norm.cdf(USL, mean, pool_sd)) +(1 - norm.cdf(USL, mean, pool_sd))]}
    df_zt = pd.DataFrame(data_zt)
    return df_zt
def Get_P_Value(P):
    if P>0.05:
        return "符合正态分布"
    else:
        return "不符合正态分布"
def get_CPM(flat_data,size,USL,LSL,SL):
    data = [flat_data[i:i + size] for i in range(0, len(flat_data), size)]
    mean = np.mean(flat_data)
    std = np.std(flat_data, ddof=0)
    pool_sd = calculate_pooled_standard_deviation(data)
    std = sum((x - SL) ** 2 for x in flat_data)
    std = sum((x - SL) ** 2 for x in flat_data) / len(flat_data)
    std = np.sqrt(std)
    CPM=""
    if (USL - LSL) / 2 + LSL == SL:
        a = abs(USL - LSL)
        b = 6 * std
        CPM=a/b
    else:
        a = min(abs(USL - SL), abs(SL - LSL))
        b = 3 * std
        CPM=a/b
    return CPM
def Get_CPK(data,size,USL,LSL):
    # 总平均数
    mu = np.mean(data)
    # 计算每行平均数
    row_means = [sum(row) / len(row) for row in data]
    # 计算行的每个值和平均数之差的平方
    squared_diffs = [[(x - mean) ** 2 for x in row] for row, mean in zip(data, row_means)]
    # 对所有结果求和
    total_sum = sum(sum(row) for row in squared_diffs)
    # 计算自由度和标准差
    degrees_of_freedom = len(data) * (size - 1)
    pooled_std_dev = np.sqrt(total_sum / degrees_of_freedom)
    # 计算C4(d+1)，【标准差修正用】
    c4 = np.sqrt(2 / degrees_of_freedom) * gamma((degrees_of_freedom + 1) / 2) / gamma(degrees_of_freedom / 2)
    # 修正标准差，即组内标准差
    pooled_std_dev_unbiased = pooled_std_dev / c4
    CP=(USL-LSL)/6*pooled_std_dev_unbiased
    CPU = (USL - mu) / (3 * pooled_std_dev_unbiased)
    CPL = (mu - LSL) / (3 * pooled_std_dev_unbiased)
    CPK = min(CPU, CPL)
    return pooled_std_dev_unbiased,CP,CPU,CPL,CPK
#size=1
def Get_CPK_no_size(data,USL,LSL):
    mr_data = [abs(data[i] - data[i - 1]) for i in range(1, len(data))]
    mean = np.mean(data)
    Rbar = np.mean(mr_data)
    #组内标准差
    sigma = Rbar / 1.128
    CP = (USL - LSL) / (6 * sigma)
    CPU = (USL - mean) / (3 * sigma)
    CPL = (mean - LSL) / (3 * sigma)
    CPK = min(CPU, CPL)
    return sigma,CP,CPU,CPL,CPK
def Get_PP(flat_data,USL,LSL):
    mean = np.mean(flat_data)
    std = np.std(flat_data, ddof=1)
    pp=(USL-LSL)/6*std
    ppu=(USL-mean)/3*std
    ppl=(mean-LSL)/3*std
    ppk=min(ppu,ppl)
    return std,pp,ppu,ppl,ppk
#判断批号是否异常
def highlight_short_strings(s):
    for i in range(len(s) - 7):
        if s[i:i+8].isdigit():
          return 'background-color: white'
    else:
        return 'background-color: red'
def get_trend_plot(df_index,df):
    for i in range(len(df.columns)):
        fig = go.Figure()
        # 绘制用户数折线图
        name=df.columns[i]
        st.header(name+"趋势图")
        fig.add_trace(go.Scatter(x=df_index, y=df.iloc[:,i], name=name ,mode='lines+markers'))
        # 绘制上下限、均值
        st.plotly_chart(fig)
def get_zt(data,test):
  for i in range(len(data.columns)):
    data_1=data.iloc[:,i]
    # D'Agostino and Pearson omnibus normality test
    k2, p_k = stats.normaltest(data_1)
    p_kc = Get_P_Value(p_k)
    # Shapiro-Wilk normality test
    w, p_w = stats.shapiro(data_1)
    p_wc = Get_P_Value(p_w)
    # Anderson_darling Test
    stat, critical_values, p_s = stats.anderson(data_1)
    if stat<critical_values[2]:
        p_sc = "符合正态分布"
    else:
        p_sc = "不符合正态分布"
    # 进行Kolmogorov-Smirnov检验
    d, p_d = stats.kstest(data_1, 'norm')
    p_dc = Get_P_Value(p_d)
    # 进行Kolmogorov-Smirnov-lilliefors检验
    li, p_li = lilliefors(data_1)
    p_lic = Get_P_Value(p_li)
    data_zt = {'检测方法': ["D'Agostino and Pearson omnibus normality test"
        , "Shapiro-Wilk normality test", "Anderson_darling Test", "Kolmogorov-Smirnov Test","Kolmogorov-Smirnov-lilliefors Test"],
               '统计量': [k2, w, stat, d,li], 'P值': [p_k, p_w, critical_values[2], p_d,p_li],
               '判定': [p_kc, p_wc, p_sc, p_dc,p_lic]}
    df_zt = pd.DataFrame(data_zt)
    st.markdown("***"+test[i]+"***")
    st.write(df_zt)
    Fig = plt.figure(figsize=(6, 4))
    (norm_quantiles, actual_values), (slope, intercept, r) = stats.probplot(data_1, dist="norm")
    plt.grid(color='#eee', linestyle='-', linewidth=0.5, zorder=0)
    ax = plt.gca()
    # 配置所有轴脊柱的颜色和线型
    for spine in ax.spines.values():
        spine.set_color('#808080')
        spine.set_linestyle('-')
        spine.set_linewidth(0.5)
    x = actual_values
    y = stats.norm.cdf(norm_quantiles)
    plt.plot(x, y, 'o', color='#0054a6', label='Data Points')
    # 使用 numpy 的 polyfit 方法从 x 和 y 拟合一条直线
    m, b = np.polyfit(x, y, 1)
    # 绘制拟合的直线
    plt.plot(x, m * x + b, color='#931313', label='')
    # 设置标签和标题
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.xlabel('Observations')
    plt.ylabel('percent')
    plt.title('Normal Probability Plot')
    plt.grid(True)
    st.pyplot(Fig)

st.set_option('deprecation.showPyplotGlobalUse', False)
uploaded_file = st.file_uploader("上传xlsx文件")
if uploaded_file==None:
    st.stop()
#去掉批号的
df = pd.read_excel(uploaded_file, engine="openpyxl")
#file_path="D:\回溯数据来源\\成品数据改.xlsx"
#没有去掉批号的
df2 = pd.read_excel(uploaded_file, engine="openpyxl")
df2=df2.drop_duplicates()
df=df2.drop_duplicates()
#index
df_index=df.iloc[:,0]
#删除index
df=df.iloc[:,1:]
#处理数据
st.title('SPC分析')
contatiner2=st.container(border=True)
with contatiner2:
   test = st.multiselect(
        '请选择输入:',
        df.columns.tolist())
   st.write("你选择的变量", test)
   data_1 = df.loc[:, test]
   tabb,tabc, tabc0, tabc1, tabc2= st.tabs(["数据明细",'趋势图', '变量分布', '正态分布检验', "CPK计算"])
   if len(test)>0:
    with tabb:
        # 数据明细表
        st.markdown("**数据明细**")
        st.write("批号红色部分表示批号异常")
        st.dataframe(df2.style.applymap(highlight_short_strings, subset=[df2.columns.tolist()[0]]))
    with tabc:
        get_trend_plot(df_index,data_1)
    with tabc0:
        st.markdown("**变量分布**")
        st.write(data_1.describe())
        st.markdown("**散点图**")
        fig=sns.pairplot(data_1)
        st.pyplot(fig)
        st.markdown("**箱式图**")
        fig2 = px.box(data_1,points="all")
        fig2.update_traces(quartilemethod="exclusive")
        st.plotly_chart(fig2)
    with tabc1:
        st.markdown("**正态分布检验**")
        data_2 = data_1.T.values.tolist()
        group_labels = test
        fig = ff.create_distplot(data_2, group_labels=group_labels)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        st.markdown("**Note Anderson_darling Test只要统计量大于评判值P值就显著**")
        st.write("正态分布检验结果")
        get_zt(data_1,test)
    with tabc2:
        st.markdown("**CPK计算**")
        st.write("有点问题")
container4 = st.container(border=True)
with container4:
 st.markdown("***相关系数分析***.")
 options = st.multiselect(
        '选择指标:',
        df.columns.tolist())
 if len(options)!=0:
  st.write("你选择的是:", options)
  data = df.loc[:, options]
  tab1, tab2, tab3= st.tabs(["Pearson相关系数", "Kendall相关系数", "Spearman相关系数"])
  with tab1:
    pearson = data.corr(method="pearson")
    # 绘制热力图
    fig1 = px.imshow(pearson, text_auto=True)
    st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
    st.write("|r|≥0.8,高度相关")
    st.write("0.8＞|r|≥0.5,中等相关")
    st.write("0.5＞|r|≥0.3,低相关")
    st.write("0.3＞|r|,无相关")
  with tab2:
    kendall = data.corr(method="kendall")
    # 绘制热力图
    fig2 = px.imshow(kendall, text_auto=True)
    st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
    st.write("|r|越大越相关")
  with tab3:
    spearman = data.corr(method="spearman")
    # 绘制热力图
    fig3 = px.imshow(spearman, text_auto=True)
    st.plotly_chart(fig3, theme="streamlit", use_container_width=True)
    st.write("|r|越大越相关")
container5=st.container(border=True)
with container5:
 st.markdown("***回归分析***.")
 options_x = st.multiselect(
        '选择自变量:',
        df.columns.tolist())
 st.write("你选择的自变量是:", options_x)
 data_x = df.loc[:, options_x]
 option_y = st.selectbox(
        '请选择因变量',
        tuple(df.columns.tolist()))
 data_y = df.loc[:, option_y]
 st.write("你选择的因变量", option_y)
 if len(options_x)!=0:
  #首先进行VIF测试
  tab5, tab6, tab7, tab8,tab9,tab10,tab11,tab12,tab13,tab14,tab15 = st.tabs(["多重共线性", "OLS", "多项式", "决策树","随机森林","支持向量","神经网络","XGBoost","GBTD","CatBoost","LightBoost"])
  with tab5:
      st.header("一般VIF大于10则说明变量存在严重的多重共线性，在线性回归中需删除这些变量")
      data_x[option_y]=data_y.tolist()
      data_x = sm.add_constant(data_x)
      name = data_x.columns
      x = np.matrix(data_x)
      VIF_list = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
      VIF = pd.DataFrame({'feature': name, "VIF": VIF_list})
      VIF = VIF.drop(0)
      st.write(VIF)
  with tab6:
      st.write('P值＜0.05或者|t|＞1.96')
      agree = st.checkbox('是否去中心化')
      if agree:
        mean = np.mean(data_x, axis=0)
        data_x=data_x-mean
        data_x.iloc[:, 0] += 1
        model = sm.OLS(data_y,data_x.drop(option_y,axis=1)).fit()
      else:
         model = sm.OLS(data_y, data_x.drop(option_y, axis=1)).fit()
      st.write(model.summary())
  with tab7:
    col01, col02 = st.columns(2)
    with col01:
      data_x=data_x.drop(option_y, axis=1)
      data_x=data_x.drop("const", axis=1)
      X = np.array(data_x)
      y = np.array(data_y)
      # 产生多项式
      option_degree = st.selectbox(
          '请选择阶数',
          (2,3))
      poly_features = PolynomialFeatures(degree=option_degree)
      X_poly = poly_features.fit_transform(X)
      feature_name=poly_features.get_feature_names_out()
      # 创建线性回归模型并拟合数据
      model = LinearRegression()
      model.fit(X_poly, y)
      coef=model.coef_
      dffe_name = pd.DataFrame(feature_name)
      dffe_name['系数']=coef
      st.write( dffe_name)
      y_pred = model.predict(X_poly)
      # 计算并打印均方误差 (MSE) 以评估模型的性能
      mse = mean_squared_error(y, y_pred)
      st.write('Mean Squared Error:', mse)
      st.write("多项式回归的R^2值为:", r2_score(y, y_pred))
    with col02:
     try :
      input_sample = st.text_input('输入预测数据1', '用英文逗号隔开,注意位数一样')
      if input_sample!='用英文逗号隔开,注意位数一样':
        input_sample = input_sample.split(",")
        input_sample = [float(x) for x in input_sample]
        input_sample = np.array([input_sample])
        input_sample = poly_features.transform(input_sample)
        y_pred = model.predict(input_sample)
        st.write("预测结果:", y_pred)
     except Exception as e:
         st.write(e)
  with tab8:
    col1, col2 = st.columns(2)
    with col1:
      X8=data_x
      Y8=data_y
      st.write("决策树参数设定")
      test_size = st.slider('测试样本比例', 0.0, 1.0, 0.3)
      max_depth = st.slider('决策树深度', 1, 20, None)
      random_state= st.slider('随机数种子', 1, 500, 42)
      X_train, X_test, y_train, y_test = train_test_split(X8, Y8, test_size=test_size, random_state=random_state)
      # 创建决策树回归模型  
      regressor = DecisionTreeRegressor(max_depth=max_depth,random_state=random_state)
      # 训练模型  
      regressor.fit(X_train, y_train)
      # 进行预测  
      y_pred = regressor.predict(X_test)
      # 计算均方误差  
      mse = mean_squared_error(y_test, y_pred)
      #重要参数
      DecisionTreeRe_imp= pd.DataFrame({'feature': X8.columns, "重要程度": regressor.feature_importances_})
      st.write("回归树的MSE值为:", mse)
      st.write("回归树的R^2值为:", r2_score(y_test, y_pred))
      st.write(DecisionTreeRe_imp)
      PartialDependenceDisplay.from_estimator(regressor, X8, options_x)
      st.pyplot()
    with col2:
      input_sample = st.text_input('输入预测数据', '用英文逗号隔开,注意位数一样')
      st.write(input_sample)
      if input_sample!='用英文逗号隔开,注意位数一样':
        input_sample = input_sample.split(",")
        input_sample = [float(x) for x in input_sample]
        input_sample = np.array([input_sample])
        y_pred = regressor.predict(input_sample)
        st.write("预测结果:", y_pred)
  with tab9:
    col3, col4 = st.columns(2)
    with col3:
      X9 = data_x
      Y9 = data_y
      test_sizes = st.slider('随机森林测试样本比例', 0.0, 1.0, 0.3)
      n_estimators9= st.slider('决策树数量', 1, 200, 100)
      min_samples_leaf=st.slider('叶节点处需要的最小样本数', 1, 20, 1)
      max_depth = st.slider('随机森林树的深度', 1, 20, None)
      random_states = st.slider('随机森林随机数种子', 1, 500, 42)
      X_train, X_test, y_train, y_test = train_test_split(X9, Y9, test_size=test_sizes, random_state=random_states)
      # 创建随机森林回归模型
      regressor = RandomForestRegressor(n_estimators=n_estimators9, min_samples_leaf=min_samples_leaf,max_samples=max_depth,random_state=random_states)
      # 训练模型
      regressor.fit(X_train, y_train)
      # 进行预测
      y_pred = regressor.predict(X_test)
      # 计算均方误差（MSE）
      mse = mean_squared_error(y_test, y_pred)
      st.write("Mean Squared Error:", mse)
      st.write("随机森林的R^2值为:", r2_score(y_test, y_pred))
      RandomForestRe_imp = pd.DataFrame({'feature': X9.columns, "重要程度": regressor.feature_importances_})
      st.write(RandomForestRe_imp)


      PartialDependenceDisplay.from_estimator(regressor,X9 ,options_x)
      st.pyplot()
    with col4:
        input_sample = st.text_input('输入随机森林预测数据', '用英文逗号隔开,注意位数一样')
        st.write(input_sample)
        if input_sample != '用英文逗号隔开,注意位数一样':
            input_sample = input_sample.split(",")
            input_sample = [float(x) for x in input_sample]
            input_sample = np.array([input_sample])
            y_pred = regressor.predict(input_sample)
            st.write("预测结果:", y_pred)
  with tab10:
    col5,col6 = st.columns(2)
    with col5:
      X10 = data_x
      Y11= data_y
      test_sizesvr = st.slider('支持向量机测试样本比例', 0.0, 1.0, 0.3)
      kernel = st.selectbox(
        '核函数',
          ("linear","poly","rbf","sigmoid"))
      C = st.slider('正则化参数', 0.0, 10.0, 1.0)
      random_statesvr = st.slider('支持向量机随机数种子', 1, 500, 42)
      epsilon=st.slider('训练损失函数范围', 0.0, 0.3, 0.1)
      X_train, X_test, y_train, y_test = train_test_split(X10, Y11, test_size= test_sizesvr, random_state=random_statesvr)
      # 创建支持向量回归模型
      svr = SVR(kernel=kernel, C=C , epsilon=epsilon)
      # 训练模型
      svr.fit(X_train, y_train)
      # 进行预测
      y_pred = svr.predict(X_test)
      # 计算均方误差（MSE）
      mse = mean_squared_error(y_test, y_pred)
      st.write("Mean Squared Error:", mse)
      st.write("支持向量的R^2值为:", r2_score(y_test, y_pred))
      PartialDependenceDisplay.from_estimator(svr, X10, options_x)
      st.pyplot()
    with col6:
        input_sample = st.text_input('输入支持向量机预测数据', '用英文逗号隔开,注意位数一样')
        st.write(input_sample)
        if input_sample != '用英文逗号隔开,注意位数一样':
            input_sample = input_sample.split(",")
            input_sample = [float(x) for x in input_sample]
            input_sample = np.array([input_sample])
            y_pred = svr.predict(input_sample)
            st.write("预测结果:", y_pred)
  with tab11:
    col7, col8 = st.columns(2)
    with col7:
      X11=data_x
      Y11=data_y
      test_sizemlp = st.slider('神经网络训练样本比例', 0.0, 1.0, 0.3)
      random_statemlp = st.slider('神经网络随机数种子', 1, 500, 42)
      hidden_layer_sizes=st.slider('神经元数量', 1, 200, 100)
      activation= st.selectbox(
        '隐藏层激活函数',
          ("relu","tanh","logistic","identity"))
      solver = st.selectbox(
          '梯度下降',
          ("adam", "sgd", "lbfgs"))
      max_iternlp = st.slider('神经网络最大迭代数量', 1, 500, 200)
      X_train, X_test, y_train, y_test = train_test_split(X11, Y11, test_size=test_sizemlp, random_state=random_statemlp)
      # 创建MLP回归模型
      mlp = MLPRegressor( solver= solver,activation=activation,hidden_layer_sizes=(hidden_layer_sizes,), max_iter=max_iternlp, random_state=random_statemlp)
      # 训练模型
      mlp.fit(X_train, y_train)
      # 进行预测
      y_pred = mlp.predict(X_test)
      # 计算均方误差（MSE）
      mse = mean_squared_error(y_test, y_pred)
      st.write("Mean Squared Error:", mse)
      st.write("神经网络的R^2值为:", r2_score(y_test, y_pred))
      PartialDependenceDisplay.from_estimator(mlp, X11, options_x)
      st.pyplot()
    with col8:
        input_sample = st.text_input('输入神经网络预测数据', '用英文逗号隔开,注意位数一样')
        st.write(input_sample)
        if input_sample != '用英文逗号隔开,注意位数一样':
            input_sample = input_sample.split(",")
            input_sample = [float(x) for x in input_sample]
            input_sample = np.array([input_sample])
            y_pred = mlp.predict(input_sample)
            st.write("预测结果:", y_pred)
  with tab12:
    col9, col10 = st.columns(2)
    with col9:
      X12=data_x
      Y12=data_y
      test_sizexb = st.slider('XBoost训练样本比例', 0.0, 1.0, 0.3)
      random_statexb = st.slider('XBoost随机数种子', 1, 500, 42)
      max_depthxb = st.slider('XBoost树的最大深度', 1, 50, 6)
      eta = st.slider('学习率', 0.001, 2.0, 0.1)
      booster = st.selectbox(
          '基础学习器',
          ("gbtree", "gblinear"))
      X_train, X_test, y_train, y_test = train_test_split(X12,Y12, test_size=test_sizexb, random_state=random_statexb)
      # 定义模型参数
      params = {
          'objective': 'reg:squarederror',  # 回归任务，使用均方误差作为损失函数
          'booster': booster,  # 使用 GBM 树作为基础学习器
          'eval_metric': 'rmse',  # 使用均方根误差作为评估指标
          'max_depth': max_depthxb ,  # 树的最大深度
          'eta': eta,  # 学习率
          'seed': random_statexb  # 随机种子，确保结果可复现
      }
      # 训练模型
      model = xgb.XGBRegressor(**params)
      model.fit(X_train, y_train)
      Xboost_imp = pd.DataFrame({'feature': X12.columns, "重要程度": model.feature_importances_})
      st.write(Xboost_imp)
      # 进行预测
      y_pred = model.predict(X_test)
      # 计算均方根误差（RMSE）以评估模型性能
      rmse = np.sqrt(mean_squared_error(y_test, y_pred))
      st.write("RMSE: %.4f" % rmse)
      st.write("XGBoost的R^2值为:", r2_score(y_test, y_pred))
      PartialDependenceDisplay.from_estimator(model, X12, options_x)
      st.pyplot()
    with col10:
        input_sample = st.text_input('输入XGBoost预测数据', '用英文逗号隔开,注意位数一样')
        st.write(input_sample)
        if input_sample != '用英文逗号隔开,注意位数一样':
            input_sample = input_sample.split(",")
            input_sample = [float(x) for x in input_sample]
            input_sample = np.array([input_sample])
            y_pred = model.predict(input_sample)
            st.write("预测结果:", y_pred)
  with tab13:
      X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
      # 创建GBTD回归模型
      gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
      # 训练模型
      gb_reg.fit(X_train, y_train)
      # 在测试集上进行预测
      y_pred = gb_reg.predict(X_test)
      # 输出预测结果和实际值的比较
      st.write("预测值:", y_pred)
      st.write("实际值:", y_test)

      #pip install -U matplotlib==3.6 -i https://pypi.tuna.tsinghua.edu.cn/simple
      #open-interpreter






