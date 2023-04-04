from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz,plot_tree
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import pandas as pd
import seaborn as sns
from matplotlib import rc
import streamlit as st
import graphviz
##################################################################################################################################################################
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")
##################################################################################################################################################################

st.markdown(""" <style> .font_title {
font-size:50px ; font-family: 'times';text-align: center;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_header2 {
font-size:30px ; font-family: 'times';text-align: center;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_header {
font-size:50px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subheader {
font-size:35px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subsubheader {
font-size:28px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_text {
font-size:26px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subtext {
font-size:18px ; font-family: 'times';text-align: center;} 
</style> """, unsafe_allow_html=True)

font_css = """
<style>
button[data-baseweb="columns"] > div[data-testid="stMarkdownContainer"] > p {
  font-size: 24px; font-family: 'times';
}
</style>
"""
# def sfmono():
    # font = "Times"
    
    # return {
        # "config" : {
             # "title": {'font': font},
             # "axis": {
                  # "labelFont": font,
                  # "titleFont": font
             # }
        # }
    # }

# alt.themes.register('sfmono', sfmono)
# alt.themes.enable('sfmono')
####################################################################################################################################################################
st.markdown('<p class="font_title">Decision Tree Vs. Random Forest</p>', unsafe_allow_html=True)
cols = st.columns(2,gap='small')
cols[0].markdown('<p class="font_header2">Decision Tree </p>', unsafe_allow_html=True)
cols[0].image("https://opendatascience.com/wp-content/uploads/2017/06/dt-diagram-711x350.jpg")
cols[1].markdown('<p class="font_header2">Random Forest </p>', unsafe_allow_html=True)
cols[1].image("https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/rfc_vs_dt1.png")
####################################################################################################################################################################
st.markdown('<p class="font_text"> For this ICA, we are trying to see how decision tree is compared with random forest with hyperparameter tuning. There are 4 datasets which you can utilize for comparison.</p>', unsafe_allow_html=True)
####################################################################################################################################################################
cols_new = st.columns(2,gap='small')
Dataset = st.sidebar.selectbox('Select Dataset',('Wine', 'Digits', 'Breast Cancer','Iris'),index = 0)
if Dataset == 'Wine':
    wine = datasets.load_wine()
    X = wine.data
    y = wine.target
elif Dataset == 'Digits':
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
elif Dataset == 'Breast Cancer':
    breast_cancer = datasets.load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
else:
    iris = datasets.load_Iris()
    X = iris.data
    y = iris.target

Criterion = st.sidebar.selectbox('Select criterion for both classifier',('gini', 'entropy'),index = 0)
Max_Depth = st.sidebar.number_input('Select the value for max depth of Tree for both classifier'             , min_value=1, max_value=10000,step=1, value=2, format='%i')
Min_Sample_Split = int(st.sidebar.number_input('Select the value for minimum sample splitter for both classifier', min_value=1, max_value=100  ,step=1, value=2, format='%i'))
Min_Sample_Leaf  = int(st.sidebar.number_input('Select the value for minimum sample in leaf for both classifier' , min_value=1, max_value=100  ,step=1, value=1, format='%i'))
Max_Features = st.sidebar.selectbox('Select method for splitting features for both classifier',('sqrt', 'log2','auto'),index = 0)
Complexity  = st.sidebar.number_input('Select the value for complexity paramter for both classifier' , min_value=0.0, max_value=100.0  ,step=0.1, value=0.0, format='%f')
Random_State = st.sidebar.slider('Select a value for the random state for both classifier', 0, 100, value=50, format = '%i')
Splitter_DT = st.sidebar.selectbox('Select splitter for Decision Tree',('best','random'),index = 0)

DT=DecisionTreeClassifier(criterion=Criterion, splitter=Splitter_DT , max_depth=Max_Depth, min_samples_split=Min_Sample_Split,
    min_samples_leaf=Min_Sample_Leaf, min_weight_fraction_leaf=0.0, max_features=Max_Features,
    random_state=Random_State, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=Complexity)

Tree_Number  = st.sidebar.number_input('Select the number of tree for Random Forest',        min_value=1, max_value=100  ,step=1, value=2, format='%i')
 
RF= RandomForestClassifier(n_estimators=Tree_Number, criterion=Criterion, max_depth=Max_Depth, min_samples_split=Min_Sample_Split,
    min_samples_leaf=Min_Sample_Leaf, min_weight_fraction_leaf=0.0, max_features=Max_Features, max_leaf_nodes=None,
    min_impurity_decrease=0.0, bootstrap=True, oob_score=False, random_state=Random_State, verbose=0, warm_start=False,
    class_weight=None, ccp_alpha=0.0, max_samples=None)
DT.fit(X,y)

if Dataset == 'Wine':
    dot_data = export_graphviz(DT, out_file=None, feature_names=wine.feature_names,class_names=wine.target_names, filled=True)
elif Dataset == 'Digits':
    dot_data = export_graphviz(DT, out_file=None, feature_names=digits.feature_names,class_names=digits.target_names, filled=True)
elif Dataset == 'Breast Cancer':
    dot_data = export_graphviz(DT, out_file=None, feature_names=breast_cancer.feature_names,class_names=breast_cancer.target_names, filled=True)
else:
    dot_data = export_graphviz(DT, out_file=None, feature_names=iris.feature_names,class_names=iris.target_names, filled=True)
cols_new[0].graphviz_chart(dot_data)
# plot_tree(DT, feature_names=wine.feature_names,  class_names=wine.target_names, filled=True)
# Fig=plt.figure(figsize=(50,40))
# plot_tree(DT)
# cols_new[0].pyplot(Fig)

RF.fit(X,y)

for i in range(0,Tree_Number):
    # Fig=plt.figure(figsize=(50,40))
    cols_new[1].write(str(i+1)+"th Tree in the forest")
    # plot_tree(RF.estimators_[i])
    # cols_new[1].pyplot(Fig)
    if Dataset == 'Wine':
        dot_data = export_graphviz(RF.estimators_[i], out_file=None, feature_names=wine.feature_names,class_names=wine.target_names, filled=True)
    elif Dataset == 'Digits':
        dot_data = export_graphviz(RF.estimators_[i], out_file=None, feature_names=digits.feature_names,class_names=digits.target_names, filled=True)
    elif Dataset == 'Breast Cancer':
        dot_data = export_graphviz(RF.estimators_[i], out_file=None, feature_names=breast_cancer.feature_names,class_names=breast_cancer.target_names, filled=True)
    else:
        dot_data = export_graphviz(RF.estimators_[i], out_file=None, feature_names=iris.feature_names,class_names=iris.target_names, filled=True)
    cols_new[1].graphviz_chart(dot_data)

# DOT data
# Fig=plt.figure(figsize=(50,40))
# plot_tree(RF.estimators_[Tree_Visualization-1])
# plot_tree(DT, feature_names=wine.feature_names,  class_names=wine.target_names, filled=True)
# cols_new[1].pyplot(Fig)
####################################################################################################################################################################
st.markdown('<p class="font_header">References: </p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">1) Buitinck, Lars, Gilles Louppe, Mathieu Blondel, Fabian Pedregosa, Andreas Mueller, Olivier Grisel, Vlad Niculae et al. "API design for machine learning software: experiences from the scikit-learn project." arXiv preprint arXiv:1309.0238 (2013). </p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">2) https://opendatascience.com/decision-trees-tutorial/</p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">3) https://www.analyticsvidhya.com/blog/2021/05/bagging-25-questions-to-test-your-skills-on-random-forest-algorithm/</p>', unsafe_allow_html=True)
##################################################################################################################################################################