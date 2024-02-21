# coding=utf-8
import streamlit as st
from st_pages import Page, Section, show_pages, add_page_title
add_page_title()
#streamlit run main.py
#streamlit run streamlitdemo.py
# Specify what pages should be shown in the sidebar, and what their titles and icons
# should be
show_pages(
    [
        Page("SPC_gr.py", "真实追溯"),
        Page("outlier_detection.py", "异常检测"),
        Page("regression_analysis_gr.py", "回归分析"),
        Page("Fibergr.py", "纤维异物匹配")
    ]
)
