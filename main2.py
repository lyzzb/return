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

        Page("SPC2.py", "SPC1"),
        Page("SPC.py", "图追溯")
    ]
)
