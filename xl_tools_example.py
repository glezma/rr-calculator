# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 23:03:56 2016

@author: gonza
"""
import openpyxl as opxl
import xl_tools as xl
filename = 'In.CE.xlsm'
# filename = "F:\OneDrive\python_projects\cir_economic_capital\In.CE.xlsm"
wb = opxl.load_workbook(filename,data_only=True,use_iterators = False)
a=xl.xl_load(wb, 'gap_mn')
# print(a)