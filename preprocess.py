#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 20:52:27 2017

@author: dakotashen
"""

import csv
from data_pre import PreProcessor

p = PreProcessor()

# read in cvs
csvFile = open("Sentiment Analysis Dataset .csv", "r")
positive = csv.reader(csvFile)

# build empty dict
result = []
col0 =[]
col1 = []
col2 = []

for item in positive:
    # ignore first line
    if positive.line_num == 1:
        continue
    sentence = item[3]
    sentence_list = sentence.split(" ")
    after = p.process(sentence_list)
    after_list = " ".join(after)
    result.append(after_list)
    col0.append(item[0])
    col1.append(item[1])
    col2.append(item[2])
    
    #item[3] = after_list
    
csvFile.close()
   

# fileheader
fileHeader = ["sentence_id", "Sentiment", "Source", "Text"]

# write data

csvFile1 = open("final_data.csv", "w")
writer = csv.writer(csvFile1)

writer.writerow(fileHeader)
for i in range(len(result)):
    writer.writerow([col0[i], col1[i], col2[i],result[i]])

csvFile1.close()
