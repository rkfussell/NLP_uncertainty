# -*- coding: utf-8 -*-
"""
One big issue with labnotes is segmentation. Can NER help us address this?


Created on Mon Jan 30 14:13:38 2023
See if NER from Spacy can extract useful information from labnotes
@author: rkf33

https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
"""

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_md

import pandas as pd

df = pd.read_excel("All_paragraphs_sample2.xlsx")

all_para = ' '.join(map(str, df[0].tolist()))

nlp = en_core_web_md.load()

notes = nlp(all_para)

print(len(notes.ents))

labels = [x.label_ for x in notes.ents]
print(Counter(labels))
items = [(x.text, x.label_) for x in notes.ents]
#print(Counter(items).most_common(20))

processed = [(x.lemma_, x.pos_) for x in [y 
                                        for y
                                        in notes 
                                        if not y.is_stop and y.pos_ != 'PUNCT']]


items = [(x.text, x.label_) for x in notes.ents if x.label_ not in ["CARDINAL", "DATE"]]
NER_dict = dict(items)
print(Counter(items))



df2 = pd.read_excel("Video_triangulation_mechanism_testing_all_paragraphs.xlsx")

all_para = ' '.join(map(str, df2[0].tolist()))

nlp = en_core_web_md.load()

notes = nlp(all_para)

print(len(notes.ents))
items = [(x.text, x.label_) for x in notes.ents if x.label_ not in ["CARDINAL", "DATE"]]
NER_dict = dict(items)
print(Counter(items))

#could pull out sentences surrounding all entities, focus on ones that are most important to the experiment (10 degrees and 20 degrees, Hooke), categorize them. Go through spreadsheet for error rate. 
print([(x.text, x.ent_iob_) for x in notes if  x.ent_iob_ != "O"])

#starting with b of interest, going back to previous period and next period. extract that sentence

notes = nlp("This experiment rules out Model 1 because the data does not match g. We will now test model two by examining the peak of the trajectory of the ball. ")

print(len(notes.ents))
items = [(x.text, x.label_) for x in notes.ents ]
NER_dict = dict(items)
print(Counter(items))

#could pull out sentences surrounding all entities, focus on ones that are most important to the experiment (10 degrees and 20 degrees, Hooke), categorize them. Go through spreadsheet for error rate. 
print([(x.text, x.ent_iob_) for x in notes if  x.ent_iob_ != "O"])