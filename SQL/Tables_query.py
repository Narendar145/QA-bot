import torch
from transformers import pipeline
import pandas as pd

# print(torch.__version__)

# pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0.html

tqa = pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq")   # <-- works except for avg, age


table = pd.read_csv("data1/sys.csv")
# table = table.replace('_', ' ', regex=True)
table = table.astype(str)
print(table)

query = "show me top 10 SYSTEM NAME"
print("Answer  ->" + tqa(table=table, query = query)['answer'])



# Random notes below. Please ignore
#################################
# below date based questions dont work
#query = "who have birth dates in month of 06?" # June does not work
#print("June folks ->" + tqa(table=table, query = query)['answer'])

#query = "who have birth dates in month of September?"
#print("Sept folks ->" + tqa(table=table, query = query)['answer'])


# try this - https://www.youtube.com/watch?v=qgaM0weJHpA  <- trains based upon the data

'''
The model 'BertForMaskedLM' is not supported for table-question-answering. Supported models are
 ['TapasForQuestionAnswering', 'BartForConditionalGeneration', 'BigBirdPegasusForConditionalGeneration', 
 'BlenderbotForConditionalGeneration', 'BlenderbotSmallForConditionalGeneration', 
 'EncoderDecoderModel', 'FSMTForConditionalGeneration', 'GPTSanJapaneseForConditionalGeneration', 
 'LEDForConditionalGeneration', 'LongT5ForConditionalGeneration', 'M2M100ForConditionalGeneration', 
 'MarianMTModel', 'MBartForConditionalGeneration', 'MT5ForConditionalGeneration', 
 'MvpForConditionalGeneration', 'NllbMoeForConditionalGeneration', 
 'PegasusForConditionalGeneration', 'PegasusXForConditionalGeneration', 
 'PLBartForConditionalGeneration', 'ProphetNetForConditionalGeneration', 
 'SwitchTransformersForConditionalGeneration',
 'T5ForConditionalGeneration', 'XLMProphetNetForConditionalGeneration'].
'''

# No module named 'keras.engine'
# run in venv
#   python -m venv  .venv_csv (done alraedy)
#   source .venv_csv/bin/activate
#       - deactivate
# pip uninstall tensorflow, torch, keras
#  pip install torch, keras