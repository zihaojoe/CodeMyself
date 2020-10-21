# check the encoding methods
import chardet 
rawdata = open('./Question_Answer_Dataset_v1.2/S08/question_answer_pairs.txt', 'rb').read()
result = chardet.detect(rawdata)
charenc = result['encoding']
print(charenc)