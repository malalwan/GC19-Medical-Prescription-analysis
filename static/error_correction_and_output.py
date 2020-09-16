import sys
import cv2
import numpy as np
import json
from pprint import pprint
import os
import re
from collections import Counter
from nltk.stem import *


timestamp = sys.argv[-2]
file_ext = sys.argv[-1]

fullpath=os.path.dirname(os.path.realpath(__file__))


def words_lang(text): return re.findall(r'\w+', text.lower())

WORDS_lang = Counter(words_lang(open(fullpath+'/big.txt').read()))

def P_lang(word, N=sum(WORDS_lang.values())): 
    "Probability of `word`."
    return WORDS_lang[word] / N

def lang_correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P_lang)

def candidates(word):
    "Generate possible spelling corrections for word."
    return (known_lang([word]) or known_lang(edit1(word)) or known_lang(edit2(word)) or [word])

def known_lang(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)


def edit1(word):
    letters  = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in xrange(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edit2(word):
    return (e2 for e1 in edit1(word) for e2 in edit1(e1))

# returns all the set of matching values in the dict 
def known(words): 
    possible,matching=[],[]
    for word in words:
        if word in med_dict:
            possible.extend(med_dict[word])
            a=[word for i in xrange(len(med_dict[word]))]
            matching.extend(a) 
    return possible,matching

def predict1(word):
    if word in WORDS: return [word],[word]
    return known(edit1(word.lower()))

def predict2(word):
    if word in WORDS: return [word],[word]
    return known(edit2(word.lower()))

def known_sym(symptoms):
    possible=[]
    for symptom in symptoms:
        if symptom in SYMPTOMS: possible.append(symptom)
    return possible

def predict1_sym(symptom):
    if symptom in SYMPTOMS: return [symptom]
    return known_sym(edit1(symptom))

def predict2_sym(symptom):
    if symptom in SYMPTOMS: return [symptom]
    return known_sym(edit2(symptom))
# predict medicine from a list of medicines by symptom matching as obtd. after 1 nd 2 edits
def predict_medicine(medicine,symptoms):
    pred1,match1=predict1(medicine)
    pred2,match2=predict2(medicine)
    pred_set=match_set=[]
    n1,n2=len(pred1),len(pred2)
    if n1==0 and n2==0:return None
    elif n1==0:
        match_set=match2
        pred_set=pred2
    elif n2==0:
        match_set=match1
        pred_set=pred1
    else:
        match_set=match1
        match_set.extend(match2)
        pred_set=pred1
        pred_set.extend(pred2)
    max_score=0
    med_index=0
    for i in xrange(len(pred_set)):
        med=pred_set[i]
        score=0
        symptom_set=set()  
        if med in meds:
            for symptom in meds[med]:
                symptom_set.update(map(lambda x: x.lower(),symptom.split()))
        for symptom in symptom_set:
            if symptom in symptoms: score+=1
        if max_score<score:
            max_score=score
            med_index=i
    return match_set[med_index]

def P(word): 
    "Probability of `word`."
    if word in count:
        return count[word] / float(total_count)
    else:
        return 0

# predict symptoms as obtd. after 1 nd 2 edits
def predict_symptom(symptom):
    pred1=predict1_sym(symptom)
    pred2=predict2_sym(symptom)
    n1,n2=len(pred1),len(pred2)
    if n1==0 and n2==0:return max([symptom],key=P)
    elif n1==0:return max(pred2,key=P)
    else :return max(pred1,key=P)

def add_token(inp,inp1,flag,tokens,symptoms):
    res=0
    if flag==0:
        if inp in stopwords:
            tokens.append(inp)
        elif inp in SYMPTOMS:
            symptoms.add(inp)
            tokens.append(inp)
            res=1
        else: res=-1
    else:
        if inp in stopwords:
            tokens.append(inp1)
        elif inp in SYMPTOMS:
            symptoms.add(inp)
            tokens.append(inp1)
            res=1
        else: res=-1
    return tokens,symptoms,res

def stemming(inp_symptoms):
    stemmer=PorterStemmer()
    symptoms=set()
    tokens=[]
    inp_symptoms.split()
    for inp in inp_symptoms.split():
        inp=inp.lower()
        inp_pred=predict_symptom(inp)
        tokens,symptoms,res=add_token(inp,inp,0,tokens,symptoms)
        if res==-1:tokens,symptoms,res=add_token(inp_pred,inp_pred,0,tokens,symptoms)
        if res==-1:
            inp_stem=stemmer.stem(inp)
            inp_pred_stem=stemmer.stem(inp_pred)     
            stem1=predict_symptom(inp_stem)
            stem2=predict_symptom(inp_pred_stem)
           
            tokens,symptoms,res=add_token(inp_stem,inp,1,tokens,symptoms)
            if res==-1: tokens,symptoms,res=add_token(inp_pred_stem,inp_pred,1,tokens,symptoms)
            if res==-1: tokens,symptoms,res=add_token(stem1,inp,1,tokens,symptoms)
            if res==-1: tokens,symptoms,res=add_token(stem2,inp_pred,1,tokens,symptoms)
            if res==-1: tokens.append(inp)
    return symptoms,tokens

def getSymptoms(inp_symptoms):
    symptoms,tokens=stemming(inp_symptoms)
    return symptoms,tokens




# Reading the stop words
f=open(fullpath+"/stopwords.txt","r")
stopwords=set(f.read().split(','))
f.close()

# Reading the med dict
f=open(fullpath+"/med-list.txt","r")
WORDS=set(f.read().split('|'))
med_dict={}
for med in WORDS:
    for part in med.split():
        if part in med_dict:med_dict[part].append(med)
        else:med_dict[part]=[med]
f.close()

# Reading the med dict with symptoms
f=open(fullpath+"/main_dict.json","r")
meds=json.loads(f.read())
f.close()

# get all the symptoms
SYMPTOMS=set()
count={}
total_count=0
for med in meds:
    for symptom in meds[med]:
        SYMPTOMS.update(map(lambda x: x.lower(),symptom.split()))
        for x in symptom.split():
            if x in count:
                count[x.lower()]+=1
            else:
                count[x.lower()]=1
            total_count +=1



img1=cv2.imread(fullpath+'/uploads/'+timestamp+'/preprocessed-resized.'+file_ext)
img2= cv2.imread(fullpath+'/img/blank.png')
img2=cv2.resize(img2,(589,821))
for i in range(0,140):
	for j in range(0,589):
		img2[i][j]=img1[i][j]

img3= cv2.imread(fullpath+'/img/blank.png')
img3=cv2.resize(img3,(589,821))
for i in range(0,140):
	for j in range(0,589):
		img3[i][j]=img1[i][j]

# cv2.imshow("intermediate",img3)
# cv2.waitKey(0)
cv2.rectangle(img2,(9,6),(629*589//640,517*821//520),(0,0,0),2)
cv2.line(img2,(11,130*821//520),(628*589//640,130*821//520),(0,0,0),2)
cv2.line(img2,(11,319*821//520),(628*589//640,319*821//520),(0,0,0),2)

cv2.rectangle(img3,(9,6),(629*589//640,517*821//520),(0,0,0),2)
cv2.line(img3,(11,130*821//520),(628*589//640,130*821//520),(0,0,0),2)
cv2.line(img3,(11,319*821//520),(628*589//640,319*821//520),(0,0,0),2)
#cv2.imshow('output', img2)
# x123 - blank
# cv2.imwrite(fullpath+'/uploads/'+timestamp+'/ROIsegmented.'+file_ext,img2)
data = json.load(open(fullpath+'/uploads/'+timestamp+'/data.txt'))
lines = data['recognitionResult']['lines']
# pprint (lines)


for i in range(len(lines)):
	coordinates=lines[i]['boundingBox']
	
	if coordinates[7]>150:

		font= cv2.FONT_HERSHEY_SIMPLEX
		fontScale= 0.7
		fontColor= (0,0,0)
		lineType= 2
		
		words=lines[i]['words']
		#print words
		flag=-1
		text1= ''
		for l in range (len(words)):
			text=words[l]['text']
			
			bottomLeftCornerOfText = (words[l]['boundingBox'][6],(words[l]['boundingBox'][7]+words[l]['boundingBox'][1]+words[l]['boundingBox'][3]+words[l]['boundingBox'][5])/4)
			
			#print words[l]['boundingBox'][6],words[l]['boundingBox'][7]
			if text=='Name':
				flag=0
				continue
			elif text=='Age':
				flag=1
				text1=''
				continue
			elif text=='Gender':
				flag=2
				text1=''
				continue
			elif text=='Date':
				flag=3
				text1=''
				continue
			elif text==':':
				continue


			if flag==0 and 	words[l]['boundingBox'][7]<240:
				text1+=text
				cv2.putText(img2,text1, (90,180), font, fontScale,fontColor, lineType)

			if flag==1 and 	words[l]['boundingBox'][7]<240:
				text1+=text
				cv2.putText(img2,text1, (235,180), font, fontScale,fontColor, lineType)
			if flag==2 and 	words[l]['boundingBox'][7]<240:
				text1+=text
				cv2.putText(img2,text1, (355,180), font, fontScale,fontColor, lineType)
			if flag==3 and 	words[l]['boundingBox'][7]<240:
				text1+=text
				cv2.putText(img2,text1, (465,180), font, fontScale,fontColor, lineType)

			elif text!='Name' and text!='Age' and text!='Gender' and text!='Date' and text!=':' and (words[l]['boundingBox'][7]>270) and (words[l]['boundingBox'][7]<520 or words[l]['boundingBox'][7]>570) and (words[l]['boundingBox'][6]>0):
				cv2.putText(img2,text, bottomLeftCornerOfText, font, fontScale,fontColor, lineType)




#cv2.imshow("final",img2)
# x123 - Printed output
cv2.imwrite(fullpath+'/uploads/'+timestamp+'/before_correction.'+file_ext,img2)

extracted_symptom=[]

for i in range(len(lines)):
	coordinates=lines[i]['boundingBox']
	
	if coordinates[7]>150:

		font= cv2.FONT_HERSHEY_SIMPLEX
		fontScale= 0.7
		fontColor= (0,0,0)
		lineType= 2
		
		words=lines[i]['words']
		#print words
		flag=-1
		text1= ''
		for l in range (len(words)):
			text=words[l]['text']
			
			bottomLeftCornerOfText = (words[l]['boundingBox'][6],(words[l]['boundingBox'][7]+words[l]['boundingBox'][1]+words[l]['boundingBox'][3]+words[l]['boundingBox'][5])/4)
			
			#print words[l]['boundingBox'][6],words[l]['boundingBox'][7]
			if text=='Name':
				flag=0
				continue
			elif text=='Age':
				flag=1
				text1=''
				continue
			elif text=='Gender':
				flag=2
				text1=''
				continue
			elif text=='Date':
				flag=3
				text1=''
				continue
			elif text==':':
				continue


			if flag==0 and 	words[l]['boundingBox'][7]<240:
				text1+=text
				cv2.putText(img3,text1, (90,180), font, fontScale,fontColor, lineType)

			if flag==1 and 	words[l]['boundingBox'][7]<240:
				text1+=text
				cv2.putText(img3,text1, (235,180), font, fontScale,fontColor, lineType)
			if flag==2 and 	words[l]['boundingBox'][7]<240:
				text1+=text
				cv2.putText(img3,text1, (355,180), font, fontScale,fontColor, lineType)
			if flag==3 and 	words[l]['boundingBox'][7]<240:
				text1+=text
				cv2.putText(img3,text1, (465,180), font, fontScale,fontColor, lineType)

			elif text!='Name' and text!='Age' and text!='Gender' and text!='Date' and text!=':' and (words[l]['boundingBox'][7]>270) and (words[l]['boundingBox'][7]<520 or words[l]['boundingBox'][7]>570) and (words[l]['boundingBox'][6]>0):
				corrected=""
				if(words[l]['boundingBox'][7]<520):
					sym, tok = getSymptoms(text)
					if len(sym)==0:
							text = lang_correction(text)
					else:
						corrected = tok[0]
					extracted_symptom.extend(sym)
				elif(words[l]['boundingBox'][7]>570):
					corrected = predict_medicine(text, extracted_symptom)
				if corrected!="" and corrected is not None:
					corrected=corrected.title()
					text=corrected

				cv2.putText(img3,text, bottomLeftCornerOfText, font, fontScale,fontColor, lineType)


cv2.imwrite(fullpath+'/uploads/'+timestamp+'/after_correction.'+file_ext,img3)

