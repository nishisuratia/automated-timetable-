#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('Resultdata.csv')
df


# In[ ]:





# In[3]:


df = pd.DataFrame(df,  columns = ['syear','student_id','StudentName','standard_id','standard_title','subject_id','subject_title','exam_id','exam_title','obtain_marks','total_marks'])
df


# In[4]:


df['percent'] = (df['obtain_marks'] /
                      df['total_marks'])  * 100
df


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


import requests


# In[ ]:





# In[8]:


df['syear'].value_counts()


# In[9]:


df['StudentName'].value_counts()


# In[10]:


df['standard_id'].value_counts()


# In[11]:


df['subject_title'].value_counts()


# In[12]:


df['exam_title'].value_counts()


# In[13]:


df.nunique() #number of unique values in each column


# In[14]:


sns.set(rc={"figure.figsize":(30, 5)})


# In[15]:


ax = sns.scatterplot(x="subject_title", y="obtain_marks", data=df)


# In[16]:


ax = sns.scatterplot(x="subject_title", y="percent", data=df)


# In[17]:


grade=df.groupby("standard_title").aggregate({'obtain_marks':'mean'})
grade.reset_index(inplace=True)
grade


# In[ ]:





# In[18]:


grade1=df.groupby("subject_title").aggregate({'obtain_marks':'mean'})
grade1.reset_index(inplace=True)
grade1


# In[19]:


sns.barplot(data=grade,x='standard_title',y='obtain_marks');
plt.show(); #on an average marks are obtained highest in std1


# In[ ]:





# In[20]:


df = df.sort_values('syear')
print(df)


# In[21]:


df1=df[:16341] #2017
df1 #splitting data into resptive years 


# In[22]:


df2=df[16341:34449]#2018
df2


# In[23]:


df4=df[51738:59634] #2020
df4


# In[24]:


df5=df[59634:]#2021
df5


# In[25]:


df3=df[34449:51738] #2019
df3


# In[26]:


sns.barplot(data=df,x='syear',y='percent');
plt.show();


# In[27]:


sns.set(rc={"figure.figsize":(30, 5)})


# In[28]:


ax=sns.boxplot(x="subject_title", y="percent", data=df1);
ax.set_title(" subject vs percent in 2017")


# In[29]:


ax=sns.boxplot(x="subject_title", y="percent", data=df2);
ax.set_title(" subject vs percent in 2018")


# In[30]:


ax=sns.boxplot(x="subject_title", y="percent", data=df3);
ax.set_title(" subject vs percent in 2019")


# In[31]:


ax=sns.boxplot(x="subject_title", y="percent", data=df4);
ax.set_title(" subject vs percent in 2020")


# In[32]:


ax=sns.boxplot(x="subject_title", y="percent", data=df5);
ax.set_title(" subject vs percent in 2021")


# In[33]:


ax=sns.boxplot(x="exam_title", y="percent", data=df1);
ax.set_title(" type of exam vs percent in 2017")


# In[34]:


ax=sns.boxplot(x="exam_title", y="percent", data=df2);
ax.set_title(" type of exam vs percent in 2018")


# In[35]:


ax=sns.boxplot(x="exam_title", y="percent", data=df3);
ax.set_title(" type of exam vs percent in 2019")


# In[36]:


ax=sns.boxplot(x="exam_title", y="percent", data=df4);
ax.set_title(" type of exam vs percent in 2020")# more exams in 2020 and 2021


# In[37]:


ax=sns.boxplot(x="exam_title", y="percent", data=df5);
ax.set_title(" type of exam vs percent in 2021")


# In[38]:


ax=sns.stripplot(x="standard_title", y="percent", data=df1);
ax.set_title(" standard vs percent in 2017") #std 1-8 data is present in the year 2017 with student scoring in higher percentile


# In[39]:


ax=sns.stripplot(x="standard_title", y="percent", data=df2);
ax.set_title(" standard vs percent in 2018")


# In[40]:


ax=sns.stripplot(x="standard_title", y="percent", data=df3);
ax.set_title(" standard vs percent in 2017")


# In[41]:


ax=sns.stripplot(x="standard_title", y="percent", data=df4);
ax.set_title(" standard vs percent in 2020")


# In[42]:


ax=sns.stripplot(x="standard_title", y="percent", data=df5);
ax.set_title(" standard vs percent in 2021")


# In[43]:


ax=sns.stripplot(x="standard_title", y="percent", data=df);
ax.set_title(" standard vs percent overall for 5 years")


# In[44]:


dff = df.sort_values('subject_title')#sorting according to standard
print(dff)


# In[45]:


df['subject_title'].value_counts()


# In[46]:


dff.head()


# In[47]:


dff.tail()


# In[48]:


dff1=dff[:71] #subject -activity
dff1


# In[49]:


dff2=dff[71:215]#subject- computer
dff2


# In[50]:


dff3=dff[215:430]#subject- drawing
dff3


# In[51]:


dff4=dff[430:12788]#subject -english
dff4


# In[52]:


dff5=dff[12788:15133]#subject -environment
dff5


# In[53]:


dff6=dff[15133:26835]#subject gujarati
dff6


# In[54]:


dff7=dff[26835:36498]#subject -hindi
dff7


# In[55]:


dff8=dff[36498:38000]#subject -mari aas paas
dff8


# In[56]:


dff9=dff[38000:50281]#subject -mathematics
dff9


# In[57]:


dff10=dff[50281:50352]#subject -music
dff10


# In[58]:


dff11=dff[50352:57101]#subject -sanskrit
dff11


# In[59]:


dff12=dff[57101:64690]#subject -science and technology
dff12


# In[60]:


dff13=dff[64690:]#subject -social science
dff13


# In[61]:


#1 activity 
#2 computer
#3 drawing
#4 english
#5 environment 
#6 gujarati
#7 hindi 
#8 mari as pas
#9 maths
#10 music
#11 sanskrit
#12 science and technology
#13 social science


# In[62]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff1);# standard analysis
ax.set_title(" standard vs percent for activity")#only std1 and std2 have activity as a subject


# In[63]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff2);
ax.set_title(" standard vs percent for computer")#few students in computer for std-3-7


# In[64]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff3);
ax.set_title(" standard vs percent for drawing")#std 1-7 have drawing


# In[65]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff4);
ax.set_title(" standard vs percent for english")# all std have english


# In[66]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff5);
ax.set_title(" standard vs percent for environment")#std 2-5 have environment subject


# In[67]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff6);
ax.set_title(" standard vs percent for gujarati")# all std have gujarati


# In[68]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff7);
ax.set_title(" standard vs percent for hindi") #std 1-9 have hindi


# In[69]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff9);
ax.set_title(" standard vs percent for mathematics")#all std 1-10 have mathematics


# In[70]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff10);
ax.set_title(" standard vs percent for music")#few students in music and in std1-2 only


# In[71]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff11);
ax.set_title(" standard vs percent for sanskrit")#std 6-10 have sanskrit


# In[72]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff12);
ax.set_title(" standard vs percent for science and technology")#std 6-10


# In[73]:


ax=sns.stripplot(x="standard_title", y="percent", data=dff13);
ax.set_title(" standard vs percent for social science")#std 6-10


# In[74]:


ax=sns.stripplot(x="exam_title", y="percent", data=dff2);# exam analysis
ax.set_title(" exam vs percent for computer")# computer only has prelim exams


# In[75]:


ax=sns.stripplot(x="exam_title", y="percent", data=dff1);
ax.set_title(" exam vs percent for activity")#activity only have prelims


# In[76]:


ax=sns.stripplot(x="exam_title", y="percent", data=dff3);
ax.set_title(" exam vs percent for drawing")#drawing only have prelims


# In[77]:


ax=sns.stripplot(x="exam_title", y="percent", data=dff4);
ax.set_title(" exam vs percent for english")#english has all sorts of exams


# In[78]:


ax=sns.stripplot(x="exam_title", y="percent", data=dff5);
ax.set_title(" exam vs percent for environment")


# In[79]:


ax=sns.stripplot(x="exam_title", y="percent", data=dff6);
ax.set_title(" exam vs percent for gujarati")


# In[80]:


ax=sns.stripplot(x="exam_title", y="percent", data=dff9);
ax.set_title(" exam vs percent for mathematics")#below mentioned exams for math


# In[81]:


ax=sns.stripplot(x="subject_title", y="percent", data=dff);
ax.set_title(" subject vs percent")#


# In[82]:


frames=[dff9,dff12]


# In[83]:


result=pd.concat(frames)


# In[84]:


result


# In[85]:


value = result.sort_values('standard_title')#sorting according to standard
print(value)


# In[86]:


value['standard_title'].value_counts()


# In[87]:


one=value[326:3577]# dataset of science and maths with std 6-10
one


# In[88]:


two=value[7735:]
two


# In[89]:


frame=[one,two]


# In[90]:


final=pd.concat(frame)
final


# In[91]:


final.rename(columns={"standard_title": "standard"}, inplace=True)
final


# In[92]:


final['standard'] = final['standard'].replace(['GM-STD-10'],'10')
final['standard'] = final['standard'].replace(['GM-STD-9'],'9')
final['standard'] = final['standard'].replace(['GM-STD-8'],'8')
final['standard'] = final['standard'].replace(['GM-STD-7'],'7')
final['standard'] = final['standard'].replace(['GM-STD-6'],'6')
final['subject_title'] = final['subject_title'].replace(['Science & Technology'],'Science')
final


# In[ ]:





# In[93]:


final.head()


# In[ ]:





# In[94]:


final.info()


# In[95]:


categorical=final.select_dtypes(include=['object']).columns.tolist()
categorical


# In[96]:


final = final.astype({'standard':'int'})
final


# In[97]:


final.info()


# In[ ]:





# In[98]:


part2=pd.read_csv("PAL_Questions_v1.csv")
part2


# In[99]:


merged=pd.merge(final,part2, on='standard')
merged


# # training on entire data 

# In[100]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectFromModel


# In[101]:


features=merged["title"]
target=merged["Cognitive Difficulty"]
label_encoder = LabelEncoder()
encoded_target = label_encoder.fit_transform(target)




# In[102]:


splitter=StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=42)
for train_index,test_index in splitter.split(features,encoded_target):
  X_train,X_test=features.iloc[train_index],features.iloc[test_index]
  y_train,y_test=encoded_target[train_index],encoded_target[test_index]


# In[103]:


count_vectorizer=CountVectorizer(max_features=1000)
X_train_count=count_vectorizer.fit_transform(X_train)
X_test_count=count_vectorizer.transform(X_test)


# In[104]:


model=RandomForestClassifier(n_estimators=10, max_depth=5)
model.fit(X_train_count,y_train)

sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X_train, y_train)
sel.get_support()


# In[ ]:


selected_feat= X_train.columns[(sel.get_support())]
len(selected_feat)
print(selected_feat)
pd.Series(sel.estimator_,feature_importances_,.ravel()).hist()


# In[ ]:


predictions=model.predict(X_test_count)

accuracy=accuracy_score(y_test,predictions)
print(f'Accuracy:{accuracy}')


# In[ ]:


import random
student_id=540

merged['percent_category']=pd.cut(merged['percent'],bins=[0,33,66,100],labels=['low','medium','high'])

desired_category=merged.loc[df_merged['student_id']==student_id,'percent_category'].values[0]
if desired_category=='low':
    desired_bloom=['Knowledge','Comprehension']
elif desired_category=='medium':
    desired_bloom=['Application','Analysis']
else:
    desired_bloom=['Synthesis','Evaluation']

selected_questions=merged.loc[(df_merged['percent_category']==desired_category) & (merged["Bloom's Taxonomy"].isin(desired_bloom)),'title'].sample(n=5)
print("Selected Questions:")
for i,question in enumerate(selected_questions,start=1):
    print(f"Question {i}: {question}")


# # Standard 6 science questions

# In[ ]:





# In[105]:


part3=part2[8771:]
part3


# In[106]:


part4=final.sort_values(by=['subject_title','standard'])
part4


# # Stanndard 10 Science results

# In[107]:


part5=part4[10525:12042]
part5


# In[108]:


df_merged = part3.merge(part5, on='standard', how='outer')
df_merged


# In[109]:


df_merged.info()


# In[110]:


ax=sns.stripplot(x="Cognitive Difficulty", y="percent", data=df_merged);
ax.set_title(" difficulty vs percent ")


# In[111]:


ax=sns.stripplot(x="Difficulty Levels", y="percent", data=df_merged);
ax.set_title(" difficulty vs percent ")


# In[112]:


df_merged['percent'].value_counts()


# In[129]:


df_merged['id'].value_counts()


# In[113]:


sns.histplot(df_merged['percent'])


# In[114]:


sns.histplot(df_merged['Cognitive Difficulty'])


# In[115]:


df_merged['Cognitive Difficulty'].value_counts()


# In[116]:


sns.histplot(df_merged["Bloom's Taxonomy"])


# In[117]:


df_merged["Bloom's Taxonomy"].value_counts()


# In[118]:


df_merged.isnull().sum()


# In[119]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit


# In[ ]:





# In[120]:


features=df_merged["title"]
target=df_merged["Cognitive Difficulty"]


# In[121]:


label_encoder = LabelEncoder()
encoded_target = label_encoder.fit_transform(target)


# In[122]:


splitter=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in splitter.split(features,encoded_target):
  X_train,X_test=features.iloc[train_index],features.iloc[test_index]
  y_train,y_test=encoded_target[train_index],encoded_target[test_index]


# In[123]:


count_vectorizer=CountVectorizer(max_features=1000,stop_words='english')
X_train_count=count_vectorizer.fit_transform(X_train)
X_test_count=count_vectorizer.transform(X_test)


# In[124]:


model=RandomForestClassifier(n_estimators=10, max_depth=5)
model.fit(X_train_count,y_train)


# In[125]:


predictions=model.predict(X_test_count)


# In[126]:


accuracy=accuracy_score(y_test,predictions)
print(f'Accuracy:{accuracy}')


# In[138]:


import random
import requests
student_id=540

df_merged['percent_category']=pd.cut(df_merged['percent'],bins=[0,33,66,100],labels=['low','medium','high'])

desired_category=df_merged.loc[df_merged['student_id']==student_id,'percent_category'].values[0]
if desired_category=='low':
    desired_bloom=['Knowledge','Comprehension']
elif desired_category=='medium':
    desired_bloom=['Application','Analysis']
else:
    desired_bloom=['Synthesis','Evaluation']

selected_questions=df_merged.loc[(df_merged['percent_category']==desired_category) & (df_merged["Bloom's Taxonomy"].isin(desired_bloom)),'id'].sample(n=5)
print("Selected Questions:")
for i,number in enumerate(selected_questions,start=1):
    print(f"Question id {i}: {number}")
    
    
api_endpoint = "https://erp.triz.co.in/api/pal_questions"

# Prepare data associated with the selected student ID
selected_questions = df_merged[df_merged['student_id'] == 555].to_dict(orient='records')[0]
#selected_questions=np.random.choice([df_merged['student_id'] == '540'], 5, replace=False)
# Send data to API
response = requests.post(api_endpoint, json=selected_questions)

# Check response
print(response.text)



# In[128]:


import requests

# Assuming your API endpoint is "http://your-api.com/predict"
api_endpoint = "https://erp.triz.co.in/api/pal_questions"

# Prepare data associated with the selected student ID
data = df_merged[df_merged['student_id'] == 540].to_dict(orient='records')[0]

# Send data to API
response = requests.post(api_endpoint, json=data)

# Check response
print(response.text)


# In[ ]:


import pandas as pd
import numpy as np
import requests
import json
import random


# Select a random student ID from the DataFrame
random_student_id = np.random.choice(df_merged['student_id'])

# Define the ML model function to predict 5 questions based on the student ID
#def predict_questions(random_student_id):
    # Your ML model implementation here
    # This function should return predictions for 5 questions based on the student ID
    # For demonstration purposes, let's assume we have a function predict_questions_from_model
    # that takes a student_id as input and returns predictions as a list
   
student_id=720
    
df_merged['percent_category']=pd.cut(df_merged['percent'],bins=[0,33,66,100],labels=['low','medium','high'])

desired_category=df_merged.loc[df_merged['student_id']==student_id,'percent_category'].values[0]
if desired_category=='low':
 desired_bloom=['Knowledge','Comprehension']
elif desired_category=='medium':
 desired_bloom=['Application','Analysis']
else:
 desired_bloom=['Synthesis','Evaluation']

selected_questions=df_merged.loc[(df_merged['percent_category']==desired_category) & (df_merged["Bloom's Taxonomy"].isin(desired_bloom)),'title'].sample(n=5)
print("Selected Questions:")
for i,question in enumerate(selected_questions,start=1):
 print(f"Question {i}: {question}")
   
 #predictions = predict_questions_from_model(student_id)
 #return predictions

# Prepare the data for the API request
data = {
    'student_id': 720,
    'predictions': selected_questions
}

# Make a POST request to send the data to the API
api_url = 'https://erp.triz.co.in/api/pal_questions'
response = requests.post(api_url,data)

# Print the response from the API
print(response.json())

# Now let's demonstrate using Postman to get data and post data using this API.
# In Postman:
# 1. Set the request type to GET and enter the API endpoint to get data (e.g., http://your-api-url.com/students)
# 2. Set the request type to POST and enter the API endpoint to post data (e.g., http://your-api-url.com/predictions)
# 3. Set the request body to JSON and enter the sample data for posting (e.g., {"student_id": 12345, "predictions": [1, 2, 3, 4, 5]})
# 4. Send the request and observe the response from the API




# In[ ]:





# In[90]:


# K NEAREST NEIGHBOUR


# In[91]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


# Step 2: Feature Selection
X = df_merged[['percent']]
y = df_merged['Cognitive Difficulty']

# Step 3: Normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Model Training
k = 5  # You can adjust the value of k
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)

# Step 6: Prediction
y_pred = knn_model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print(f'Accuracy:{accuracy}')

# Step 7: Question Assignment
def assign_questions(percent):
    if percent <= 33:
        return np.random.choice(df_merged[df_merged['Cognitive Difficulty'] == 'Easy']['title'], 5, replace=False)
    elif 34 <= percent <= 66:
        return np.random.choice(df_merged[df_merged['Cognitive Difficulty'] == 'Medium']['title'], 5, replace=False)
    else:
        return np.random.choice(df_merged[df_merged['Cognitive Difficulty'] == 'Hard']['title'], 5, replace=False)

# Example usage for a specific student
student_percent = 90  # Replace with the actual percent value for a student
questions_for_student = assign_questions(student_percent)
print(f"Questions for Student :{questions_for_student}")


# In[92]:


import requests

def get_random_questions(student_id):
    # Define the API endpoint
    api_endpoint = "https://erp.triz.co.in/api/pal_questions"
    
    # Prepare the payload with the student ID
    payload = {
        "student_id": student_id
    }
    
    try:
        # Send POST request to the API endpoint
        response = requests.post(api_endpoint, json=payload)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Print the response data
            print("Response received successfully:")
            print(response.json())
        else:
            print("Error:", response.status_code)
    except Exception as e:
        print("An error occurred:", e)

# Example usage: Provide the student ID as input
student_id = "your_student_id_here"
get_random_questions(student_id)


# In[93]:


import pickle

# Assuming your trained model is named 'rf_model'
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)


# In[94]:


import joblib

# Load the model
model = joblib.load('model.pkl')


# In[ ]:





# In[133]:


from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json(force=True)
    
    # Make predictions
    predictions = model.predict(data['features'])
    
    # Return the predictions as JSON
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:





# In[ ]:





# In[ ]:




