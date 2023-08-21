#!/usr/bin/env python
# coding: utf-8
Factor Analysis is a diagnostic analytics method used to discover factors that causes variability among correlated variables. 
The analysis below is an explanatory factor analysis which aims at identifying factors (hypothetical variables) that explain the correlations among a set of variables.
# In[1]:


#pip install factor_analyzer


# In[2]:


#Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt

Diabetes risk factor prediction dataset obtained from https://www.kaggle.com/datasets/abhishekkingsley/diabetes-risk-factor-prediction
# In[3]:


#Loading data
data=pd.read_csv("C:/DA_KE/diabetesdata.csv")
data.head()


# In[4]:


#Viewing columns
data.columns


# In[5]:


#Dropping unnecessary data
data.drop(['Unnamed: 0'], axis=1, inplace=True)
data=data.dropna()
data


# In[6]:


#Step 1: Evaluating factorability to determine the appropriateness of factor analysis using the Kaiser-Meyer-Olkin Measure of Sampling Adequacy

from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(data)
print("\n KMO Model\n", kmo_model)

The overall KMO for the data is 0.81 which is meritorious,
The value indicates that the planned factor analysis process can proceed. 
# In[7]:


#Step 2: Extracting factors(hypothetical variables) using the Principal Component Analysis Method
factor=FactorAnalysis().fit(data)
Factors=pd.DataFrame(factor.components_)
print(Factors)


# In[8]:


#Step 3: Choosing Factors 
# i: Using Kaiser Criterion  to get the eigen values
fa = FactorAnalyzer()
fa.fit(data)
fa.set_params(n_factors="data", rotation="varimax")

#Checking the Eigen Values
ev, v = fa.get_eigenvalues()
print(ev)

From the output, there are seven factors whose variance are greater than 1.
Therefore we choose only 7 factors.
# In[9]:


#ii: Creating scree plot using matplotlib and checking the shape of the plot 
plt.scatter(range(1,data.shape[1]+1),ev)
plt.plot(range(1,data.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.axhline(y=1, color='m',linestyle='solid')
plt.grid()
plt.show()


# In[10]:


#Step 4: Rotating Factors
#i: Generating rotated factor matrix
fa=FactorAnalyzer(rotation='varimax', n_factors=7)
fa.fit(data)

loadings=pd.DataFrame(fa.loadings_)
loadings.rename(columns=lambda x: 'Factor'+str(x+1), inplace=True)
loadings.index=data.columns
loadings

From the output above, 
Factor 1 has high factor loadings for GenHlth, MentHlth, PhysHlth and DiffWalk.
Factor 2 has high factor loadings for Diabetes_012, HighBP, HighChol, HeratDiseaseorAttack, GenHlth, Age
Factor 3 has high factor loadings for Education and Income
Factor 4 has high factor loadings for AnyHealthcare,NoDocdcCost and Age
Factor 5 has high factor loadings for BMI
Factor 6 has high factor loadings for Fruits and veggies
Factor 7 has high factor loadings for Smoker
# In[11]:


segments = loadings[loadings>=0.3].fillna(loadings[loadings<=-0.3])
segments


# # Conclusion
Factor 1 - Health Status
(GenHlth, MentHlth, PhysHlth, DiffWalk)
Factor 2 - Underlying Issues
(HighBP, HighChol, HeartDiseaseorAttack,Age)
Factor 3 - Lifestyle
(Education, Income)
Factor 4 - Medical History
(AnyHealthcare, NoDocdcCost, Age)
Factor 5 - Body Fat
(BMI)
Factor 6 - Diet
(Fruits, Veggies)
Factor 7 - Smoker
(Smoker)
# In[12]:


# Replacing factors with the interpreted segment names.
segment_names = ['Health Status','Underlying Issues','Lifestyle','Medical History','Body Fat','Diet', 'Smoker']
segments.columns = segment_names
segments

