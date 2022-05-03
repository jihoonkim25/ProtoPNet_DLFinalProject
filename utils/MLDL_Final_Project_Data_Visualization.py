# Data visualization for the final project of CS 682 Machine Learning: Deep learning
import pandas as pd
import matplotlib.pyplot as plt
# data visualization
dataset = pd.read_csv('reference.csv')
original_patientID = dataset['PatientID']
original_probCOVID = dataset['probCOVID']
original_probSevereCOVID = dataset['probSevere']
new_patientID = dataset['PatientID_T']
new_probCOVID = dataset['probCOVID_T']
new_probSevereCOVID = dataset['probSevere_T']
new_patientID= new_patientID.iloc[0:400]
new_probCOVID = new_probCOVID.iloc[0:400]
new_probSevereCOVID = new_probSevereCOVID.iloc[0:400]
total_patient_number = len(original_patientID)
new_patient_number = len(new_patientID)
count_original_notCOVID = (original_probCOVID == 0).sum()
count_original_COVID = (original_probCOVID == 1).sum()
count_new_notCOVID = (new_probCOVID == 0).sum()
count_new_COVID = (new_probCOVID == 1).sum()
count_new_severeCOVID = (new_probSevereCOVID == 1).sum()
count_new_not_severeCOVID = (new_probSevereCOVID == 0).sum()
count_original_severeCOVID = (original_probSevereCOVID == 1).sum()
count_original_not_severeCOVID = (original_probSevereCOVID == 0).sum()
print('Original total patient number: ', total_patient_number)
print('New dataset patient number: ', new_patient_number)
print('Original patient number with COVID: ', count_original_COVID)
print('Original patient number without COVID: ', count_original_notCOVID)
print('New patient number with COVID: ', count_new_COVID)
print('New patient number without COVID: ', count_new_notCOVID)
print('New patient number with severe COVID: ', count_new_severeCOVID)
print('New patient number without severe COVID: ', count_new_not_severeCOVID)
print('Original patient number with severe COVID: ', count_original_severeCOVID)
print('Original patient number without severe COVID: ', count_original_not_severeCOVID)
# plot the distribution of the original dataset
original_label_1 = ['COVID', 'Not COVID']
original_label_2 = ['Severe COVID', 'Not severe COVID']
size_original_1 = [count_original_COVID, count_original_notCOVID]
size_original_2 = [count_original_severeCOVID, count_original_not_severeCOVID]
plt.pie(size_original_1, labels=original_label_1, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Original dataset COVID-19 distribution')
plt.show()
plt.pie(size_original_2, labels=original_label_2, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Original dataset severe COVID-19 distribution')
plt.show()
# plot the distribution of the new dataset
new_label_1 = ['COVID', 'Not COVID']
new_label_2 = ['Severe COVID', 'Not severe COVID']
size_new_1 = [count_new_COVID, count_new_notCOVID]
size_new_2 = [count_new_severeCOVID, count_new_not_severeCOVID]
plt.pie(size_new_1, labels=new_label_1, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('New dataset after trim COVID-19 distribution')
plt.show()
plt.pie(size_new_2, labels=new_label_2, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('New dataset after trim severe COVID-19 distribution')
plt.show()

