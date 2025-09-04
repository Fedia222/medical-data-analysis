import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Load the dataset
df = pd.read_csv("patient_data_up.csv", index_col="patient")
print("Dataset loaded successfully.")

# Display basic information about the dataset in a .txt file
with open("dfsummary.txt", "w") as f:
    # Dataset info
    f.write("\nDataset Info:\n")
    df.info(buf=f)

    # Statistical summary
    f.write("\n\nStatistical Summary:\n")
    f.write(df.describe().to_string())

    #number of patients by age
    f.write("\n\nNumber of Patients by Age:\n")
    f.write(df['age'].value_counts().sort_index().to_string())
    
    #number of pations by blood group
    f.write("\n\nNumber of Patients by Blood Group:\n")
    f.write(df['blood_group'].value_counts().to_string())
    
    # First rows
    f.write("\n\nFirst Rows of the Dataset:\n")
    f.write(df.head(20).to_string())

#lets make some graphics
# Histogram for Age
plt.figure(figsize=(10, 6))
plt.hist(df['age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig("age_distribution.png", dpi=600)   #save with high resolution
#plt.show()




