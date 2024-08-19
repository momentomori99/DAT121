import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

from mpl_toolkits.mplot3d import Axes3D


#Importing the data
data = pd.read_csv("heart.csv")
target = data.iloc[:,-1]
data_no_targets = data.drop(data.columns[-1], axis=1) #drop the target column

#convert to numpy
X = data_no_targets.to_numpy()
t = target.to_numpy()

#Normalization of data
min_X = X.min()
max_X = X.max()
norm_X = (X - min_X) / (max_X - min_X)



#Calculating the corralation
corr = scipy.stats.pearsonr(X[:,0], t)

all_corr = np.zeros(13)

for i in range(13):
    corr = scipy.stats.pearsonr(X[:,i], t)
    #print(f"feature: {i} - {corr[0]} ")
    all_corr[i] = float(corr[0])


min_corr = all_corr.min()
max_corr = all_corr.max()
norm_corr = (all_corr - min_corr) / (max_corr - min_corr)



labels = np.array(['Age', 'sex', 'Chest pain', 'Resting Blood Pressure', 'Chol', 'Fasting Blood Sugar', 'Rest ECG', 'Max Heart Rate', 'Exercise induces angina', 'Old Peak', 'slp', 'caa', "thall"])

# Number of variables/features
num_vars = len(labels)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is a circle, so we need to "close the loop" by appending the start value to the end
values = norm_corr.tolist()
values += values[:1]
angles += angles[:1]

# Create the radar plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

ax.fill(angles, values, color='blue', alpha=0.25)
ax.plot(angles, values, color='blue', linewidth=2)

# Set the labels for each axis
ax.set_yticklabels([])  # Optionally hide the radial labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# Title of the plot
plt.title('Feature Correlations Radar Plot', size=15, color='black', y=1.1)

# Show the plot
plt.show()
######
