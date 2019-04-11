import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

model_path = "models/"

experiment_name = "vent"

training_df = pd.read_csv(model_path + experiment_name + ".csv")

epochs = training_df.iloc[:,0]
loss = training_df.iloc[:,1]
val_loss = training_df.iloc[:,4]


plt.plot(epochs,loss,val_loss)
plt.show()
