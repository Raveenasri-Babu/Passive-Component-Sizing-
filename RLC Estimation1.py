#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
df = pd.read_csv(r"C:\DC-DC Converter\boost_converter.csv")
df.head()
df.describe()


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation between variables')
plt.show()

# Vin Vs Time
plt.figure(figsize=(8,5))
plt.plot(df['Time'], df['Vin'], color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Input Voltage (Vin)')
plt.title('Input Voltage vs Time')
plt.grid()
plt.show()

# Vout Vs Time
plt.figure(figsize=(8,5))
plt.plot(df['Time'], df['Vout'], color='green')
plt.xlabel('Time (s)')
plt.ylabel('Output Voltage (Vout)')
plt.title('Output Voltage vs Time')
plt.grid()
plt.show()

#  Histograms for RLC
df[['Resistance1', 'Inductance', 'Capacitance']].hist(bins=30, figsize=(12,6), color='orange', edgecolor='black')
plt.suptitle('Distribution of Resistance1, Inductance, and Capacitance')
plt.show()


# In[14]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#  Rolling Stat
df['Vin_rolling_mean'] = df['Vin'].rolling(window=10).mean() 
df['Vout_rolling_mean'] = df['Vout'].rolling(window=10).mean()  
df['Vin_rolling_std'] = df['Vin'].rolling(window=10).std() 
df['Vout_rolling_std'] = df['Vout'].rolling(window=10).std() 

# Rate of Change
df['Vin_rate_of_change'] = df['Vin'].diff()  
df['Vout_rate_of_change'] = df['Vout'].diff()  

# Lag Features 
df['Vin_lag_1'] = df['Vin'].shift(1) 
df['Vout_lag_1'] = df['Vout'].shift(1)  

# Interaction Features
df['Vin_x_Inductance'] = df['Vin'] * df['Inductance']  
df['Vout_x_Resistance'] = df['Vout'] * df['Resistance1'] 

# Dimensionality Reduction with PCA
pca = PCA(n_components=2) 
pca_components = pca.fit_transform(df[['Vin', 'Resistance1', 'Inductance', 'Capacitance']])

# Add the PCA components as new features
df['PCA_1'] = pca_components[:, 0]
df['PCA_2'] = pca_components[:, 1]

# Preview the transformed dataframe with new features
print(df.head())

# Visualize the first few features to ensure the transformations
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(df['Time'], df['Vin_rolling_mean'], label='Vin Rolling Mean')
plt.title('Vin Rolling Mean')
plt.xlabel('Time (s)')
plt.ylabel('Vin')

plt.subplot(2, 2, 2)
plt.plot(df['Time'], df['Vout_rolling_mean'], label='Vout Rolling Mean')
plt.title('Vout Rolling Mean')
plt.xlabel('Time (s)')
plt.ylabel('Vout')

plt.subplot(2, 2, 3)
plt.plot(df['Time'], df['Vin_rate_of_change'], label='Vin Rate of Change')
plt.title('Vin Rate of Change')
plt.xlabel('Time (s)')
plt.ylabel('Rate of Change')

plt.subplot(2, 2, 4)
plt.plot(df['Time'], df['Vout_rate_of_change'], label='Vout Rate of Change')
plt.title('Vout Rate of Change')
plt.xlabel('Time (s)')
plt.ylabel('Rate of Change')

plt.tight_layout()
plt.show()


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

X = df[['Vin','Vout']]  
y = df[['Resistance1', 'Inductance', 'Capacitance']] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


base_model = RandomForestRegressor(random_state=42)
multi_output_model = MultiOutputRegressor(base_model)
multi_output_model.fit(X_train, y_train)

y_pred = multi_output_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²) Score:", r2)

# actual vs predicted
comparison = pd.DataFrame({'Actual_Resistance1': y_test['Resistance1'].values,
                           'Predicted_Resistance1': y_pred[:, 0],
                           'Actual_Inductance': y_test['Inductance'].values,
                           'Predicted_Inductance': y_pred[:, 1],
                           'Actual_Capacitance': y_test['Capacitance'].values,
                           'Predicted_Capacitance': y_pred[:, 2]})

print(comparison.head())


# In[16]:


#  Plot Actual vs Predicted (R)
plt.figure(figsize=(6,6))
plt.scatter(y_test['Resistance1'], y_pred[:,0], color='blue', edgecolor='k')
plt.plot([y_test['Resistance1'].min(), y_test['Resistance1'].max()],
         [y_test['Resistance1'].min(), y_test['Resistance1'].max()],
         'r--', lw=2)
plt.xlabel('Actual Resistance1')
plt.ylabel('Predicted Resistance1')
plt.title('Actual vs Predicted Resistance1')
plt.grid()
plt.show()

# Plot Actual vs Predicted (L)
plt.figure(figsize=(6,6))
plt.scatter(y_test['Inductance'], y_pred[:,1], color='green', edgecolor='k')
plt.plot([y_test['Inductance'].min(), y_test['Inductance'].max()],
         [y_test['Inductance'].min(), y_test['Inductance'].max()],
         'r--', lw=2)
plt.xlabel('Actual Inductance')
plt.ylabel('Predicted Inductance')
plt.title('Actual vs Predicted Inductance')
plt.grid()
plt.show()

# Actual vs Predicted (C)
plt.figure(figsize=(6,6))
plt.scatter(y_test['Capacitance'], y_pred[:,2], color='purple', edgecolor='k')
plt.plot([y_test['Capacitance'].min(), y_test['Capacitance'].max()],
         [y_test['Capacitance'].min(), y_test['Capacitance'].max()],
         'r--', lw=2)
plt.xlabel('Actual Capacitance')
plt.ylabel('Predicted Capacitance')
plt.title('Actual vs Predicted Capacitance')
plt.grid()
plt.show()

residuals = y_test.values - y_pred

plt.figure(figsize=(10,6))
plt.hist(residuals.flatten(), bins=30, color='orange', edgecolor='black')
plt.title('Distribution of Residuals (Errors)')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df[['Vin', 'Vout']]
y = df[['Resistance1', 'Inductance', 'Capacitance']]

model = LinearRegression()
model.fit(X, y)

def recommend_components(vin_value, vout_value):
    new_input = pd.DataFrame({'Vin': [vin_value], 'Vout': [vout_value]})
    prediction = model.predict(new_input)
    print(f"Recommended Resistance1: {prediction[0][0]:.6f} ohms")
    print(f"Recommended Inductance: {prediction[0][1]:.6f} H")
    print(f"Recommended Capacitance: {prediction[0][2]:.6f} F")


recommend_components(18, 14)


# In[22]:


recommend_components(14.67,19)


# In[ ]:




