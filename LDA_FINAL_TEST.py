import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
# === STEP 1: Load and label each CSV ===

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load dataset
file_path = "/Users/aminentezari/Desktop/Thesis/Pyhton /LDA_KNN/final_re/zigzag_noise_results.csv"  # Replace with your actual path
df = pd.read_csv(file_path)

# Log-transform tau
df['log_tau'] = np.log10(df['tau'])

# Define independent variables
X = df[['amplitude', 'frequency', 'log_tau']]
X = sm.add_constant(X)  # Add intercept
y = df['accuracy']

# Fit linear regression model
model = sm.OLS(y, X).fit()

# Print regression summary
print(model.summary())

# Predict values for visualization
df['predicted_accuracy'] = model.predict(X)

# Plot actual vs predicted accuracy
plt.figure(figsize=(8, 5))
plt.scatter(y, df['predicted_accuracy'], color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
plt.xlabel("Accuracy")
plt.ylabel("Noise")
plt.title("Noise vs  Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()


file_path = "/Users/aminentezari/Desktop/Thesis/Pyhton /LDA_KNN/final_re/gaussian_noise_results1.csv"


df = pd.read_csv(file_path)

# Log-transform tau
df['log_tau'] = np.log10(df['tau'])

# Define independent variables and target
X = df[['noise_std', 'log_tau']]
X = sm.add_constant(X)  # Adds intercept
y = df['accuracy']

# Fit regression model
model = sm.OLS(y, X).fit()

# Print regression summary
print(model.summary())

# Predict and store predicted accuracy
df['predicted_accuracy'] = model.predict(X)

# Plot actual vs predicted accuracy
plt.figure(figsize=(8, 5))
plt.scatter(y, df['predicted_accuracy'], color='blue', label="Data Points")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Ideal Fit")
plt.xlabel("Actual Accuracy")
plt.ylabel("Predicted Accuracy")
plt.title("Actual vs Predicted Accuracy (Gaussian Noise Regression)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



# Load dataset
df = pd.read_csv("/Users/aminentezari/Desktop/Thesis/Pyhton /LDA_KNN/final_re/salt_pepper_noise_results.csv")

# Clean column names (safe to include)
df.columns = df.columns.str.strip()

# Add log_tau and interaction term
df['log_tau'] = np.log10(df['tau'])
df['interaction'] = df['density'] * df['log_tau']

# Use formula-style regression for interaction
model = smf.ols("accuracy ~ density + log_tau + density:log_tau", data=df).fit()

# Print model summary
print(model.summary())


plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="density", y="accuracy", hue="log_tau", palette="viridis", legend="full")
plt.title("Accuracy vs. Density with log(tau) Interaction")
plt.xlabel("Noise Density")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()
# Load the dataset
df = pd.read_csv("/Users/aminentezari/Desktop/Thesis/Pyhton /LDA_KNN/final_re/speckle_noise_results.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Log-transform tau
df['log_tau'] = np.log10(df['tau'])

# Define features and target
X = df[['noise_std', 'log_tau']]
X = sm.add_constant(X)  # Adds intercept
y = df['accuracy']

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print regression summary
print(model.summary())

# Predict accuracy and store in the DataFrame
df['predicted_accuracy'] = model.predict(X)

# Plot actual vs predicted accuracy
plt.figure(figsize=(8, 5))
plt.scatter(y, df['predicted_accuracy'], color='blue', label='Data Points')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit')
plt.xlabel("Actual Accuracy")
plt.ylabel("Predicted Accuracy")
plt.title("Actual vs Predicted Accuracy (Speckle Noise)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()