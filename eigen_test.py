import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define eigenvalues for each tau (9 values per tau)
eigenvalues_dict = {
    "1e-10": [3.87990819, 3.29354178, 2.86815017, 1.72900023, 1.53820242,
              1.1276864, 0.82311341, 0.56398039, 0.45017821],
    "1e-08": [3.87945371, 3.29319947, 2.86790408, 1.72871499, 1.53784389,
              1.12751187, 0.82294073, 0.56387027, 0.45009828],
    "1e-06": [3.87353184, 3.28998006, 2.86438538, 1.72586388, 1.53638188,
              1.12600691, 0.82208586, 0.56281613, 0.44930001],
    "1e-04": [3.84058135, 3.26796225, 2.83846886, 1.70160832, 1.52281785,
              1.10601145, 0.81567231, 0.55384941, 0.4416238]
}

# Convert to DataFrame
df = pd.DataFrame(eigenvalues_dict)
df.index = np.arange(1, 10)  # Change index from 1 to 9
df.index.name = "Eigenvalue Index"

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5, cbar_kws={'label': 'Eigenvalue'})

# Plot aesthetics
plt.title("Trend of LDA Eigenvalues Across Regularization (τ)", fontsize=14)
plt.xlabel("Regularization Parameter (τ)", fontsize=12)
plt.ylabel("Eigenvalue Index", fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
