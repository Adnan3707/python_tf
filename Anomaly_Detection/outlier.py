import pandas as pd
import numpy as np

# Create the DataFrame
df = pd.DataFrame({
    'Invoice ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'Amount': [85, 92, 91, 87, 90, 94, 95, 60, 89, 120, 129, 450]
})

# Apply IQR method to detect outliers in Amount
q1 = df['Amount'].quantile(0.25)
q3 = df['Amount'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[(df['Amount'] < lower_bound) | (df['Amount'] > upper_bound)]

print(f"Q1: {q1}")
print(f"Q3: {q3}")
print(f"IQR: {iqr}")
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")
print("Outliers:")
print(outliers)
