import pandas as pd
import numpy as np
# Create the DataFrame
data = {
    'Invoice ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'Amount': [85, 92, 91, 87, 90, 94, 95, 60, 89, 120, 129, 450]
}

df = pd.DataFrame(data)

# print(df)

mn = np.mean(df['Amount'])
sd = np.std(df['Amount'])
print(sd)

# cal z score
df['Z-score']=(df['Amount']-mn)/sd
print(df)
print('-'* 5)
th = 3
df['Outlier'] = np.abs(df['Z-score']) > th

print(df)