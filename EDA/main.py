# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
data = pd.read_csv('./Diabetes_prediction.csv')
print('Dataset shape:', data.shape)
print('\nFirst 5 rows:')
print(data.head())
print('\nSummary statistics:')
print(data.describe(include='all'))
print('\nMissing values:')
print(data.isnull().sum())

# %%
# Identify target column if present
possible_targets = ['Outcome', 'Diabetes', 'diabetes', 'target', 'Target']
target_col = next((col for col in possible_targets if col in data.columns), None)

# Plot target distribution if available
if target_col is not None:
    plt.figure(figsize=(6, 4))
    data[target_col].value_counts().plot(kind='bar', color=['steelblue', 'tomato'])
    plt.title(f'{target_col} Distribution')
    plt.xlabel(target_col)
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# %%
# Plot histogram for all numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 0:
    ncols = 2
    nrows = int(np.ceil(len(numeric_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(data[col], bins=20, edgecolor='black', color='skyblue')
        axes[i].set_title(f'{col} Distribution')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# %%
# Boxplots for numeric columns grouped by target (if available)
if target_col is not None and target_col in numeric_cols:
    other_numeric_cols = [col for col in numeric_cols if col != target_col]
    if len(other_numeric_cols) > 0:
        fig, axes = plt.subplots(len(other_numeric_cols), 1, figsize=(10, 3 * len(other_numeric_cols)))
        for idx, col in enumerate(other_numeric_cols):
            data.boxplot(column=col, by=target_col, ax=axes[idx], patch_artist=True)
            axes[idx].set_title(f'{col} by {target_col}')
            axes[idx].set_xlabel(target_col)
            axes[idx].set_ylabel(col)
        plt.tight_layout()
        plt.show()

# %%
# Correlation heatmap for numeric columns
if len(numeric_cols) > 1:
    corr = data[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.03)

    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
    ax.set_yticklabels(numeric_cols)
    ax.set_title('Correlation Heatmap')

    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', color='black')

    plt.tight_layout()
    plt.show()

# %%
# Scatter plots for first few continuous variables
if len(numeric_cols) >= 2:
    plot_cols = numeric_cols[:4]
    if len(plot_cols) > 1:
        fig, axes = plt.subplots(len(plot_cols) - 1, len(plot_cols) - 1, figsize=(12, 12))
        idx = 0
        for i in range(len(plot_cols)):
            for j in range(len(plot_cols)):
                if i != j:
                    ax = axes[i - 1 if i > 0 else 0, j if j > 0 else 0] if len(plot_cols) > 2 else axes[i if i > 0 else 0, j if j > 0 else 0]
                    ax.scatter(data[plot_cols[j]], data[plot_cols[i]], alpha=0.6)
                    ax.set_xlabel(plot_cols[j])
                    ax.set_ylabel(plot_cols[i])
                    idx += 1
        if len(plot_cols) > 2:
            for ax in axes.flat:
                ax.set_visible(True)
        plt.tight_layout()
        plt.show()

# %%
print('\nExploratory plotting complete.')
# %%
