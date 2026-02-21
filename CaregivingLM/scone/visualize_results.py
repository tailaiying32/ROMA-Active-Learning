import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file
df = pd.read_csv('results.csv', index_col=0)

# Create the heatmap
plt.figure(figsize=(8, 6))

# Since lower scores are better, we'll use a colormap where darker colors represent lower values
# We'll invert the data for visualization so darker = better (lower values)
# Invert the values by subtracting from max value
# df_inverted = df.max().max() - df

sns.heatmap(df, 
           annot=df,  # Show original values in cells
           fmt='.4f',   # Format numbers to 4 decimal places
           cmap='viridis',  # viridis (darker = higher inverted values = lower original values)
           cbar_kws={'label': 'Movement Tracking Score (darker = better performance)'},
           linewidths=2.0,  # Increased line width for more separation
           linecolor='white')

plt.title('Movement Tracking Scores by Condition and Direction')
plt.xlabel('Movement Direction')
plt.ylabel('Condition')
plt.tight_layout()

# Save the plot
plt.savefig('result.png', dpi=300, bbox_inches='tight')
print("Heatmap saved as result.png")

# Also display some statistics
print("\nDataset Statistics:")
print(f"Min value: {df.min().min():.4f}")
print(f"Max value: {df.max().max():.4f}")
print(f"Mean value: {df.mean().mean():.4f}")