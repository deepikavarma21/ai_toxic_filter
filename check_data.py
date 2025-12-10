import pandas as pd

# 1. Load your combined CSV
combined_df = pd.read_csv("datasets/combined_data.csv", low_memory=False)

# 2. Check first few rows
print("First 5 rows of the dataset:")
print(combined_df.head())

# 3. Check for missing values in all columns
print("\nMissing values in all columns:")
print(combined_df.isnull().sum())

# 4. Specifically check the target column
target_col = 'toxic'  # replace this with the actual column you want to predict
missing_labels = combined_df[target_col].isnull().sum()
print(f"\nNumber of missing labels in '{target_col}':", missing_labels)

# 5. See which rows have missing labels
rows_with_missing = combined_df[combined_df[target_col].isnull()]
print(f"\nRows with missing '{target_col}':")
print(rows_with_missing)

# 6. Drop rows with missing comment text
combined_df = combined_df.dropna(subset=['comment_text'])

# 7. Optional: Drop rows with missing labels in 'toxic'
combined_df = combined_df.dropna(subset=[target_col])

print("Shape after dropping missing rows:", combined_df.shape)

# 8. Now you can continue with preprocessing and training
