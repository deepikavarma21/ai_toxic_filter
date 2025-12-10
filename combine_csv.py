import pandas as pd
import glob
import os

# 1. Make sure the datasets folder exists
if not os.path.exists("datasets"):
    print("Error: 'datasets' folder not found!")
else:
    # 2. List all CSV files in the folder
    all_files = glob.glob("datasets/*.csv")
    
    if len(all_files) == 0:
        print("No CSV files found in 'datasets' folder.")
    else:
        # 3. Read and combine all CSV files
        df_list = [pd.read_csv(file) for file in all_files]
        combined_df = pd.concat(df_list, ignore_index=True)
        
        # 4. Show the first few rows
        print("First 5 rows of combined data:")
        print(combined_df.head())
        
        # 5. Explore the dataset
        print("\nBasic info:")
        print(combined_df.info())
        
        print("\nSummary statistics:")
        print(combined_df.describe())
        
        print("\nMissing values per column:")
        print(combined_df.isnull().sum())
        
        # 6. Optional: Save the combined CSV
        combined_df.to_csv("datasets/combined_data.csv", index=False)
        print("\nCombined CSV saved as 'datasets/combined_data.csv'")