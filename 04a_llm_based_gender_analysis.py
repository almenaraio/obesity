# -*- coding: utf-8 -*-
"""
Obesity research in specialty journals from 2000 to 2023: A bibliometric analysis
"""

import os
import pandas as pd
import ast
import json
import time
from google.colab import drive
import google.generativeai as genai

# Mount Google Drive
drive.mount('/content/drive')

# Paths
path_dir = '/content/drive/My Drive/DATASETS/OBESITY.JOURNALS/'
output_dir = '/content/drive/My Drive/DATASETS/OBESITY.JOURNALS/gender_results/'

# Load the dataset
file_name = os.path.join(path_dir, 'merged_results_filtered.csv')
df = pd.read_csv(file_name)
df.head()

df.info()

# Define periods
periods = {
    '2000-2007': (2000, 2007),
    '2008-2015': (2008, 2015),
    '2016-2023': (2016, 2023)
}

def parse_researchers(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

df['researchers'] = df['researchers'].apply(parse_researchers)

# Create period DataFrames
dfs_periods = {}
for label, (start, end) in periods.items():
    df_period = df[(df['year'] >= start) & (df['year'] <= end)][['id', 'researchers']].reset_index(drop=True)
    dfs_periods[label] = df_period

# OPTIONAL: print counts per period
for label, df_p in dfs_periods.items():
    print(f"{label}: {len(df_p)} rows")

# Now we have:
# dfs_periods['2000-2007'], dfs_periods['2008-2015'], dfs_periods['2016-2023']

genai.configure(api_key="YOUR_API_KEY")

# Load Gemini model
model = genai.GenerativeModel('gemini-1.5-pro')

# Function to get gender of a single author
def get_gender_single(first_name, last_name):
    prompt = (
        "You are an expert in determining gender by names.\n"
        "Given the author's first and last name, respond with exactly one of these words only: man, woman, or undetermined.\n"
        f"First Name: {first_name}\n"
        f"Last Name: {last_name}\n\n"
        "Gender:"
    )
    try:
        response = model.generate_content(prompt)
        text = response.text.strip().lower()
        # Only keep the first word if there are extra explanations
        text = text.split()[0]
        if text in {"man", "woman", "undetermined"}:
            return text
        else:
            return "undetermined"
    except Exception as e:
        print(f"Error: {e}")
        return "undetermined"

# Function to process a dataframe and save results
def process_dataframe(df_period, output_csv_path):
    results = []

    for idx, row in df_period.iterrows():
        authors = row['researchers']
        genders_list = []
        man = woman = undetermined = 0

        for author in authors:
            first_name = author.get("first_name", "")
            last_name = author.get("last_name", "")
            author_id = author.get("id", "")

            gender = get_gender_single(first_name, last_name)

            if gender == "man":
                man += 1
                genders_list.append({"man": author_id})
            elif gender == "woman":
                woman += 1
                genders_list.append({"woman": author_id})
            else:
                undetermined += 1
                genders_list.append({"undetermined": author_id})

            # Small delay to be nice to API
            time.sleep(0.2)

        results.append({
            'id': row['id'],
            'researchers': json.dumps(authors),
            'gender': json.dumps(genders_list),
            'man': man,
            'woman': woman,
            'undetermined': undetermined
        })

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df_period)} records")

    # Create and save DataFrame
    df_result = pd.DataFrame(results)
    df_result.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")


# Process loop
for label, df_p in dfs_periods.items():
    output_path = os.path.join(output_dir, f'gender_results_{label}.csv')
    print(f"\nProcessing period: {label}")
    start_time = time.time()
    process_dataframe(df_p, output_path) # Process dataframe
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Finished period '{label}' in {elapsed:.2f} seconds.")