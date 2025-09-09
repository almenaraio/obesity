# -*- coding: utf-8 -*-
"""
# Obesity research in specialty journals from 2000 to 2023: A bibliometric analysis

# Install dimcli:
!pip install dimcli
"""

import os
import pandas as pd
import json
import dimcli
from dimcli import Dsl
import sys

# Set working directory
os.chdir('C:/temp')

# Open the file containing the API key
with open('dimensions_key.txt', 'r') as file:
    lines = file.readlines()

# Get the first line (the API key is on the first line)
api_key = lines[0].strip()

# Login
# See: https://digital-science.github.io/dimcli/getting-started.html#authentication
ENDPOINT = "https://app.dimensions.ai"
dimcli.login(key=api_key, endpoint=ENDPOINT)
dsl = dimcli.Dsl()

# List of ISSNs
issn_list = [
    "18768237", "21624968", "20552238", "24518476",
    "20529538", "17588103", "1871403x", "21532168",
    "16624025", "20900708", "20476302", "09608923",
    "14677881", "19307381", "03070565"
]

# List of field sets to query
field_sets = [
    "id+abstract",
    "id+altmetric",
    "id+basics",
    "id+authors_count",
    "id+researchers",
    "id+categories",
    "id+concepts",
    "id+concepts_scores"
    "id+funders",
    "id+supporting_grant_ids",
    "id+recent_citations",
    "id+reference_ids"
]

# Query the database using 'query_iterative' (i.e., iterative querying for looping over a query that produces more than 1000 results)
for issn in issn_list:
    for field_set in field_sets:
        my_query = f'''search publications
                       where issn = "{issn}"
                       return publications[{field_set}]'''
        results = dsl.query_iterative(my_query) # limit=250
        # Save results to a JSON file named by ISSN and field_set
        filename = f"results_{issn}_{field_set.replace('+', '_')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"results": results.json}, f, ensure_ascii=False, indent=4)
        print(f"ISSN: {issn} | Fields: {field_set} | Results saved to {filename}")

# Save all results as CSV files
for issn in issn_list:
    for field_set in field_sets:
        my_query = f'''search publications
                       where issn = "{issn}"
                       return publications[{field_set}]'''
        results = dsl.query_iterative(my_query)
        # Save results to a CSV file named by ISSN and field_set
        filename = f"results_{issn}_{field_set.replace('+', '_')}.csv"
        df = results.as_dataframe()
        df.to_csv(filename, index=False, encoding="utf-8")
        print(f"ISSN: {issn} | Fields: {field_set} | Results saved to {filename}")
