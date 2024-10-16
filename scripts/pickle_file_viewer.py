import pickle

# Path to the pickle file
file_path = 'scripts/inchikeys.pkl'

# Load the pickle file
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Print the first few rows/items (assuming it's a list or similar)
for item in data[:5]:
    print(item)

# convert to csv the first 1000 items
import csv

# Path to the CSV file
csv_file_path = 'scripts/inchikeys.csv'

# Write the data to a CSV file
with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    for item in data[:1000]:
        writer.writerow([item])