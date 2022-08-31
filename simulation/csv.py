import csv
csv_path = './test.csv'

with open(csv_path, newline='' ,encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:

        for column in row:
            print(column)