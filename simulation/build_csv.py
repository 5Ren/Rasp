import csv
csv_path = 'pharos_simu/test.csv'

data_list = [
    'scanner',
    1100,
    1200
]

with open(csv_path, 'a') as f:
    writer = csv.writer(f, lineterminator='\n')
    for i in range(10):
        data_list[1] += 100

        writer.writerow(data_list)
