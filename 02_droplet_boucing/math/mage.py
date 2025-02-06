import csv

my_name = []
my_suryo = []

path = 'results/0001.csv'

with open(path) as f:
    reader = csv.reader(f)