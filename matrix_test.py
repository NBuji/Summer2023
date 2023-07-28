# Created by: nbujiashvili

# testing csv file importation


import csv
matrix = []
with open('A.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        matrix.append(row)
print(matrix[28][28])