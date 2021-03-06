import csv
import random

colors = ['black', 'blue', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']

with open('./merged_labelled_csv/train.csv', 'w') as fouttrain:
    fieldnames = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX', 'Color']
    csvwritertrain = csv.DictWriter(fouttrain, fieldnames=fieldnames, lineterminator='\n')
    csvwritertrain.writeheader()
    with open('./merged_labelled_csv/test.csv', 'w') as fouttest:
        fieldnames = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX', 'Color']
        csvwritertest = csv.DictWriter(fouttest, fieldnames=fieldnames, lineterminator='\n')
        csvwritertest.writeheader()
        for color in colors:
            with open('./trimmed_csv/' + color + '.csv') as fin:
                csvreader = csv.DictReader(fin)
                i = 0
                for row in csvreader:
                    del row['timestamps']
                    row['Color'] = colors.index(color)
                    if(i < 5000):
                        if(i % 10 == 0):
                            csvwritertest.writerow(row)
                        else:
                            csvwritertrain.writerow(row)
                    i+=1
