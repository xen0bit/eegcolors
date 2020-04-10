import csv

colors = ['black', 'blue', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']

with open('./merged_labelled_csv/out.csv', 'w') as fout:
    fieldnames = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX', 'Color']
    csvwriter = csv.DictWriter(fout, fieldnames=fieldnames, lineterminator='\n')
    csvwriter.writeheader()
    for color in colors:
        with open('./trimmed_csv/' + color + '.csv') as fin:
            csvreader = csv.DictReader(fin)
            for row in csvreader:
                del row['timestamps']
                row['Color'] = str(color)
                #print(row)
                csvwriter.writerow(row)
