import os

def create_labels():
    file_list = os.listdir(r'CNN2DBbox\data\images')
    for element in file_list:
        with open(f'CNN2DBbox\data\labels\{element}.txt', 'w') as label_file:
            label_file.write('0, 0, 0, 0, 0')

create_labels()