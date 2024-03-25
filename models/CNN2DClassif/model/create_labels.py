import os

def create_labels():
    file_list = os.listdir(r'CNN2DClassif\data\images')
    for element in file_list:
        with open(f'CNN2DClassif\data\labels\{element}.txt', 'w') as label_file:
            if ('PET' in element):
                label_file.write('1, 0')
            elif ('PLA' in element):
                label_file.write('0, 1')

create_labels()