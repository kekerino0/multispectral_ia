import cv2
import matplotlib.pyplot as plt
import os


from matplotlib.image import imread

# Remplacez 'chemin_vers_image.jpg' par le chemin de votre propre image

# Charger l'image avec Matplotlib
#image = imread(chemin_image)
#image = cv2.resize(image, (1000,1000))

#r = cv2.selectROI('select',image)
#print(r)


# with open(r'data\MeldrickData\labels'):
# Afficher l'image avec Matplotlib
# plt.imshow(image)
# plt.axis('off')  # Optionnel : désactiver les étiquettes des axes
# plt.show()
    
labelPath = 'data\MeldrickData\labels'
imagePath = 'data\MeldrickData\images'

def get_bbox(path, image_size):
    image = imread(path)
    image = cv2.resize(image, image_size)
    coords1 = cv2.selectROI('select darker object : ',image)
    coords2 = cv2.selectROI('select lighter object : ',image) 
    return coords1, coords2

def annoter(dossier, image_size):
    listImages = os.listdir(dossier)
    for i in range(1,len(listImages)):
        imagePath = f'data\MeldrickData\images\ScenePET{i}\Rendered_Image\Bands\\band_1.png'
        labelPath = f'data\MeldrickData\labels\ScenePET{i}.txt'
        coords1, coords2 = get_bbox(imagePath, image_size)
        coords1 = ', '.join(map(str, coords1))
        coords2 = ', '.join(map(str, coords2))
        print('coords1 : ',coords1)
        print('coords2 : ',coords2)
        with open(labelPath,'x') as labelFile:
            labelFile.write('1, ')
            labelFile.write(coords1)
            labelFile.write('\n')
            labelFile.write('2, ')
            labelFile.write(coords2)    
    return 0

annoter(imagePath, (500,500))