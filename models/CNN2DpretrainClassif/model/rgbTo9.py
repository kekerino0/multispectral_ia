import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import tensorflow_datasets as tfds
from keras.utils import Sequence
import os
from PIL import Image

## Calculating SSIM
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

datasetPath = r'C:\Users\STAGIAIRE\Documents\model\data\imagenet\tiny-imagenet-200'

#imgPath = r'C:\Users\STAGIAIRE\Documents\model\CNN2DpretrainClassif\model\rouge.jpg'

# path to rgb image to 9 bandify
imgPath = r'CNN2DpretrainClassif\data\gears\images\ScenePET1\Rendered_Image\ENVI\ScenePET1_rgb.png'

# path to grey original images
dir = r'C:\Users\STAGIAIRE\Documents\model\CNN2DpretrainClassif\data\gears\images\ScenePET1\Rendered_Image\Bands'


#imgPath = r'C:\Users\STAGIAIRE\Documents\model\CNN2DpretrainClassif\model\240.0.0.png'

curveMultiPath = r'C:\Users\STAGIAIRE\Documents\SSF_Integral_Multi'
curveRGBPath = r'C:\Users\STAGIAIRE\Documents\SSF_Nikon_D850'

# afficher viridis crop 
# afficher viridis crop normalisé
# objectif : voir si les les différences sont proportionnelles
# PSNR

image = cv2.imread(imgPath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def cropColorChecker(image):
    image = cv2.resize(image, (500,500))
    x,y,w,h = cv2.selectROI('select color checker', image)
    image = image[y:y+h,x:x+w]
    cv2.imshow('color checker',image)
    #cv2.waitKey(0)
    # print(image[24][25])
    return image, x,y,w,h
#print(type(image))

cropped,x,y,w,h = cropColorChecker(image)


def getCurvesData(curvePath):
    # gets curve data from file
    rgbCurves = os.listdir(curvePath)
    curves = {}
    for i, curve in enumerate(rgbCurves):
        temp = []
        with open(curvePath + '\\' + curve) as curveData:
            next(curveData)
            for line in curveData:
                splitLine = line.split()
                temp.append(splitLine)
                # wavelength.append(line[0])
                # amplitude.append(line[1])
        curves[curve] = temp
    #print(curves)
    return curves

# RGBCurves = getCurvesData(curveRGBPath)
# MultiCurves = getCurvesData(curveMultiPath)
#MultiCurves = getCurvesData(curveMultiPath)
# print(RGBCurves)

#######################    Integrale en commun   ################

def integraleRectangle(curve):
    # aire sous courbe par echantillon et total
    rectangles = []
    integrale = 0
    for i in range(0, len(curve)-1):
        width = float(curve[i + 1][0]) - float(curve[i][0])
        height = float(curve[i][1])
        integRect = width * height
        # print('integrect : ', integRect)
        rectangles.append(integRect)
        integrale = integrale + integRect
    return rectangles, integrale
        
# rectanglesRGB, integraleRGB = integraleRectangle(RGBCurves['GreenValues.spd'])
# rectanglesMulti, integraleMulti = integraleRectangle(MultiCurves['spectral_Order_Normalized_Band_1.spd'])
#print(integ)

def IntegraleCommune(rectanglesCurveRGB, rectangleCurveMulti, integraleMulti):
    diff = []
    for i in range(0, len(rectanglesCurveRGB)):
        temp = min(rectanglesCurveRGB[i], rectangleCurveMulti[i])
        diff.append(temp)
    #print('diff : ',diff)
    #print('sum diff : ',sum(diff), 'integrale totale : ',integraleMulti)
    percent = sum(diff) / integraleMulti
    return percent

# percent = IntegraleCommune(rectanglesRGB, rectanglesMulti, integraleMulti)
# print(percent)

def ratioCalculIntegral(RGBFolder, MultiFolder):
    mat = []
    RGBData = getCurvesData(RGBFolder)
    MultiData = getCurvesData(MultiFolder)
    #if (len(RGBData) != len(MultiData)):
    #    print('Not the same sampling')
    #    return 0
    for multiCurve in MultiData:
        multiValues = []
        for RGBCurve in RGBData:
            rectanglesMulti, integMulti = integraleRectangle(MultiData[multiCurve])
            rectanglesRGB, integRGB = integraleRectangle(RGBData[RGBCurve])
            IntegCommune = IntegraleCommune(rectanglesRGB, rectanglesMulti, integMulti)
            multiValues.append(IntegCommune)
        mat.append(multiValues)
    return mat

def percent(mat):
    percentMat = []
    for i, line in enumerate(mat):
        percentMatLine = []
        total = sum(line)
        percentMatLine.append(line[0]/total)
        percentMatLine.append(line[1]/total)
        percentMatLine.append(line[2]/total)
        percentMat.append(percentMatLine)
    #print('percentMat : ',percentMat)
    return percentMat

# mat = ratioCalculIntegral(curveRGBPath, curveMultiPath)
# mat = percent(mat)
#print('mat : ', mat)

###########################   correlation   ############################################

def correlation(curveRGB, curveMulti):
    #print('curve RGB : ',curveRGB)
    integ = 0
    corr = []
    for i in range(0,len(curveMulti)):
        corrValue = float(curveRGB[i][1]) * float(curveMulti[i][1])
        corr.append(corrValue)
        integ = integ + corrValue * 10
    return corr, integ

#corr, integ = correlation(RGBCurves['GreenValues.spd'], MultiCurves['spectral_Order_Normalized_Band_1.spd'])
#print('correlation curve : ', corr, 'integrale : ', integ)

def ratioCalcul(RGBFolder, MultiFolder):
    mat = []
    RGBData = getCurvesData(RGBFolder)
    MultiData = getCurvesData(MultiFolder)
    #if (len(RGBData) != len(MultiData)):
    #    print('Not the same sampling')
    #    return 0
    for multiCurve in MultiData:
        multiValues = []
        for RGBCurve in RGBData:
            corrValue, integ = correlation(MultiData[multiCurve], RGBData[RGBCurve])
            multiValues.append(integ)
        mat.append(multiValues)
    return mat

mat = ratioCalcul(curveRGBPath, curveMultiPath) 
mat = np.array(mat) * 10
#print('rat : ',rat)

####################### afficher courbes de sensibilité #############################

def dispCurves(RGBFolder, MultiFolder):
    RGBData = getCurvesData(RGBFolder)
    MultiData = getCurvesData(MultiFolder)
    plt.figure(8)
    arr = []
    for element in RGBData:
        temp = []
        for value in RGBData[element]:
            temp.append(float(value[1]))
        arr.append(temp)
    for element in MultiData:
        temp = []
        for value in MultiData[element]:
            temp.append(float(value[1]))
        arr.append(temp)
    X = np.linspace(360,760,39)
    for i, curve in enumerate(arr):
        #print('curve : ',curve)
        plt.plot(X, curve, label = f'Curve {i+1}', color = f'C{i}')
    plt.legend()
    plt.ylim(0, 0.02)
    return arr

dispCurves(curveRGBPath, curveMultiPath)
####################### passage de RGB à 9 canaux #################################
#def dispCubeDiff(cube1, cube2):


def rgbTo9(img, weights):
    img = cv2.resize(img, (500,500))
    img9cropped = []
    B = img[:, :, 2]
    G = img[:, :, 1]
    R = img[:, :, 0]
    
    #splitImg = [[R],
    #            [G],
    #            [B]]

    #weightedRGB = np.matmul(weights,splitImg)
    
    bands = []
    plt.figure(1, figsize=(3,3))
    plt.title('reconstructed 9 bands')
    for i, line in enumerate(weights):
        # print(line[2] , '*R' , line[1] , '*G ', line[0] , '*B')
        band = line[2]*R + line[1]*G + line[0]*B
        #print(band)
        bands.append(band)
    # max = np.amax(bands)
    # min = np.amin(bands)
    # bands = (bands - min)/(max-min)
    # print('bands size : ', len(bands))
    # print('max reconstitué : ',max)
    # for i, band in enumerate(bands):
    #     # band = band[y:y+h, x:x+w]
    #     img9cropped.append(band)
    #     # print(x,y,w,h)
    #     # print(len(band), len(band[0]))
    #     plt.subplot(3,3,i + 1)
    #     plt.title(f'c{i}')
    #     plt.imshow(band, cmap='viridis')
    #     plt.colorbar()
        
    #print(img9.shape)
    return bands, img9cropped

############################################## Affichage des trucs #########################################

# cmap = plt.get_cmap('viridis')
# norm = plt.Normalize(vmin=-1, vmax=1)


original = os.listdir(dir)
plt.figure(2,figsize = (3,3))
plt.title('original 9 bands')
croppedArr = []
originalImages = []
for i, path in enumerate(original):
    img = cv2.imread(dir + '\\' + path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (500,500))
    #img = np.array(img)
    originalImages.append(img)
    img = img[y:y+h,x:x+w]
    # print(len(img), len(img[0]))
    croppedArr.append(img)
max = np.amax(originalImages)
min = np.amin(originalImages)
originalImages = (originalImages - min)/(max - min)
croppedArr = croppedArr/max
print('max original : ',max)

for k in range(0,len(originalImages)):
    plt.subplot(3,3,k+1)
    plt.title(f'o{k}')
    plt.imshow(croppedArr[k], cmap='viridis')
    plt.colorbar()


plt.figure(3,(2,2))
plt.title('original image')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rgb = cv2.resize(rgb, (500,500))
plt.subplot(2,2,1)
plt.title('original')
plt.imshow(rgb[y:y+h, x:x+w], cmap = 'viridis')



#print('rgb : ',rgb)
bands, img9cropped = rgbTo9(image, mat)
#print(img9.shape) 
diffMat = np.array(bands) - np.array(originalImages)
#print('reconstructed bands : ',bands)
#print('original bands : ',originalImages)
diffMax = np.amax(diffMat)
diffMin = np.amin(diffMat)
diffMatNorm = (diffMat - diffMin)/(diffMax - diffMin)*2-1

plt.figure(4,(3,3))
plt.title('differences')
for i,diff in enumerate(diffMat):
    
    plt.subplot(3,3,i+1)
    plt.imshow(diff[y:y+h,x:x+w], cmap='viridis')
    plt.colorbar()


plt.figure(5,(3,3))
plt.title('differences normalisées par image')
for i,diffNorm in enumerate(diffMat):
    diffNormMax = np.amax(diffNorm)
    diffNormMin = np.amin(diffNorm)
    diffNorm = ((diffNorm - diffNormMin)/(diffNormMax - diffNormMin))*2 - 1
    plt.subplot(3,3,i+1)
    plt.imshow(diffNorm[y:y+h,x:x+w], cmap='bwr')
    plt.colorbar()

plt.figure(6,(3,3))
plt.title('differences normalisées sur la totalité des images')
for i,diffNorm in enumerate(diffMatNorm):
    plt.subplot(3,3,i+1)
    plt.imshow(diffNorm[y:y+h,x:x+w], cmap='bwr')
    plt.colorbar()


# print(info)


##################################################       regénérer images avec combinaisons de RGB        #################################


def getClasses(dataFolder):
    return os.listdir(dataFolder)

def getImages(classFolder):
    return os.listdir(classFolder)

def remakeData(imgFolder, targetFolder, weights, trainOrVal):
    classes = getClasses(imgFolder)
    os.mkdir(targetFolder + '\\' + trainOrVal)
    for classe in classes:
        os.mkdir(targetFolder + '\\' + trainOrVal + '\\' + classe)
        images = getImages(imgFolder + '\\' + classe)
        # print('images : ', images) # vides
        for imagePath in images:
            image = cv2.imread(imgFolder + '\\' + classe + '\\' + imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bands, image = rgbTo9(image, weights)
            os.mkdir(targetFolder + '\\' + trainOrVal + '\\' + classe + '\\' + imagePath)
            for i, band in enumerate(bands):
                bandImg = Image.fromarray(band)
                bandImg = bandImg.convert('L')
                bandImg.save(targetFolder + '\\' + trainOrVal + '\\' + classe + '\\' + imagePath + '\\' + str(i) +'.png' )

imageFolderTrain = r'C:\Users\STAGIAIRE\Documents\model\CNN2DpretrainClassif\data\imagenette2\train'
imageFolderVal = r'C:\Users\STAGIAIRE\Documents\model\CNN2DpretrainClassif\data\imagenette2\val'
targetFolder = 'C:\\Users\\STAGIAIRE\\Documents\\model\\CNN2DpretrainClassif\\data\\multiSpectralImagenette2'


# remakeData(imageFolderTrain, targetFolder, mat, 'train')
# remakeData(imageFolderVal, targetFolder, mat, 'val')



#################################################### calculating SSIM/PSNR ######################################################

def getSSIM(original_matrix, reconstructed_matrix):
    SSIMs = []
    SSIMImages = []
    for i in range(0,len(original_matrix)):
        SSIM, SSIMImage = compare_ssim(original_matrix[i], reconstructed_matrix[i], full=True, data_range=1)
        SSIMs.append(SSIM)
        SSIMImages.append(SSIMImage)
    return SSIMs, SSIMImages

def getPSNR(original_matrix, reconstructed_matrix):
    PSNRs = []
    for i in range(0,len(original_matrix)):
        PSNR = cv2.PSNR(original_matrix[i], reconstructed_matrix[i])
        PSNRs.append(PSNR)
    return PSNR

PSNRs = getPSNR(originalImages, bands)
SSIMs, SSIMImages = getSSIM(originalImages, bands)

print(PSNRs)
print(SSIMs)

# Visualize SSIM image
# print('SSIMSs : ',SSIMSs)
# plt.figure(7,(3,3))
# print('taille SSIMIMAGES : ',SSIMImages)
# for i,SSIMImage in enumerate(SSIMImages):
#     plt.subplot(3,3,i+1)
#     plt.imshow(SSIMImage[y:y+h, x:x+w], cmap='viridis')
#     plt.colorbar()
#     plt.title('SSIM Image')

#plt.show()



 


