import cv2
import os
import os.path
import numpy as np
from sklearn.model_selection import train_test_split
import pathlib
from sklearn.ensemble import RandomForestClassifier

NUM_CLUSTERS = 40
print(pathlib.Path().absolute())

origin_dir2 = r"C:\Users\jpgia\Documents\Projetos\ImgClassifier\Assets\Data_Filtered_Resized"
destination_dir2 = r"C:\Users\jpgia\Documents\Projetos\ImgClassifier\Assets\new"


def collect_img_names(origin_dir2):
    X = []
    Y = []
    cont = 0
    for folder in os.listdir(origin_dir2):
        for file in os.listdir(os.path.join(origin_dir2,folder)):
            if cont > 50:
                break
            if not file.endswith(('.gif','.svg','.asp')):
                X.append(os.path.join(origin_dir2,folder,file))
                Y.append(folder)
            cont += 1        
    return X,Y

def cria_vocabulario(imagens, num_clusters):
    km = cv2.BOWKMeansTrainer(num_clusters)
    akaze = cv2.KAZE_create()
    for p in imagens:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        mask = np.ones(img.shape)
        kp, desc = akaze.detectAndCompute(img, mask)
        km.add(desc)
    return km.cluster()

def representa(vocab, img):
    kaze = cv2.KAZE_create()
    kp = kaze.detect(img)
    bowdesc = cv2.BOWImgDescriptorExtractor(kaze, cv2.FlannBasedMatcher())
    bowdesc.setVocabulary(vocab)
    return bowdesc.compute(img, kp)

def transforma_imagens(imagens, vocab):
    X = []
    for p in imagens:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        X.append(representa(vocab, img).flatten())
    return np.array(X)


X,Y = collect_img_names(origin_dir2)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=0)
print(X_train)
vocab = cria_vocabulario(X_train, NUM_CLUSTERS)
X_train = transforma_imagens(X_train, vocab)
# model = RandomForestClassifier()
# model.fit(X_train,y_train)




