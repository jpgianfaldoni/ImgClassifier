import cv2
import os
import os.path
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


NUM_CLUSTERS = 40
force_new_vocab = 1

origin_dir2 = os.getcwd() + "\\Assets\\Data_Filtered_Resized" #r"C:\Users\jpgia\Documents\Projetos\ImgClassifier\Assets\Data_Filtered_Resized"


def collect_img_names(origin_dir2):
    X = []
    Y = []
    cont = 0
    for folder in os.listdir(origin_dir2):
        for file in os.listdir(os.path.join(origin_dir2,folder)):
            if cont > 150:
                break
            if not file.endswith(('.gif','.svg','.asp')):
                X.append(os.path.join(origin_dir2,folder,file))
                Y.append(folder)
            cont += 1  
        cont = 0          
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

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Greens, save_to_file = True):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots(figsize = (16,16))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if save_to_file:
        plt.savefig('Assets/files/' + title + '.pdf')
    return ax

print("Collecting image names...")
X,Y = collect_img_names(origin_dir2)
print("Separating train/test split...")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=0)
print("Bag of visual words...")
try:
    if force_new_vocab:
        raise(Exception)
    print("Attempting to load vocabulary.")
    if "cache" not in os.listdir(os.getcwd()):
        os.mkdir("cache")
    vocab = np.load(os.getcwd() + "\\cache\\vocab.npy", allow_pickle=True)
    print("Load successful!")
except Exception as E:
    print("Load failed! Recalculating vocabulary and saving result.")
    vocab = cria_vocabulario(X_train, NUM_CLUSTERS)
    np.save(os.getcwd() + "\\cache\\vocab.npy", vocab, allow_pickle=True)

print("Extracting features...")
X_train = transforma_imagens(X_train, vocab)
X_test = transforma_imagens(X_test, vocab)
model = RandomForestClassifier()
model.fit(X_train,y_train)
prediction = model.score(X_test, y_test)
print(prediction)
plot_confusion_matrix(y_test, model.predict(X_test), classes=model.classes_,title="ConfusionMatrix")



