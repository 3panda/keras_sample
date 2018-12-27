from PIL import Image
from sklearn import model_selection
import glob
import numpy as np

classes = ["panda", "cat", "penguin"]
num_classes = len(classes)
image_size = 50

X = []
Y = []
for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        # TODO: 今回は画像数を200枚と決め打ちしている
        if i >= 200:
            break
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

'''
画像のデータとラベルのデータをそれぞれ
訓練用と評価用のデータに分ける
オプションで無指定なので3:1
'''
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
print(xy)
# npy形式のファイルに保存
np.save("./ani.npy", xy)
