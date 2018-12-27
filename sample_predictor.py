from keras.models import load_model
from PIL import Image
import sys
import numpy as np

classes = ["panda", "cat", "penguin"]
num_classes = len(classes)
image_size = 50


def main(filepath, model):
    """
    :param filepath: 予測する対象の画像ファイルのPath
    """
    try:
        image = Image.open(filepath)
        image = image.convert('RGB')
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        print(data)
        X = []
        X.append(data)
        X = np.array(X)

        # モデルのロード
        model = load_model(model)
        # 予測
        predict = model.predict([X])
        print(predict)
        result = predict[0]
        print(result)
        predicted = result.argmax()
        print(predicted)
        percentage = int(result[predicted] * 100)
        print(classes[predicted] + ", 確率：" + str(percentage) + " %")

    except Exception as e:
        print(e.args)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(filepath=sys.argv[1], model=sys.argv[2])
    else:
        print("usage.")
        print("{filename} <filepath(ex:xxxx.jpg), model(xxx.h5)>"
              .format(filename=sys.argv[0]))
