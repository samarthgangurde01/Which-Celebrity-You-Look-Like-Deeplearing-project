import tensorflow as tf
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from tqdm import tqdm

import pickle
filename=pickle.load(open("filenames.pkl","rb"))
model=VGGFace(model="resnet50",include_top=False,input_shape=(224,224,3),pooling="avg")

def feature_extractor(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result


features = []
for file in tqdm(filename):
    features.append(feature_extractor(file,model))

pickle.dump(features,open("embeddings.pkl",'wb'))