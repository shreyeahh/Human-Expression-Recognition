import keras
import numpy as np
import librosa
import soundfile as sf

class modelPredictions:

    def __init__(self, path, file):
        self.path = path
        self.file = file

    def load_model(self):
        self.loaded_model = keras.models.load_model(self.path)
        #return self.loaded_model.summary()

    def predictEmotion(self):
        data, sr = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=1)
        x = np.expand_dims(x, axis=0)
        predictedEmotion = self.loaded_model.predict(x)
        classes_x=np.argmax(predictedEmotion,axis=-1)
        print("The Predicted Emotion is :", self.convertclasstoemotion(classes_x))

    @staticmethod
    def convertclasstoemotion(p):
        #predictions(int) to understandable emotion labeling
        label_conversion = {'0': 'neutral','1': 'calm','2': 'happy','3': 'sad','4': 'angry','5': 'fearful','6': 'disgust','7': 'surprised'}
        for key, value in label_conversion.items():
            if (int(key) == p):
                label = value
        return label
    
    
    
p1 = modelPredictions(path='C:/Users/HP/Desktop/FINAL PROJECT/final_model.h5',file='C:/Users/HP/Desktop/FINAL PROJECT/fear.wav')
p1.load_model()
#called predictEmotion function to predict emotion type of input file
p1.predictEmotion()