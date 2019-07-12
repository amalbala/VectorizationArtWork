import unittest
import os
import artworkclassifier
from keras.preprocessing.image import ImageDataGenerator
from keras import models
import pandas as pd

param_list = ['artwork01.png', 'artwork02.png', 'artwork03.png', 'wallart01.PNG', 'YingYangBW.jpg']

class TestVectorized(unittest.TestCase):

    def test_artwork(self):
        for p in param_list:
            with self.subTest(msg='Testing ' + p):
                awclassifier = artworkclassifier.ArtworkClassifier.fromfile('./images', p)
                isAw = awclassifier.isartwork()
                self.assertTrue(isAw, 'artwork bad classification')

    def test_realphoto(self):
        awclassifier = artworkclassifier.ArtworkClassifier.fromfile('./images', 'realphoto01.jpg')
        isAw = awclassifier.isartwork()

        self.assertTrue(not isAw, 'artwork bad classification')


    def testCNNClasifier(self):

        # awclassifier = artworkclassifier.ArtworkClassifier.fromfile('./images', 'realphoto01.jpg')
        # awclassifier.generateCNNClasifier()

        self.base_dir = "/media/antonio/Data/DataSets/Projects/ArtWorkClassification"
        self.test_dir = os.path.join(self.base_dir, 'test')
        model = models.load_model('ArtvectorClassifer.h5')

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            shuffle=False,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary'
        )

        test_generator.reset()

        filenames = test_generator.filenames

        prediction_artwork = model.predict_generator(
            test_generator,
            steps=50)

        prediction_filenames = [
            ('Real Photo' if prediction_artwork[i] > 0.5 else 'ArtWork', prediction_artwork[i], filenames[i]) for i in
            range(len(filenames))]

        prediction = pd.DataFrame(prediction_filenames, columns=['Class', 'Prediction', 'filenames']).to_csv(
            'prediction.csv')

        predicted_artwork = prediction_artwork[:50]
        predicted_realphotos = prediction_artwork[:-500]

        predicted_correctly = (sum(i <= 0.5 for i in predicted_artwork) +
                               sum(i > 0.5 for i in predicted_realphotos)) / 500.

        self.assertTrue(predicted_correctly > 0.8)


if __name__ == '__main__':
    unittest.main()
