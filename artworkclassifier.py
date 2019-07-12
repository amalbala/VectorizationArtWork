import cv2
import os, shutil
import numpy as np
import histograms
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
import pandas as pd


class ArtworkClassifier:

    def __init__(self, imageinput):
        self.image = imageinput
        self.artworkpath = "/media/antonio/Data/DataSets/Raw/5kIcons"
        self.realimagespath = "/media/antonio/Data/DataSets/Raw/FlickrLogos-32_dataset_v2/FlickrLogos-v2/allimages"
        self.base_dir = "/media/antonio/Data/DataSets/Projects/ArtWorkClassification"
        self.train_dir = os.path.join(self.base_dir, 'train')
        self.validation_dir = os.path.join(self.base_dir, 'validation')
        self.test_dir = os.path.join(self.base_dir, 'test')


    @classmethod
    def fromfile(cls, path, name):
        return cls(cv2.imread(os.path.join(path,name)))

    def generateDataForOneClass(self, originalpath, label, trainsize, validationsize, testsize):

        train_ar_dir = os.path.join(self.train_dir, label)
        os.mkdir(train_ar_dir)

        validation_ar_dir = os.path.join(self.validation_dir, label)
        os.mkdir(validation_ar_dir)

        test_ar_dir = os.path.join(self.test_dir, label)
        os.mkdir(test_ar_dir)
        file_list = os.listdir(originalpath)
        file_list_sorted = sorted(file_list)

        filesfortraining = [file_list_sorted[i] for i in range(trainsize)]
        filesforvalidation = [file_list_sorted[i]for i in range(trainsize, trainsize + validationsize)]
        filesfortest = [file_list_sorted[i] for i in range(trainsize + validationsize, trainsize + validationsize + testsize)]

        for file in filesfortraining:
            src = os.path.join(originalpath, file)
            dst = os.path.join(train_ar_dir, file)
            shutil.copy(src, dst)

        for file in filesfortest:
            src = os.path.join(originalpath, file)
            dst = os.path.join(test_ar_dir, file)
            shutil.copy(src, dst)

        for file in filesforvalidation:
            src = os.path.join(originalpath, file)
            dst = os.path.join(validation_ar_dir, file)
            shutil.copy(src, dst)


    def createDirectories(self):
        os.mkdir(self.base_dir)

        os.mkdir(self.train_dir)
        os.mkdir(self.validation_dir)
        os.mkdir(self.test_dir)

        self.generateDataForOneClass(self.artworkpath, 'artwork', 400, 50, 50)
        self.generateDataForOneClass(self.realimagespath, 'realphoto', 4000, 500, 500)

    def generateModel(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                      loss='binary_crossentropy',
                      metrics=['acc'])

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)


        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary'
        )

        validation_generator = test_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary'
        )



        csv_logger = CSVLogger('ArtvectorClassiferTraining.log', separator=',', append=False)


        history = model.fit_generator(
            train_generator,
            steps_per_epoch=200,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=50,
            callbacks=[csv_logger]

        )

        model.save('ArtvectorClassifer.h5')


    def plottraining(self):
        log_data = pd.read_csv('ArtvectorClassiferTraining.log', sep=',', engine='python')

        acc = log_data['acc']
        val_acc = log_data['val_acc']
        loss = log_data['loss']
        val_loss = log_data['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.interactive(False)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and valnidation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()


    def generateCNNClasifier(self):
        self.createDirectories()
        self.generateModel()
        self.plottraining()


    def isartwork(self):

        # self.generateCNNClasifier()

        model = models.load_model('ArtvectorClassifer.h5')

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary'
        )

        filenames = test_generator.filenames

        prediction_artwork = model.predict_generator(
            test_generator,
            steps=50)

        prediction_filenames = [(prediction_artwork[i], filenames[i]) for i in range(len(filenames)) ]

        prediction = pd.DataFrame(prediction_filenames, columns=['predictions', 'filenames']).to_csv('prediction.csv')

        # horizdifkernel = np.array([1, -1])
        # vertdifkernel = np.array([[1], [-1]])
        # diagdifKernel = np.array([[1, 0], [0, -1]])
        # antidiagdifKernel = np.array([[0, 1], [-1, 0]])
        # kernels = [horizdifkernel, vertdifkernel, diagdifKernel, antidiagdifKernel]
        # imagesdif = []
        # # First order differences
        # for kernel in kernels:
        #     imageresidual = cv2.filter2D(self.image, ddepth=cv2.CV_32F, kernel=kernel)
        #     imagesdif.append(imageresidual)
        #
        # # Second order differences
        # for i in range(len(kernels)):
        #     for j in range (i, len(kernels)):
        #         imagerfirst = cv2.filter2D(self.image, ddepth=cv2.CV_32F, kernel=kernels[i])
        #         imageresidual = cv2.filter2D(imagerfirst, ddepth=cv2.CV_32F, kernel=kernels[j])
        #         imagesdif.append(imageresidual)
        #
        # print("Images Generated:  " + str(len(imagesdif)))
        #
        # eqh_b, eqh_r, eqh_g  = histograms.histCalculation_BGR(imageresidual,
        #                                                       size=512,
        #                                                       rangemin=-255,
        #                                                       rangemax=255)
        #
        # imagesize = (self.image.shape[0] * self.image.shape[1])
        #
        # eqh_b /= imagesize
        # eqh_r /= imagesize
        # eqh_g /= imagesize
        # histograms.histPlot(imageresidual, size=512, rangemin=-255, rangemax=255)

        return True

