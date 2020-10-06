from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import numpy as np
import cv2
import matplotlib.pyplot as plt

# loading the MNIST digits dataset
mnist = datasets.load_digits()

# taking the MNIST dataset and creating the training and testing split, where 75% of the
# data is for training and the other 25% for testing

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
mnist.target, test_size=0.25, random_state=42)

#taking 10% of the training data for validation...

(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
test_size=0.1, random_state=84)

# show the no. of samples used
print("No. of training samples: "+str(len(trainLabels)))
print("No. of testing samples: "+str(len(testLabels)))


# training the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(trainData, trainLabels)
# evaluate the model and display the accuracy
score = model.score(valData, valLabels)
print("Accuracy of the model is %.2f%%" % ( score * 100))


print("Choosing 5 random digits and testing to see models prediction...")
          
#Looping over random digits to see the models prediction

for i in np.random.randint(0, high=len(testLabels), size=(5,)):
         image = testData[i]
         prediction = model.predict([image])[0]
         
          # show the prediction
         imgdata = np.array(image, dtype='float')
         pixels = imgdata.reshape((8,8))
         plt.imshow(pixels,cmap='gray')
         plt.annotate(prediction,(3,3),bbox={'facecolor':'white'},fontsize=16)
         print("The digit is : {}".format(prediction))
         plt.show()
         cv2.waitKey(0)
