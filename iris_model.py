#This is a simpe IRIS model.This uses SVM and persist the model so that
#it can be used by other consumers

# Import libraries and packages
from sklearn import svm, datasets
import pickle 

# Load IRIS dataset
iris = datasets.load_iris()

# Split loaded data into independent and target features
X = iris.data  
y = iris.target

# Train Support Vector Machine (SVM) model with all data 
svmModel = svm.SVC(kernel='poly', degree=3, C=1.0).fit(X, y)

# Persisting model 
svmFile = open('SVMModel.pckl', 'wb')
pickle.dump(svmModel, svmFile)
svmFile.close()

print("done")