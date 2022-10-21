import time
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics


TRANSLATION_FIELD = 'translation'
ORIGINAL_FIELD = 'review'
TARGET_FIELD = 'rating'

def grid(param_grid,model,bow,Y, cv):
    start = time.time()
    grid_clf = GridSearchCV(model, param_grid = param_grid, scoring = 'accuracy', n_jobs=2,verbose=4, cv=cv)
    grid_clf.fit(bow, y=Y) #ou sur train
    end = time.time()
    print("Best parameters = ", grid_clf.best_params_, " grid time : ", round((end - start)/60, 4))
    return(grid_clf.best_params_)


def train (pipeline,X0,Y0,X1,Y1):
    start = time.time()
    n = len(np.unique(Y1))
    matrix = np.zeros((n,n))
    accuracy = 0
    run = 0
    for train_ix, test_ix in CV.split(X0, Y0):
        run+=1
        xtrain, xtest = X0[train_ix], X1[test_ix]
        ytrain, ytest = Y0[train_ix], Y1[test_ix]

        #Training
        pipeline.fit(xtrain,ytrain)

        # Predicting with a test set
        predicted = pipeline.predict(xtest)

        # Accuracy
        a = metrics.accuracy_score(ytest, predicted)
        print("Model Accuracy ",run," : ", a)
        accuracy = accuracy + a

        #Confusion matrix
        m = metrics.confusion_matrix(ytest, predicted)
        matrix = matrix + m

    end = time.time()
    print("fit time : ", round((end - start)/60, 4))
    return matrix,accuracy/cv

# Load DataFrame
instance_df = pickle.load(open("english_to_french_film.dataset", "rb"))
ressource_df = pickle.load(open("french_to_english_film.dataset", "rb"))

# Define train and test
X_train = ressource_df[TRANSLATION_FIELD].values
y_train = ressource_df[TARGET_FIELD].values

X_test = instance_df[ORIGINAL_FIELD].values
y_test = instance_df[TARGET_FIELD].values

# Define model
name = "Bayes Naif"
vec = CountVectorizer(lowercase=True, ngram_range=(1, 2), max_df=0.90, min_df=5)
clf = MultinomialNB()
param_grid = {'alpha':[0.1,0.3,0.7,1,2,3,4,5,7,10]}
cv = 5

# Prepare the pipeline
bow = vec.fit_transform(X_train)
param = grid(param_grid,clf,bow,y_train, cv)
clf = MultinomialNB(**param)
pipeline = Pipeline([('vec', vec), ('clf', clf)])

#Train de classifier with the best parameters then show accuracy and plot confusion matrix
CV = StratifiedKFold(n_splits=cv, shuffle=True, random_state=1)
matrix, accuracy = train(pipeline, X_train, y_train, X_test, y_test)
