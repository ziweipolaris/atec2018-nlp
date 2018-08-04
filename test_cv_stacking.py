
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import datasets
from sklearn.svm import l1_min_c

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y != 2]
y = y[y != 2]

X -= np.mean(X, 0)
cs = l1_min_c(X, y, loss='log') * np.logspace(0, 3)


print("Computing regularization path ...")
start = datetime.now()
clf = linear_model.LogisticRegressionCV(penalty='l2', tol=1e-6)
clf.fit(X, y)
print("This took ", datetime.now() - start)


# =============================================================
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression()

print('3-fold cross validation:\n')

stack = 2
if stack == 1:
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], 
                            meta_classifier=lr)
    for clf, label in zip([clf1, clf2, clf3, sclf], 
                        ['KNN', 
                        'Random Forest', 
                        'Naive Bayes',
                        'StackingClassifier']):

        scores = model_selection.cross_val_score(clf, X, y, 
                                                cv=3, scoring='accuracy')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
        % (scores.mean(), scores.std(), label))

elif stack == 2:
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                            use_probas=True,
                            average_probas=False,
                            meta_classifier=lr)
    for clf, label in zip([clf1, clf2, clf3, sclf], 
                        ['KNN', 
                        'Random Forest', 
                        'Naive Bayes',
                        'StackingClassifier']):

        scores = model_selection.cross_val_score(clf, X, y, 
                                                cv=3, scoring='accuracy')
    
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
        % (scores.mean(), scores.std(), label))
elif stack == 3:
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], 
                            meta_classifier=lr)

    params = {'kneighborsclassifier__n_neighbors': [1, 5],
            'randomforestclassifier__n_estimators': [10, 50],
            'meta-logisticregression__C': [0.1, 10.0]}

    grid = GridSearchCV(estimator=sclf, 
                        param_grid=params, 
                        cv=5,
                        refit=True)
    grid.fit(X, y)

    cv_keys = ('mean_test_score', 'std_test_score', 'params')

    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        print("%0.3f +/- %0.2f %r"
            % (grid.cv_results_[cv_keys[0]][r],
                grid.cv_results_[cv_keys[1]][r] / 2.0,
                grid.cv_results_[cv_keys[2]][r]))

    print('Best parameters: %s' % grid.best_params_)
    print('Accuracy: %.2f' % grid.best_score_)


import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools

gs = gridspec.GridSpec(2, 2)

fig = plt.figure(figsize=(10,8))

for clf, lab, grd in zip([clf1, clf2, clf3, sclf], 
                         ['KNN', 
                          'Random Forest', 
                          'Naive Bayes',
                          'StackingClassifier'],
                          itertools.product([0, 1], repeat=2)):

    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf)
    plt.title(lab)
plt.show()