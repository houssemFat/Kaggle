from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype='f8')[1:]    
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open('Data/test.csv','r'), delimiter=',', dtype='f8')[1:]

    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    n_estimators = 100
    rf = RandomForestClassifier(n_estimators=n_estimators)
    rf.fit(train, target)
    predicted_probs = [[index + 1, x[1]] for index, x in enumerate(rf.predict_proba(test))]

    print predicted_probs
    """savetxt('Data/submission.csv', predicted_probs, delimiter=',', fmt='%d,%f', 
            header='MoleculeId,PredictedProbability', comments = '')
    
    print("RF: max_depth = {0}".format(n_estimators))
    print(metrics.f1_score(ytest, ypred))
    plt.figure()
    plt.imshow(metrics.confusion_matrix(predicted_probs, test),
               interpolation='nearest', cmap=plt.cm.binary)
    plt.colorbar()
    plt.xlabel("true label")
    plt.ylabel("predicted label")
    plt.title("RF: max_depth = {0}".format(n_estimators))
    plt.show()"""

if __name__=="__main__":
    main()