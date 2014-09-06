import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def main():
    print ('>>> enter ');
    # create the training & test sets
    print ('>>> reading data file')
    dataset = pd.read_csv('Data/train.csv')
    
    print ('>>> remove label ')
    target = dataset.label.values
    train = dataset.drop('label', axis=1).values
    
    #print (dataset.columns)
    
    print ('>>> reading test file')
    testdf = pd.read_csv('Data/test.csv')
    test = testdf.values
    imageIds =  range (1, testdf.axes[0].size + 1)
    # create and train the random forest
    # n_jobs set to -1 will use the number of cores present on your system.
    print ('>>> creating classifier')
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    
    print ('>>> training the  model')
    rf.fit(train, target)
   
    print ('>>> prediciton')
    # urf.predict_prob(test) for probabilty
    fr_results = rf.predict(test)
    
    #predicted_results = [x[1] for x in fr_results]
    print (fr_results)
    
    print ('>>> extract series')
    predicted_results = pd.Series(fr_results)
    imageIds = pd.Series(imageIds)
    print (imageIds)
    
    result = pd.DataFrame({  'ImageId' : imageIds, 'Label' : predicted_results })
    print ('>>> saving file ')
    result.to_csv('Results/submission.csv', index=False)
    
    print ('>>> End ')
if __name__ == "__main__":
    main()
