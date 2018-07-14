import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sklearn.decomposition
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AffinityPropagation, SpectralClustering, AgglomerativeClustering, DBSCAN
import pickle
import datetime

# To reduce the size of data set and allow the code to analyze cluster in each state the data frame filtered by State
def getDataFrameforState(inputframe,stateName='CA'):
    df=inputframe[inputframe.StateName==stateName]
    df=df[df.Accepted>0]
    #df = df.reset_index()
    return df

	# get algo for no of cluster 
def getAlgoForCluster(algoType, noOfCluster, randomstate=10):
    if(algoType.lower()=='kmean'):
        kmean = KMeans(n_clusters=noOfCluster, random_state=randomstate)
    elif(algoType.lower()=='spectcluster'):
        return SpectralClustering(n_clusters=noOfCluster)
    elif(algoType.lower()=='affipropogation'):
        return  AffinityPropagation(damping=noOfCluster)
    elif(algoType.lower()=='algocluster'):
        return AgglomerativeClustering(n_clusters=noOfCluster)
    elif(algoType.lower()=='dbscan'):
        return DBSCAN(min_samples=noOfCluster)
    else :
        return None
        
    return kmean 

# get the fit predcit for Algo
def getFitPredictForAlgo(kMean,xcols):
    algo_val =kMean.fit_predict(xcols)
    return algo_val
	
# get the data frame  with PCA with component dimension and K mean for 
#one component as all x cols and another component as the unique value
# this provides the clutser for each row in data frame 
def getPCADataFrame(df,noOfCluster,algo_val,xcols,componetNum=2):
    pca = PCA(n_components = componetNum)
    matrix = np.matrix(pca.fit_transform(xcols))
    df_pca_matrix = pd.DataFrame(matrix)
    df_pca_matrix.columns = ['x','y']

    df_clusters = pd.DataFrame(df.iloc[:,0])
    #df_clusters['x'], df_clusters['y'] = df_pca_matrix['x'], df_pca_matrix['y']
    #df_clusters['cluster_label'] = algo_val
    #df_clusters['x']  = np.NAN
    df_clusters['x']  = df_pca_matrix['x'].values
    #df_clusters['y']  = np.NAN
    df_clusters['y']  = df_pca_matrix['y'].values
    df_clusters['cluster_label'] = algo_val

    return df_clusters

def getFilterDatasetForRowCount(df,noofRows,random=True, samplesize=0.5):
    if(len(df)>noofRows):
        if(random):
            df1,df2=train_test_split(df, shuffle=True,train_size=samplesize,test_size=samplesize)
            if(len(df1)>noofRows):
                df=df1.iloc[:noofRows,:]
            else:
                 df=(df1.append(df2,ignore_index=True)).iloc[:noofRows,:]
        else:
            df=df.iloc[:noofRows,:]
    
    return df

# Method to convert the category column into dummy columns 
def AddDummyColumnsToDataFrame(dfinput,colname,removeOrgColumn=False,removelastdummy=False):
    print('Add {}'.format(colname))
    temp =pd.get_dummies(dfinput[colname])
    # remove one column from dummies with least value.
  
    if removelastdummy:
        t=dfinput.groupby(colname).count().state
        col_name=((t[t.values==t.min()]).index).get_values()[0]
        if col_name in temp.columns:
            print('removed column {}'.format(col_name))
            temp=temp.drop([col_name], axis=1)
    
    # remove the main column after extracting dummy
    if removeOrgColumn:
        if colname in dfinput.columns:
            print('removed column {}'.format(colname))
            dfinput =dfinput.drop([colname], axis=1)
    else:
        print('left column {} in dataframe'.format(colname))
        
        
    for col in temp:
        temp.rename(columns={col: colname+'_'+str(col)}, inplace=True)
    
    return  pd.concat([dfinput,temp], axis=1,ignore_index=False)
	
def main():
        
   

    print("Begin DBSCAN Method !")
    pickle_file='pickle_selectdata_ML_All_Col_CA.sa'
    df_filterdata = pickle.load( open( pickle_file, "rb" ) )
    print('pick loaded')
    noofRows=100000
    df_filterdata=getFilterDatasetForRowCount(getDataFrameforState(df_filterdata,'CA'),noofRows)
    
	# Convert category columns to dummy columns=
    categoryColumns=['CountyCode']
    for col in categoryColumns:
        print(col)
        df_filterdata=AddDummyColumnsToDataFrame(df_filterdata,col)
    
    x_cols = np.matrix(df_filterdata.iloc[:,69:])
    print(len(df_filterdata))
    print (len(df_filterdata.index.unique()))
    print(len(x_cols))
  
    range_n_clusters = [1,2,3]#range(2,11)
    #silhouette_avgscores = []
    #silhouette_samples = []
	
	#Agglomerative Clustering
    print(str(datetime.datetime.now()))
	#find the best value for n_clusters parameter. 
    aag_predict_col={}
    aag_silh_score={}
    best_score = 0.0
    for n_clusters in range_n_clusters:
        ac = getAlgoForCluster('dbscan', n_clusters)#AgglomerativeClustering(n_clusters=n_clusters)
        #labels = ac.fit_predict(x_cols)
        labels = getFitPredictForAlgo(ac,x_cols)
        if n_clusters not in aag_predict_col:
            aag_predict_col[n_clusters]=labels
        silhouette_avg = silhouette_score(x_cols, labels, random_state=10)
        if n_clusters not in aag_silh_score:
            aag_silh_score[n_clusters]=silhouette_avg

        print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_n_clusters = n_clusters


    print('Best n_clusters parameter:',best_n_clusters)
    print ('Best DBSCAN score is', best_score)
    print(str(datetime.datetime.now()))

    print('completed the silhouette. saving pickle file')
    picklefilename='silhouette_avgscores_DBSCAN _ML.sa'
    print(picklefilename)
    # create pickle file for further use 
    pickle.dump(aag_silh_score,open(picklefilename,'wb'), protocol=4)
    picklefilename='silhouette_samples_DBSCAN _ML.sa'
    
    print(picklefilename)
    # create pickle file for further use 
    pickle.dump(aag_predict_col,open(picklefilename,'wb'), protocol=4)
    print('finish')


if __name__== "__main__":
    main()

print("done")