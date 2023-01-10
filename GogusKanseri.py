import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

#for visualizng results
from matplotlib.colors import ListedColormap

#for standartilation
from sklearn.preprocessing import StandardScaler

#for finding best paramters for knn
from sklearn.model_selection import train_test_split, GridSearchCV

#for examining results of our model
from sklearn.metrics import accuracy_score , confusion_matrix

#we choose KNN for this project , outlier detect -> local outlier factor
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis , LocalOutlierFactor

from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

#all data set
dataset = pd.read_csv("kanser.csv")

dataset.drop(['id'], inplace = True, axis = 1)

dataset = dataset.rename(columns = {"diagnosis" : "result"})

#labeling results
dataset["result"] = [1 if i.strip() == "M" else 0 for i in dataset.result]

#number of positive and negative results 
print(dataset.result.value_counts())

#for spotting missing value 
dataset.info()

#for wieving dataset
describe = dataset.describe()

#after exploring dataset summary we figured we need to do standardization because numbers scale much different

#correlation
corr_matrix = dataset.corr()

# for spotting features with high correlation so that we can remove some features
sns.clustermap(corr_matrix, annot = True )

#filtered plot with correlation > 0.70 we have some correlated features
threshold    = 0.70
plot_filter  = np.abs(corr_matrix["result"]) > threshold
corr_feature = corr_matrix.columns[plot_filter].tolist()
sns.clustermap( dataset[corr_feature].corr(), annot = True )
plt.show()

# %% outlier handling
results        = dataset.result
dataset_cutted = dataset.drop(["result"],axis = 1)
columns        = dataset_cutted.columns.tolist()

#LOF > 1 outlier else inlier 
clf     = LocalOutlierFactor()
predict = clf.fit_predict( dataset_cutted )
score   = clf.negative_outlier_factor_ 

outlier_score = pd.DataFrame()
outlier_score["score"] = score

outlier_threshold = -2
filter_outlier    = outlier_score["score"] < outlier_threshold
outlier_index     = outlier_score[filter_outlier].index.tolist()
 

#visualizng outliers 
plt.figure()
plt.scatter(dataset_cutted.iloc[outlier_index,0] , dataset_cutted.iloc[outlier_index,1] ,
            color = "blue" , s = 50 , label = "Outliers" )
plt.scatter(dataset_cutted.iloc[:,0] , dataset_cutted.iloc[:,1] , color = "k" , s = 3 , label = "Data Points" )

#if radius is big data point is more likely to be an outlier 
radius = (score.max() - score) / ( score.max() - score.min() )
outlier_score["radius"] = radius 
plt.scatter(dataset_cutted.iloc[:,0] , dataset_cutted.iloc[:,1] , s = 1000 * radius ,
            edgecolors="r" , facecolors = "none" , label= "Outlier Scores" )

#necesary for code to run
plt.legend()
plt.show()


#dropping outliers
results        = results.drop(outlier_index).values
dataset_cutted = dataset_cutted.drop(outlier_index)

# %% train & test set split

X_train , X_test , Y_train , Y_test = train_test_split(dataset_cutted , results , test_size = 0.3 , random_state = 13) 

# %% Standartdization

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

X_train_df = pd.DataFrame(X_train , columns = columns)

#in this veriable we should see the area mean list and inside it we should se mean value = 1 or very close to verify standartdization
X_train_df_describe = X_train_df.describe()

#We can see the distrubition for each class and each fetaure in this plot also we can see the outlier distrubition 
#We can also see the more important features with this plot
X_train_df["results"] = Y_train
data_melted = pd.melt(X_train_df, id_vars = "results",
                          var_name     = "features",
                          value_name   = "value")
plt.figure()
sns.boxplot(x = "features" , y = "value" , hue = "results" , data = data_melted)
plt.xticks(rotation = 90)
plt.show()

# %% KNN training

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train , Y_train)
result_prediction = knn.predict(X_test)


cm    = confusion_matrix(Y_test, result_prediction)
acc   = accuracy_score(Y_test, result_prediction)
score = knn.score(X_test , Y_test)

print("Score : " , score)
print("CM : " , cm)
print("Basic KNN acc : " , acc)

#At this period we dont know our knn is underfit or overfit 

#%% choose best parameters

def KNN_Best_Params(x_train , x_test , y_train , y_test):
    
    k_range = list(range(1,31))
    weight_options = ["uniform", "distance"]
    print()
    param_grid = dict(n_neighbors = k_range, weights = weight_options)
    
    knn  = KNeighborsClassifier()
    #10 cross validation
    grid = GridSearchCV(knn, param_grid , cv = 10 , scoring = "accuracy")
    grid.fit(x_train, y_train)
    print("Best training score : {} with parameters {}".format(grid.best_score_,grid.best_params_))
    print()
    
    knn  = KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train, y_train)
    
    results_prediction_test  = knn.predict(x_test)
    results_prediction_train = knn.predict(x_train)
    
    cm_test  = confusion_matrix(y_test, results_prediction_test)
    cm_train = confusion_matrix(y_train, results_prediction_train)
    
    acc_test  = accuracy_score(y_test, results_prediction_test)
    acc_train = accuracy_score(y_train, results_prediction_train)
    
    print("Test Score : {}, Train Score : {}".format(acc_test , acc_train))
    print()
    print("Cm Test :\n" , cm_test)
    print("CM Train :\n", cm_train)

    #returns best parameters
    return grid

grid = KNN_Best_Params(X_train, X_test, Y_train, Y_test)

#if train score > test score our model is overfitted 

# %% PCA 
#reducing the dimension

scaler = StandardScaler()
x_scaled = scaler.fit_transform(dataset_cutted)

pca = PCA(n_components = 2)
pca.fit(x_scaled)
X_reduced_pca = pca.transform(x_scaled)
pca_data = pd.DataFrame(X_reduced_pca , columns = ["feature1","feature2"])
pca_data["results"] = results
sns.scatterplot(x = "feature1", y = "feature2", hue = "results" , data = pca_data)
plt.title("PCA feature1 vs feature2")
plt.show()
X_train_pca , X_test_pca , Y_train_pca , Y_test_pca = train_test_split(X_reduced_pca , results , test_size = 0.3 , random_state = 13) 
print("PCA PREDICTION RESULT")
grid_pca = KNN_Best_Params(X_train_pca , X_test_pca , Y_train_pca , Y_test_pca)

# %% NCA

nca = NeighborhoodComponentsAnalysis(n_components = 2 , random_state = 13 )
nca.fit(x_scaled , results)
X_reduced_nca = nca.transform(x_scaled)
nca_data = pd.DataFrame(X_reduced_nca, columns = ["feature1","feature2"])
nca_data["results"]= results
sns.scatterplot(x = "feature1", y = "feature2", hue = "results" , data = nca_data )
plt.title("NCA feature1 vs feature2")
plt.show()
X_train_nca , X_test_nca , Y_train_nca , Y_test_nca = train_test_split(X_reduced_pca , results , test_size = 0.3 , random_state = 13) 
print("NCA PREDICTION RESULT")
grid_nca = KNN_Best_Params(X_train_nca , X_test_nca , Y_train_nca , Y_test_nca)

