import pandas as pd
import numpy as np
import collections
import scipy
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.feature_selection import f_regression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef
from scipy import interp
from statistics import stdev
import scipy.stats as stats
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from matplotlib.legend_handler import HandlerLine2D
import warnings
warnings.filterwarnings("ignore")
import time
import itertools
import pickle


# In[2]:


df = pd.read_csv("/mnt/project/tmhpred3/osalvador/radiomics_lipoma_alt/thesis/data/harmonized/combined_harmonized.csv", sep="\t")

df2 = pd.read_csv("/mnt/project/tmhpred3/osalvador/radiomics_lipoma_alt/thesis/data/clinical/clinical_data_internal_combined.csv", sep=";", index_col=0)


# In[3]:


df2.drop(df2.columns[1:6], axis=1, inplace=True)
df2.drop(df2.columns[2], axis=1, inplace=True)
df = df2.join(df.set_index('ID_intern'), on='ID_intern')
df = df.rename({'Pathology': 'Type'}, axis=1)


# In[4]:


df


# In[5]:


# class imbalance
alts = len(df[df['Type'] == 1])
lipomas = len(df) - alts
total = len(df)
# print(f'{str("{:.0f}".format(lipomas/total*100))}:{str("{:.0f}".format(alts/total*100))} negative to positive ratio')


# In[6]:


features = list(df.columns[2:])

y = df['Type']
y = pd.DataFrame.to_numpy(y) # data object: numpy array

X = df.drop(df.columns[[0, 1]], axis=1)
X = pd.DataFrame.to_numpy(X) # data object: numpy array


# In[7]:


def plotCM(cm, i, loop, ct_o, ct_i, k, l):
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()
    plt.title("Confusion matrix " + r"$\bf{" + loop + "}$" + " loop")
    plt.savefig(f'/mnt/project/tmhpred3/osalvador/radiomics_lipoma_alt/thesis/results/mlpc/combined/mlpc-ncv{i}-cm-{loop}-o{ct_o}i{ct_i}-C-{k}-Pen-{l}.png')
    plt.close()


# In[8]:


def mrmr(xTrain, xTest, yTrain, given_features, K):
    #    X: pandas.DataFrame, features
    #    y: pandas.Series, target variable
    #    K: number of features to select
    
    xTrain = pd.DataFrame(data=xTrain, columns=given_features)
    
    # only to filter it at the end
    xTest = pd.DataFrame(data=xTest, columns=given_features)
    
    # for F-statistic
    y = pd.Series(yTrain)

    # compute F-statistics and initialize correlation matrix
    F = pd.Series(f_regression(xTrain, y)[0], index = xTrain.columns)
    corr = pd.DataFrame(.00001, index = xTrain.columns, columns = xTrain.columns)
    
    # initialize list of selected features and list of excluded features
    selected = []
    not_selected = xTrain.columns.to_list()
    
    # repeat K times
    for i in range(K):
        # compute (absolute) correlations between the last selected feature and all the (currently) excluded features
        if i > 0:
            last_selected = selected[-1]
            corr.loc[not_selected, last_selected] = xTrain[not_selected].corrwith(xTrain[last_selected]).abs().clip(.00001)
            
        # compute FCQ score for all the (currently) excluded features (this is Formula 2)
        score = F.loc[not_selected] / corr.loc[not_selected, selected].mean(axis = 1).fillna(.00001)
        
        # find best feature, add it to selected and remove it from not_selected
        best = score.index[score.argmax()]
        selected.append(best)
        not_selected.remove(best)
        
    # filter columns
    xTrain_filtered = xTrain.drop(not_selected, axis = 1)
    xTest_filtered = xTest.drop(not_selected, axis = 1)
    
    # print("Features: ", *xTrain_filtered.columns, sep='\n')
    
    # convert back to numpy array
    xTrain_filtered = pd.DataFrame.to_numpy(xTrain_filtered)
    xTest_filtered = pd.DataFrame.to_numpy(xTest_filtered)
    
    # print("!!!!", xTrain_filtered.shape, xTest_filtered.shape)
        
    return xTrain_filtered, xTest_filtered, selected


# In[9]:


# Minority Oversampling: SMOTE
def oversampling(X, y, ct_o, ct_i):
    counter = collections.Counter(y)
    # print("Before", counter)
    f.write(f'Before: {counter}')
    
    # scatter plot
    # for label, _ in counter.items():
    #     row_ix = np.where(y==label)[0]
    #     plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    # plt.legend()
    # # plt.show()
    # plt.savefig(f'../../results/rf/combined/rf-before-oversampling-o{ct_o}i{ct_i}.png')
    # plt.close()
    
    # original SMOTE paper suggests combining SMOTE with random undersampling of the majority class
    # over = SMOTE(sampling_strategy=0.1)
    # under = RandomUnderSampler(sampling_strategy=0.7)
    # 
    # steps = [('o', over), ('u', under)]
    # pipeline = Pipeline(steps=steps)
    # 
    # X, y = pipeline.fit_resample(X, y)
    
    smt = SMOTE()
    X, y = smt.fit_resample(X, y)
    
    counter = collections.Counter(y)
    # print("After", counter)
    f.write(f'After: {counter}')
    
    # for label, _ in counter.items():
    #     row_ix = np.where(y==label)[0]
    #     plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    # plt.legend()
    # # plt.show()
    # plt.savefig(f'../../results/rf/combined/rf-after-oversampling-o{ct_o}i{ct_i}.png')
    # plt.close()
    
    return X, y


# In[10]:


def pca_vis(X_train_i, features):
        X_train_i = pd.DataFrame(data=X_train_i, columns=features)
        tot_var = 0.95 # total variance
        pca_model = PCA(n_components = tot_var)
        
        X_train_pca = PCA(tot_var, svd_solver = 'full').fit(X_train_i)
        
        # print("Variance ratio:", X_train_pca.explained_variance_ratio_); print()
        # print("PCA dimensions:", X_train_pca.components_.shape[0]); print()
        # print("Reduced dimensions can explain {:.4f}".format(sum(X_train_pca.explained_variance_ratio_)*100),
        #       "% of the variance in the original data."); print()
        
        # components
        n_pcs= X_train_pca.components_.shape[0]
        
        # PCA coverts the features in array format; so, if we want to get the feature names:
        # most_important = [np.abs(X_train_pca.components_[i]).argmax() for i in range(n_pcs)]
        # most_important_names = [features[most_important[i]] for i in range(n_pcs)]
        # dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
        # p = pd.DataFrame(dic.items())
        # # print("New dimensions:\n", p); print()
        
        # X_train_i = pd.DataFrame.to_numpy(X_train_i) # data object: numpy array
        
        return n_pcs


# In[11]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(
    train_sizes_ncv,
    train_scores_mean,
    train_scores_std,
    test_scores_mean,
    test_scores_std,
    fit_times_mean,
    fit_times_std,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title("MLPC Learning Curve")
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    
    # train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
    #     estimator,
    #     X,
    #     y,
    #     cv=cv,
    #     n_jobs=n_jobs,
    #     train_sizes=train_sizes,
    #     return_times=True,
    # )
    # train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    # test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)
    # fit_times_mean = np.mean(fit_times, axis=1)
    # fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes_ncv,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes_ncv,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes_ncv, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes_ncv, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes_ncv, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes_ncv,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


# In[12]:


f = open("/mnt/project/tmhpred3/osalvador/radiomics_lipoma_alt/thesis/results/mlpc/combined/mlpc-output.txt", "w")


# In[13]:


# configure the nested CV procedure
# dataset D=(X, y)

# define number of folds based on dataset size and keep folds while looping through different models
K1 = 3 # outer
K2 = 3 # inner

# define the model
# Straight Forward modeling
model = MLPClassifier(solver='lbfgs', max_iter=1000)

# Grid Example (change according to dataset size)
mlpc_grid = {'hidden_layer_sizes': [(3,), (5,), (7,), (9,), (3, 3), (5, 5), (7, 7), (9, 9)],
             'activation': ['tanh', 'relu'],
             'alpha': np.logspace(-4, 2, 10)}


# In[14]:


# create a list of all possible hyperparameter combinations

keys, values = zip(*mlpc_grid.items())
search_space = [dict(zip(keys, v)) for v in itertools.product(*values)]

# search_space = []
# for solver_dict in mlpc_grid:
#     keys, values = zip(*solver_dict.items())
#     search_space += [dict(zip(keys, v)) for v in itertools.product(*values)]


# In[15]:


# search_space


# In[16]:


# GridSearchCV custom implementation
series = range(1, 51) # for multiple runs of CV

# metrics across all 150 models of the 50 nCV runs
accuracy_ncv  = []
balanced_accuracy_ncv  = []  
f1_ncv  = []
recall_ncv  = []
mcc_ncv = []
sensitivity_ncv = []
specificity_ncv = []

hyperparameters = []
hp_features = []

mean_accuracy = {}
mean_balanced_accuracy = {}
mean_f1 = {}
mean_recall = {}
mean_mcc = {}

# ROC -------------------------------
tprs = []
aucs = []
base_fpr = np.linspace(0, 1, 101)
colors = ['darksalmon', 'gold', 'royalblue', 'mediumseagreen', 'violet']

fig, ax = plt.subplots()
# -----------------------------------

# calibration curve -----------------
probs_true = []
probs_pred = []
# -----------------------------------

# learning curve --------------------
train_sizes_ncv = []
train_scores_ncv = []
test_scores_ncv = []
fit_times_ncv = []
# -----------------------------------

# models to file
g = open('/mnt/project/tmhpred3/osalvador/radiomics_lipoma_alt/thesis/results/mlpc/combined/mlpc_final_models.dat', 'wb')


start_time = time.time()
for i in series:
    cv_outer = StratifiedKFold(n_splits=K1, random_state=i, shuffle=True)
    cv_inner = StratifiedKFold(n_splits=K2, random_state=i, shuffle=True)

    f.write(f'\n\033[1m NESTED CV RUN #{i} \033[0m')
    print(f'\n\033[1m NESTED CV RUN #{i} \033[0m')
    
    # outer loop metrics
    accuracy_outer_folds  = []
    balanced_accuracy_outer_folds  = []  
    f1_outer_folds  = []
    recall_outer_folds  = []
    mcc_outer_folds  = []
    

    # OUTER LOOP
    ct_o = 0
    for train_indices, test_indices in cv_outer.split(X, y):
        ct_o = ct_o + 1
        f.write(f'\n----------- Outer loop #{ct_o} -----------')
        print(f'\n----------- Outer loop #{ct_o} -----------')
        
        criterion_params = []
        max_features_params = []
        max_depth_params = []
        min_samples_split_params = []
        min_samples_leaf_params = []
        
        # inner loop metrics
        best_score = 0.0
        best_hp = search_space[0]
        best_feature_set = []
        
        balanced_accuracy_inner_folds  = []
        
        selected_features = []
        
        # print("Train indices: ", train_indices, "\nTest indices: ", test_indices, "\n\n")
        X_train_i, X_test_i = X[train_indices], X[test_indices]
        y_train_i, y_test_i = y[train_indices], y[test_indices]
                
        ct_i = 0
        # INNER LOOP
        for train_indices_inner, test_indices_inner in cv_inner.split(X_train_i, y_train_i):
            ct_i = ct_i + 1
            f.write(f'\nInner loop #{ct_i}')
            
            # print("Train indices inner: ", train_indices_inner, "\nTest indices inner: ", test_indices_inner, "\n\n")
            X_train_j, X_test_j = X_train_i[train_indices_inner], X_train_i[test_indices_inner]
            y_train_j, y_test_j = y_train_i[train_indices_inner], y_train_i[test_indices_inner]
            
            # PCA visualisation ------------------------------------------------------------------------------------
            dimensions = pca_vis(X_train_j, features)
            # ------------------------------------------------------------------------------------------------------

            # MRMR
            X_train_j, X_test_j, selected_inner_features = mrmr(X_train_j, X_test_j, y_train_j, features, dimensions)
            
            # oversampling -------------------------------------------------------------------------------------------
            # X_train_j, y_train_j = oversampling(X_train_j, y_train_j, ct_o, ct_i)
            
            # over- and udnersampling
            over = SMOTE(sampling_strategy=0.5)
            under = RandomUnderSampler(sampling_strategy=0.6)
            steps = [('over', over), ('under', under)]
            pipeline = Pipeline(steps=steps)
            X_train_j, y_train_j = pipeline.fit_resample(X_train_j, y_train_j)
                        
            f.write(f'\n# of selected inner features: {len(selected_inner_features)}')
            selected_features.append(selected_inner_features) # list of lists to compare
            
            # not_selected_features.append(not_selected_inner_features)
            
            # --------------------------------------------------------------------------------------------------------
            
            # print("hp before tuning: ", best_hp)
            
            for item in search_space:
                model.set_params(**item)
                model.fit(X_train_j, y_train_j)
                
                y_predicted_inner_test = model.predict(X_test_j)
                
                score = balanced_accuracy_score(y_test_j, y_predicted_inner_test)
                
                if score > best_score:
                    best_score = score
                    best_hp = item
                    best_feature_set = selected_inner_features
                    
            # print("hp after tuning: ", best_hp)
            # 
            # model.set_params(**best_hp)
            # if (model.max_depth - 10) > 0:
            #     max_depth_values = [model.max_depth - 10, model.max_depth - 5, model.max_depth + 5, model.max_depth + 10] # values around the best max_depth value only
            # elif (model.max_depth - 5) > 0:
            #     max_depth_values = [model.max_depth - 5, model.max_depth + 5]
            # 
            # for value in max_depth_values:
            #     model.max_depth = value
            #     print("- current md: ", model.max_depth)
            #     model.fit(X_train_j, y_train_j)
            #     
            #     y_predicted_inner_test = model.predict(X_test_j)
            #     
            #     score = balanced_accuracy_score(y_test_j, y_predicted_inner_test)
            #     
            #     if score > best_score:
            #         best_score = score
            #         best_hp['max_depth'] = value
            #     
            # print("hp after tuning md: ", best_hp)

            f.write("\n------------------------------------ end of inner loop ------------------------------------")
            # print("\n------------------------------------ end of inner loop ------------------------------------")
               
        # feature selection ------------------------------------------------------------------------------------        
        X_train_i = pd.DataFrame(data=X_train_i, columns=features)
        X_test_i = pd.DataFrame(data=X_test_i, columns=features)
        
        cols = [col for col in X_train_i.columns if col in best_feature_set]
        
        # print(f'\n#common features: {len(set(best_features) & set(cols))}')
        
        X_train_i = X_train_i[cols]
        X_test_i = X_test_i[cols]
        
        X_train_i = pd.DataFrame.to_numpy(X_train_i)
        X_test_i = pd.DataFrame.to_numpy(X_test_i)
        # ------------------------------------------------------------------------------------------------------
        
        # set params(best item from inner)
        model.set_params(**best_hp)
        print("\nHP: ", best_hp)
        
        hyperparameters.append(best_hp)
        hp_features.append(best_feature_set)
        
        model.fit(X_train_i, y_train_i)
        
        pickle.dump(model, g)
       
        y_predicted_outer_test = model.predict(X_test_i)
        
        # metrics
        accuracy_outer_folds.append(accuracy_score(y_test_i, y_predicted_outer_test))
        balanced_accuracy_outer_folds.append(balanced_accuracy_score(y_test_i, y_predicted_outer_test))
        f1_outer_folds.append(f1_score(y_test_i, y_predicted_outer_test))
        recall_outer_folds.append(recall_score(y_test_i, y_predicted_outer_test))
        mcc_outer_folds.append(matthews_corrcoef(y_test_i, y_predicted_outer_test))
        
        cm_o = confusion_matrix(y_test_i, y_predicted_outer_test)
        # plotCM(cm_o, i, "outer", ct_o, 'x', '', '')
        # print("!!!!", cm_o[0,0], cm_o[0,1], cm_o[1,0], cm_o[1,1])
        
        sensitivity_o = cm_o[1,1]/(cm_o[1,1]+cm_o[1,0]) # TP/(TP+TN)
        f.write(f'\nSensitivity: {round(sensitivity_o*100, 2)}%')

        specificity_o = cm_o[0,0]/(cm_o[0,0]+cm_o[0,1]) # TN/(TN+FP)
        f.write(f'\nSpecificity: {round(specificity_o*100, 2)}%')

        sensitivity_ncv.append(sensitivity_o)
        specificity_ncv.append(specificity_o)
        
        # ROC ------------------------------------------------------------------------------------------------
        y_score = model.predict_proba(X_test_i)
        fpr, tpr, _ = roc_curve(y_test_i, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        #plt.plot(fpr, tpr, lw=1, alpha=0.6, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc), c = colors[i])
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
        # ----------------------------------------------------------------------------------------------------
        
        # calibration curve ----------------------------------------------------------------------------------
        prob_true, prob_pred = calibration_curve(y_test_i, y_score[:,1], strategy='quantile')
        probs_true.append(prob_true)
        probs_pred.append(prob_pred)
        # ----------------------------------------------------------------------------------------------------
        
        # learning curve -------------------------------------------------------------------------------------
        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            model,
            X_train_i,
            y_train_i,
            cv=cv_outer,
            n_jobs=4,
            train_sizes=np.linspace(0.1, 1.0, 5),
            return_times=True,
        )
        train_sizes_ncv.append(train_sizes)
        train_scores_ncv.append(train_scores)
        test_scores_ncv.append(test_scores)
        fit_times_ncv.append(fit_times)
        # ------------------------------------------------------------------------------------------------
        
        # check how many features were selected in all loops
        f.write(f'\n# common features: {len(set(selected_features[0]) & set(selected_features[1]) & set(selected_features[2]))}')
        
        f.write("\n------------------------------------ end of outer loop ------------------------------------")
        # print("\n------------------------------------ end of outer loop ------------------------------------")
    

    f.write(f'\nBest balanced accuracy: {best_score}')
    f.write(f'\nBest hidden_layer_sizes: {model.hidden_layer_sizes}')
    f.write(f'\nBest max activation: {model.activation}')
    f.write(f'\nBest max alpha: {model.alpha}')
    
    accuracy_ncv.append(accuracy_outer_folds)
    balanced_accuracy_ncv.append(balanced_accuracy_outer_folds)
    f1_ncv.append(f1_outer_folds)
    recall_ncv.append(recall_outer_folds)
    mcc_ncv.append(mcc_outer_folds)

    mean_accuracy['Nested CV run #' + str(i)] = [str("{:.2f}".format(np.mean(np.array(accuracy_outer_folds)))) +  " +/- " + str("{:.2f}".format(np.std(np.array(accuracy_outer_folds))))]
    mean_balanced_accuracy['Nested CV run #' + str(i)] = [str("{:.2f}".format(np.mean(np.array(balanced_accuracy_outer_folds)))) +  " +/- " + str("{:.2f}".format(np.std(np.array(balanced_accuracy_outer_folds))))]
    mean_f1['Nested CV run #' + str(i)] = [str("{:.2f}".format(np.mean(np.array(f1_outer_folds)))) +  " +/- " + str("{:.2f}".format(np.std(np.array(f1_outer_folds))))]
    mean_recall['Nested CV run #' + str(i)] = [str("{:.2f}".format(np.mean(np.array(recall_outer_folds)))) +  " +/- " + str("{:.2f}".format(np.std(np.array(recall_outer_folds))))]
    mean_mcc['Nested CV run #' + str(i)] = [str("{:.2f}".format(np.mean(np.array(mcc_outer_folds)))) +  " +/- " + str("{:.2f}".format(np.std(np.array(mcc_outer_folds))))]
    

    f.write("\n---------------------------------- end of nested cv run -----------------------------------")
    # print("\n---------------------------------- end of nested cv run -----------------------------------")
    
g.close()

ff = open('/mnt/project/tmhpred3/osalvador/radiomics_lipoma_alt/thesis/results/mlpc/combined/mlpc_features.txt', 'w')
for item in hp_features:
    ff.write(str(item) + "\n")
ff.close()

gg = open('/mnt/project/tmhpred3/osalvador/radiomics_lipoma_alt/thesis/results/mlpc/combined/mlpc_hyperparams.txt', 'w')
for item in hyperparameters:
    gg.write(str(item) + "\n")
gg.close()
    
# learning curve --------------------------------------------------------------------------------------
train_sizes_ncv = np.mean(train_sizes_ncv, axis=0)
train_sizes_ncv = np.around(train_sizes_ncv)
train_sizes_ncv = train_sizes_ncv.astype(int)

train_scores_mean = []
train_scores_std = []
tr_sc1 = []
tr_sc2 = []
tr_sc3 = []
tr_sc4 = []
tr_sc5 = []

test_scores_mean = []
test_scores_std = []
t_sc1 = []
t_sc2 = []
t_sc3 = []
t_sc4 = []
t_sc5 = []

fit_times_mean = []
fit_times_std = []
f_sc1 = []
f_sc2 = []
f_sc3 = []
f_sc4 = []
f_sc5 = []

for i in range(len(train_scores_ncv)-1):
    tr_sc1.append(train_scores_ncv[i][0])
    tr_sc2.append(train_scores_ncv[i][1])
    tr_sc3.append(train_scores_ncv[i][2])
    tr_sc4.append(train_scores_ncv[i][3])
    tr_sc5.append(train_scores_ncv[i][4])
    
    t_sc1.append(test_scores_ncv[i][0])
    t_sc2.append(test_scores_ncv[i][1])
    t_sc3.append(test_scores_ncv[i][2])
    t_sc4.append(test_scores_ncv[i][3])
    t_sc5.append(test_scores_ncv[i][4])
    
    f_sc1.append(fit_times_ncv[i][0])
    f_sc2.append(fit_times_ncv[i][1])
    f_sc3.append(fit_times_ncv[i][2])
    f_sc4.append(fit_times_ncv[i][3])
    f_sc5.append(fit_times_ncv[i][4])  
    
tr_sc1 = np.ravel(tr_sc1)
tr_sc1_std = np.std(tr_sc1)
tr_sc1 = np.mean(tr_sc1) 

train_scores_mean.append(tr_sc1)
train_scores_std.append(tr_sc1_std)

tr_sc2 = np.ravel(tr_sc2)
tr_sc2_std = np.std(tr_sc2)
tr_sc2 = np.mean(tr_sc2) 

train_scores_mean.append(tr_sc2)
train_scores_std.append(tr_sc2_std)

tr_sc3 = np.ravel(tr_sc3)
tr_sc3_std = np.std(tr_sc3)
tr_sc3 = np.mean(tr_sc3) 

train_scores_mean.append(tr_sc3)
train_scores_std.append(tr_sc3_std)

tr_sc4 = np.ravel(tr_sc4)
tr_sc4_std = np.std(tr_sc4)
tr_sc4 = np.mean(tr_sc4) 

train_scores_mean.append(tr_sc4)
train_scores_std.append(tr_sc4_std)

tr_sc5 = np.ravel(tr_sc5)
tr_sc5_std = np.std(tr_sc5)
tr_sc5 = np.mean(tr_sc5) 

train_scores_mean.append(tr_sc5)
train_scores_mean = np.array(train_scores_mean)
train_scores_std.append(tr_sc5_std)
train_scores_std = np.array(train_scores_std)
    
# print("Train scores mean: ", train_scores_mean) 
# print("Train scores std: ", train_scores_std) 

t_sc1 = np.ravel(t_sc1)
t_sc1_std = np.std(t_sc1)
t_sc1 = np.mean(t_sc1) 

test_scores_mean.append(t_sc1)
test_scores_std.append(t_sc1_std)

t_sc2 = np.ravel(t_sc2)
t_sc2_std = np.std(t_sc2)
t_sc2 = np.mean(t_sc2) 

test_scores_mean.append(t_sc2)
test_scores_std.append(t_sc2_std)

t_sc3 = np.ravel(t_sc3)
t_sc3_std = np.std(t_sc3)
t_sc3 = np.mean(t_sc3) 

test_scores_mean.append(t_sc3)
test_scores_std.append(t_sc3_std)

t_sc4 = np.ravel(t_sc4)
t_sc4_std = np.std(t_sc4)
t_sc4 = np.mean(t_sc4) 

test_scores_mean.append(t_sc4)
test_scores_std.append(t_sc4_std)

t_sc5 = np.ravel(t_sc5)
t_sc5_std = np.std(t_sc5)
t_sc5 = np.mean(t_sc5) 

test_scores_mean.append(t_sc5)
test_scores_mean = np.array(test_scores_mean)
test_scores_std.append(t_sc5_std)
test_scores_std = np.array(test_scores_std)
    
# print("Test scores mean: ", test_scores_mean) 
# print("Test scores std: ", test_scores_std) 

f_sc1 = np.ravel(f_sc1)
f_sc1_std = np.std(f_sc1)
f_sc1 = np.mean(f_sc1) 

fit_times_mean.append(f_sc1)
fit_times_std.append(f_sc1_std)

f_sc2 = np.ravel(f_sc2)
f_sc2_std = np.std(f_sc2)
f_sc2 = np.mean(f_sc2) 

fit_times_mean.append(f_sc2)
fit_times_std.append(f_sc2_std)

f_sc3 = np.ravel(f_sc3)
f_sc3_std = np.std(f_sc3)
f_sc3 = np.mean(f_sc3) 

fit_times_mean.append(f_sc3)
fit_times_std.append(f_sc3_std)

f_sc4 = np.ravel(f_sc4)
f_sc4_std = np.std(f_sc4)
f_sc4 = np.mean(f_sc4) 

fit_times_mean.append(f_sc4)
fit_times_std.append(f_sc4_std)

f_sc5 = np.ravel(f_sc5)
f_sc5_std = np.std(f_sc5)
f_sc5 = np.mean(f_sc5) 

fit_times_mean.append(f_sc5)
fit_times_mean = np.array(fit_times_mean)
fit_times_std.append(f_sc5_std)
fit_times_std = np.array(fit_times_std)
    
# print("Fit times mean: ", fit_times_mean) 
# print("Fit times std: ", fit_times_std) 
    
plot_learning_curve(train_sizes_ncv, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, 
                    fit_times_mean, fit_times_std, cv=cv_outer, n_jobs=4)
plt.savefig(f'/mnt/project/tmhpred3/osalvador/radiomics_lipoma_alt/thesis/results/mlpc/combined/ncv-mlpc-learning-curve.png')
plt.close()


#------------------------------------------------------------------------------------------------------

# Calibration Curve ------------------------------------------------------------------------------------
# fig, ax = plt.subplots()
# probs_pred = np.array(probs_pred)
# probs_true = np.array(probs_true)
# probs_pred = probs_pred.mean(axis=0)
# probs_true = probs_true.mean(axis=0)
# plt.plot(probs_pred, probs_true, marker='o', linewidth=1, label='MLPC')
# 
# line = mlines.Line2D([0, 1], [0, 1], color='black', label='Perfectly Calibrated', linestyle = '--')
# transform = ax.transAxes
# line.set_transform(transform)
# ax.add_line(line)
# fig.suptitle('Calibration plot')
# ax.set_xlabel('Predicted probability')
# ax.set_ylabel('True probability in each bin')
# plt.legend()
# # plt.show()
# plt.savefig(f'/mnt/project/tmhpred3/osalvador/radiomics_lipoma_alt/thesis/results/mlpc/combined/mlpc-cal-curve.png')
# plt.close()

# ------------------------------------------------------------------------------------------------------

   
# ROC --------------------------------------------------------------------------------------------------
tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)

mean_auc = auc(base_fpr, mean_tprs)
std_auc = np.std(aucs)

tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std

plt.figure(figsize=(12, 8))
plt.plot(base_fpr, mean_tprs, 'b', alpha = 0.8, label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),)
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'blue', alpha = 0.2)
plt.plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'r', label = 'Chance', alpha= 0.8)
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc="lower right")
plt.title('Receiver operating characteristic (ROC) curve')
#plt.axes().set_aspect('equal', 'datalim')
# plt.show()
plt.savefig(f'/mnt/project/tmhpred3/osalvador/radiomics_lipoma_alt/thesis/results/mlpc/combined/mlpc-roc.png')
plt.close()
# ------------------------------------------------------------------------------------------------------



elapsed_time = (time.time() - start_time)
f.write(f'\nElapsed time: {elapsed_time} seconds \n')

f.write(f'Mean accuracy: \n {mean_accuracy}')
f.write(f'\nMean balanced accuracy:  \n {mean_balanced_accuracy}')
f.write(f'\nMean F1 score: \n {mean_f1}')
f.write(f'\nMean recall: \n {mean_recall}')
f.write(f'\nMean MCC: \n {mean_mcc}')

accuracy_ncv = np.ravel(accuracy_ncv)
balanced_accuracy_ncv = np.ravel(balanced_accuracy_ncv)
f1_ncv = np.ravel(f1_ncv)
recall_ncv = np.ravel(recall_ncv)
mcc_ncv = np.ravel(mcc_ncv)

mean_accuracy_ncv = str("{:.2f}".format(np.mean(np.array(accuracy_ncv)))) +  " +/- " + str("{:.2f}".format(np.std(np.array(accuracy_ncv))))
f.write(f'\nMean accuracy across all folds: {mean_accuracy_ncv}') # all outer folds accs averaged

mean_balanced_accuracy_ncv = str("{:.2f}".format(np.mean(np.array(balanced_accuracy_ncv)))) +  " +/- " + str("{:.2f}".format(np.std(np.array(balanced_accuracy_ncv))))
f.write(f'\nMean balanced accuracy across all folds: {mean_balanced_accuracy_ncv}') 

mean_f1_ncv = str("{:.2f}".format(np.mean(np.array(f1_ncv)))) +  " +/- " + str("{:.2f}".format(np.std(np.array(f1_ncv))))
f.write(f'\nMean F1 across all folds: {mean_f1_ncv}') 

mean_recall_ncv = str("{:.2f}".format(np.mean(np.array(recall_ncv)))) +  " +/- " + str("{:.2f}".format(np.std(np.array(recall_ncv))))
f.write(f'\nMean recall across all folds: {mean_recall_ncv}') 

mean_mcc_ncv = str("{:.2f}".format(np.mean(np.array(mcc_ncv)))) +  " +/- " + str("{:.2f}".format(np.std(np.array(mcc_ncv))))
f.write(f'\nMean MCC across all folds: {mean_mcc_ncv}') 

sensitivity_ncv = np.array(sensitivity_ncv)
specificity_ncv = np.array(specificity_ncv)
sensitivity_ncv = np.mean(sensitivity_ncv)
specificity_ncv = np.mean(specificity_ncv)

f.write(f'\nMean sensitivity all folds: {round(sensitivity_ncv*100, 2)}%') 
f.write(f'\nMean specificity all folds: {round(specificity_ncv*100, 2)}%') 


# # External Test

# In[17]:


df_ext = pd.read_csv("/mnt/project/tmhpred3/osalvador/radiomics_lipoma_alt/thesis/data/harmonized/combined_external_harmonized.csv", sep= "\t")
df_ext2 = pd.read_csv("/mnt/project/tmhpred3/osalvador/radiomics_lipoma_alt/thesis/data/clinical/clinical_data_external_combined.csv", sep=";", index_col=0)

df_ext2['ID_intern'].replace("LIP", "", regex=True, inplace=True)
df_ext2.drop(df_ext2.columns[1:6], axis=1, inplace=True)
df_ext2.drop(df_ext2.columns[2], axis=1, inplace=True)

df_ext = df_ext2.join(df_ext.set_index('ID_intern'), on='ID_intern')

df_ext = df_ext.rename({'Pathology': 'Type'}, axis=1)


# In[18]:


features_ext = list(df_ext.columns[2:])

y_ext = df_ext['Type']
y_ext = pd.DataFrame.to_numpy(y_ext) # data object: numpy array

X_ext = df_ext.drop(df_ext.columns[[0, 1]], axis=1)
X_ext = pd.DataFrame.to_numpy(X_ext) # data object: numpy array



# In[21]:


# CV to test on external dataset
K=3

cv = StratifiedKFold(n_splits=K, random_state=42, shuffle=True)

hp = []
feat = []

# ROC -------------------------------
tprs = []
aucs = []
base_fpr = np.linspace(0, 1, 101)
colors = ['darksalmon', 'gold', 'royalblue', 'mediumseagreen', 'violet']

fig, ax = plt.subplots()
# -----------------------------------


for train_indices, test_indices in cv.split(X, y): 
    x_train, x_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    accuracies = []
    
    ct = 0
    for item in hyperparameters:
        ct = ct + 1
        
        X_train = pd.DataFrame(data=x_train, columns=features)
        X_test = pd.DataFrame(data=x_test, columns=features)
        
        # print("Params: ", item)
        # print("Features: ", hp_features[ct - 1])
        
        cols = [col for col in X_train.columns if col in hp_features[ct - 1]]
        # print(f'\n#common features for {ct-1}: {len(set(hp_features[ct - 1]) & set(cols))}')
        
        X_train = X_train[cols]
        X_test = X_test[cols]
        
        X_train = pd.DataFrame.to_numpy(X_train)
        X_test = pd.DataFrame.to_numpy(X_test)
    
        model.set_params(**item)
        model.fit(X_train, y_train) 
        
        y_predicted = model.predict(X_test)
        
        accuracies.append(balanced_accuracy_score(y_test, y_predicted))
         
        hp.append(item)
        feat.append(hp_features[ct - 1])
             
best_accuracy = np.amax(accuracies)
best_hp = hp[np.argmax(accuracies)]
best_features = feat[np.argmax(accuracies)]
    
X = pd.DataFrame(data=X, columns=features)
X_ext = pd.DataFrame(data=X_ext, columns=features)

cols = [col for col in X.columns if col in best_features]

X = X[cols]
X_ext = X_ext[cols]

X = pd.DataFrame.to_numpy(X)
X_ext = pd.DataFrame.to_numpy(X_ext)

model.set_params(**best_hp)

model.fit(X, y)

y_predicted_ext = model.predict(X_ext)

# learning curve ----------------------------------------------------------------------------------------------
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            model,
            X,
            y,
            cv=cv,
            n_jobs=4,
            train_sizes=np.linspace(0.1, 1.0, 5),
            return_times=True,
        )
        
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1) 


plot_learning_curve(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, 
                    fit_times_mean, fit_times_std, cv=cv, n_jobs=4)
plt.savefig(f'/mnt/project/tmhpred3/osalvador/radiomics_lipoma_alt/thesis/results/mlpc/combined/ncv-mlpc-learning-curve-external.png')
plt.close()
# ------------------------------------------------------------------------------------------------------------


cm = confusion_matrix(y_ext, y_predicted_ext)
plotCM(cm, 'x', "external", ct_o, 'x' , 0, 0)
# print("!!!!", cm_o[0,0], cm_o[0,1], cm_o[1,0], cm_o[1,1])

f.write(f'\n\nEXTERNAL TEST METRICS')

sensitivity = cm[1,1]/(cm[1,1]+cm[1,0]) # TP/(TP+TN)
f.write(f'\nSensitivity: {round(sensitivity*100, 2)}%')
specificity = cm[0,0]/(cm[0,0]+cm[0,1]) # TN/(TN+FP)
f.write(f'\nSpecificity: {round(specificity*100, 2)}%')

    
# metrics
accuracy_external = accuracy_score(y_ext, y_predicted_ext)
balanced_accuracy_external = balanced_accuracy_score(y_ext, y_predicted_ext)
f1_external = f1_score(y_ext, y_predicted_ext)
recall_external = recall_score(y_ext, y_predicted_ext)
mcc_external = matthews_corrcoef(y_ext, y_predicted_ext)

f.write(f'\n\nAccuracy: {accuracy_external}')
f.write(f'\nBalanced accuracy: {balanced_accuracy_external}')
f.write(f'\nF1 score: {f1_external}')
f.write(f'\nRecall score: {recall_external}')
f.write(f'\nMCC: {mcc_external}')
f.close()


# In[ ]:





# In[ ]:




