from pipeline.bayes_classifier import *
from pipeline.images_features import features_data_filter
from pipeline.regression_model import *


def bayes_regression_prediction(X_bayes, y_bayes, X_reg, y_reg, X_bayes_test, X_reg_test, bayes_model, reg_model):
    """

    Args:
        X_bayes: training data X for bayes classifier
        y_bayes: traning data y for bayes classifier
        X_reg: X for regression
        y_reg: y for regression
        X_bayes_test: X value to be predicted for bayes classifier
        X_reg_test: X value to be predicted for regression
        bayes_model: bernoulli/ gaussian/ multinomial
        reg_model: split/ merge

    Returns: prediction value, array[0]-nReads, array[1]-weighted ctr

    """
    y_class_pre, weights = classifier_prediction(X_bayes, y_bayes, X_bayes_test, bayes_model)
    output, model_pre = regression_prediction(X_reg, y_reg, X_reg_test, reg_model)
    y_pred = output[1] * weights['nonzero']
    return output[1], output[0], y_pred


def bayes_regression_evaluation(X_bayes, y_bayes, X_reg, y_reg, bayes_model, reg_model, dropout, k, n):
    """

    Args:
        dropout:
        X_bayes: training data X for bayes classifier
        y_bayes: traning data y for bayes classifier
        X_reg: X for regression
        y_reg: y for regression
        bayes_model: bernoulli/ gaussian/ multinomial
        reg_model: split/ merge
        k: f-Fold cross validation
        n: score@n

    Returns: average score

    """
    kf = KFold(n_splits=k, shuffle=False)
    scoreList = []
    lenList = []
    y_reg = y_reg.reset_index(drop=True)

    for train_index, test_index in kf.split(X_bayes):
        X_reg_train = X_reg.loc[train_index]
        y_reg_train = y_reg.loc[train_index]
        X_reg_test = X_reg.loc[test_index]
        y_reg_test = y_reg.loc[test_index]

        if dropout == True:
            X_reg_train, y_reg_train = features_data_filter(X_reg_train,
                                                            y_reg_train,
                                                            1, 0, 0)
            X_reg_train = np.array(X_reg_train)
            X_reg_test = np.array(X_reg_test)

        else:
            X_reg_train = np.array(X_reg_train)
            X_reg_test = np.array(X_reg_test)

        X_bayes_train = X_bayes[train_index]
        y_bayes_train = np.array(y_bayes)[train_index]
        X_bayes_test = X_bayes[test_index]
        y_bayes_test = np.array(y_bayes)[test_index]

        pred_ctr, pred_nreads, weighted_ctr = bayes_regression_prediction(X_bayes_train, y_bayes_train, X_reg_train, y_reg_train, X_bayes_test,
                                             X_reg_test, bayes_model, reg_model)
        score = score_evaluation(weighted_ctr, y_reg_test, n)
        scoreList.append(score)
        lenList.append(len(y_reg_test))

    avg_score = sum(np.array(lenList) * np.array(scoreList)) / sum(lenList)
    return avg_score

