import argparse
import spacy
import pandas as pd
import numpy as np
from features.feature_list import *
from pipeline.feature_annotation import (
    add_features_without_reference,
    add_features_with_reference,

)
from pipeline.postprocessing import save_processed
from pipeline.preprocessing import (
    map_labels_to_articles,
    label_txt_to_pandas,
    cleanup_labels,
    get_unique_article_categories,
    get_ctr
)
from pipeline.images_features import *
from pipeline.load_images import *
from pipeline.bayes_classifier import *
from pipeline.regression_model import *
from pipeline.bayes_regression_model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the NewsImages pipeline")
    parser.add_argument("--processed", dest="processed", action="store_true")
    parser.add_argument("--no-processed", dest="processed", action="store_false")
    args = parser.parse_args()
    use_processed = args.processed
    print(f"Using processed files: {args.processed}")

    feature_toggle_predict = False
    do_classification = False

    if not use_processed:
        print("Importing data for training and final evaluation dataset")
        # Article imports
        jan19_articles = pd.read_csv(
            r"data/content2019-01-v3.tsv", sep="\t", header=0
        )
        feb19_articles = pd.read_csv(
            r"data/content2019-02-v3.tsv", sep="\t", header=0
        )
        mar19_articles = pd.read_csv(
            r"data/content2019-03-v3.tsv", sep="\t", header=0
        )
        may19_articles = pd.read_csv(
            r"data/MediaEvalNewsImagesBatch05-Task2prediction.tsv", sep="\t", header=0
        )

        # Image label imports
        image_labels_1 = label_txt_to_pandas("data/2019-01--imagelabels.txt")
        image_labels_2 = label_txt_to_pandas("data/2019-02--imagelabels.txt")
        image_labels_3 = label_txt_to_pandas("data/2019-03--imagelabels.txt")
        image_labels_5 = label_txt_to_pandas("data/2019-05--imagelabels.txt")
        image_labels = pd.concat([image_labels_1, image_labels_2, image_labels_3, image_labels_5])

        # TODO

        # Step 1: Data Cleanup
        print("Data cleanup and mapping started")
        image_labels = cleanup_labels(image_labels, 0)

        print("Labelling jan19:")
        jan19_articles_with_labels = map_labels_to_articles(
            articles=jan19_articles, labels=image_labels
        )
        print("Labelling feb19:")
        feb19_articles_with_labels = map_labels_to_articles(
            articles=feb19_articles, labels=image_labels
        )
        print("Labelling mar19:")
        mar19_articles_with_labels = map_labels_to_articles(
            articles=mar19_articles, labels=image_labels
        )

        print("Labelling MAY19:")
        may19_articles_with_labels = map_labels_to_articles(
            articles=may19_articles, labels=image_labels
        )

        # Step 2: Calculate CTR

        jan19_articles_with_labels = get_ctr(jan19_articles_with_labels)
        feb19_articles_with_labels = get_ctr(feb19_articles_with_labels)
        mar19_articles_with_labels = get_ctr(mar19_articles_with_labels)

        # Step 3: Feature annotation
        print("Started annotating features to articles")
        print("January: add_features_without_reference")
        jan19_articles_with_labels_features = add_features_without_reference(
            jan19_articles_with_labels
        )
        print("February: add_features_without_reference")
        feb19_articles_with_labels_features = add_features_without_reference(
            feb19_articles_with_labels
        )

        print("March: add_features_without_reference")
        mar19_articles_with_labels_features = add_features_without_reference(
            mar19_articles_with_labels
        )

        print("May: add_features_without_reference")
        may19_articles_with_labels_features = add_features_without_reference(
            may19_articles_with_labels
        )

        # Step 4: Get unique article categories

        complete_category_list = get_unique_article_categories(
            "category",
            jan19_articles_with_labels_features,
            feb19_articles_with_labels_features,
            mar19_articles_with_labels_features,
            may19_articles_with_labels_features
        )

        complete_subcategory_1_list = get_unique_article_categories(
            "subcategory_1",
            jan19_articles_with_labels_features,
            feb19_articles_with_labels_features,
            mar19_articles_with_labels_features,
            may19_articles_with_labels_features
        )

        complete_subcategory_2_list = get_unique_article_categories(
            "subcategory_2",
            jan19_articles_with_labels_features,
            feb19_articles_with_labels_features,
            mar19_articles_with_labels_features,
            may19_articles_with_labels_features
        )

        # Step 5: combine training data

        # create dataset with all articles
        complete_data_no_reference = pd.concat([jan19_articles_with_labels_features,
                                                feb19_articles_with_labels_features,
                                                mar19_articles_with_labels_features
                                                ])

        complete_data_no_reference = complete_data_no_reference.reset_index(drop=True)

        print("January: add_features_with_reference")
        jan19_articles_with_labels_features = add_features_with_reference(
            jan19_articles_with_labels,
            complete_data_no_reference,
            complete_category_list,
            complete_subcategory_1_list,
            complete_subcategory_2_list
        )
        print("February: add_features_with_reference")
        feb19_articles_with_labels_features = add_features_with_reference(
            feb19_articles_with_labels,
            complete_data_no_reference,
            complete_category_list,
            complete_subcategory_1_list,
            complete_subcategory_2_list
        )
        print("March: add_features_with_reference")
        mar19_articles_with_labels_features = add_features_with_reference(
            mar19_articles_with_labels,
            complete_data_no_reference,
            complete_category_list,
            complete_subcategory_1_list,
            complete_subcategory_2_list
        )

        print("MAY: add_features_with_reference")
        may19_articles_with_labels_features = add_features_with_reference(
            may19_articles_with_labels,
            complete_data_no_reference,
            complete_category_list,
            complete_subcategory_1_list,
            complete_subcategory_2_list
        )

        save_processed(df=jan19_articles_with_labels_features, filename=JAN_FEATURES)
        save_processed(df=feb19_articles_with_labels_features, filename=FEB_FEATURES)
        save_processed(df=mar19_articles_with_labels_features, filename=MAR_FEATURES)
        save_processed(df=may19_articles_with_labels_features, filename=MAY_FEATURES)


    else:
        print("Using already processed files")
        jan19_articles_with_labels_features = pd.read_csv(
            fr"processed_data/{JAN_FEATURES}.tsv", sep="\t", header=0, index_col=0
        )
        feb19_articles_with_labels_features = pd.read_csv(
            fr"processed_data/{FEB_FEATURES}.tsv", sep="\t", header=0, index_col=0
        )
        mar19_articles_with_labels_features = pd.read_csv(
            fr"processed_data/{MAR_FEATURES}.tsv", sep="\t", header=0, index_col=0
        )
        may19_articles_with_labels_features = pd.read_csv(
            fr"processed_data/{MAY_FEATURES}.tsv", sep="\t", header=0, index_col=0
        )

    # create dataset with all articles
    complete_data = pd.concat([jan19_articles_with_labels_features,
                               feb19_articles_with_labels_features,
                               mar19_articles_with_labels_features
                               ])
    complete_data = complete_data.reset_index(drop=True)
    image_labels = pd.concat([image_labels_1, image_labels_2, image_labels_3])
    # drop hashvalue
    complete_data = complete_data.drop(columns=['hashvalue'])
    # rename nImpressions
    complete_data = complete_data.rename(columns={'nImpressions': 'nReads'})

    # list of all numerical features
    feature_labels = [
        TITLE_POLARITY,
        TITLE_SUBJECTIVITY,
        HUMAN_DELTA,
        TITLE_NEGATIVE_WORD_PROPORTION,
        TITLE_EXCLAMATION_COUNT,
        TITLE_QUESTION_COUNT,
        TITLE_HYPHEN_COUNT,
        TITLE_WORD_COUNT,
        TITLE_CHARACTER_COUNT,
        TITLE_AVG_WORD_LENGTH,
        TITLE_STOP_WORD_COUNT,
        TITLE_NON_STOP_WORD_COUNT,
        TITLE_NON_STOP_WORD_RATE,
        TITLE_ENTITY_COUNT,
        TITLE_RECENT_ENTITY_COUNT_1,
        TITLE_RECENT_ENTITY_COUNT_2,
        TITLE_SIMILARITY_TO_TOP_1,
        TITLE_SIMILARITY_TO_TOP_2,
        TITLE_SIMILARITY_TO_TOP_3,
        TITLE_SIMILARITY_TO_TOP_4,
        TITLE_SIMILARITY_TO_TOP_5,
        # TITLE_CTR_OF_MOST_SIMILAR,
        "category_enc",
        "subcategory_1_enc",
        "subcategory_2_enc"
    ]

    # load images training data
    allImgLoad = LoadImg(complete_data, r"data/imgs/")
    completeData = allImgLoad.data_filter('imgFile')
    allImgList_rgb, iidList = allImgLoad.load_image(color_mode="RGB", norm=0)
    zeroCtrList = allImgLoad.zeroctr_label('iid')

    allImgFeature = ImgFeature(allImgList_rgb, None, None)
    featureList_resnet = allImgFeature.NN_feature("ResNet50")

    # concatenate images and text features
    resnet_feature = np.array(feature_reshape(featureList_resnet))
    all_text_feature = completeData[feature_labels].values
    all_feature = np.hstack((all_text_feature, resnet_feature))

    # normalization
    completeData_new = (completeData.copy()).reset_index(drop=True)
    # Z score norm
    completeData_new[["normed_nReads", "normed_ctr"]] = completeData[['nReads', 'ctr']].apply(
        lambda x: ((x - np.mean(x)) / (np.std(x))))
    completeData_new["binary_ctr"] = pd.Series(zeroCtrList).values

    # data filter
    X_reg = pd.DataFrame(all_feature)
    y_reg = completeData_new[["iid", "normed_nReads",
                              "normed_ctr",
                              'ctr',
                              'nReads', 'nRecs',
                              'binary_ctr']]
    # filter out nRecs < 10
    X_reg, y_reg = features_data_filter(X_reg, y_reg, 1, 0, 0)
    X_reg = np.array(X_reg)
    print('training samples for regression :', len(X_reg), '\n'
                                                           'nonzero ctr samples :',
          y_reg['binary_ctr'].tolist().count(1), '\n'
                                                 'zero ctr samples :', y_reg['binary_ctr'].tolist().count(0))

    # get may data
    may_data = may19_articles_with_labels_features.drop(
        columns=['hashvalue'])
    may_data = may_data.drop(
        columns=['nImpressions'])
    may_data = may_data.drop(
        columns=['nRecs'])
    may_data = may_data.drop(
        columns=['nClicks'])
    may_data = may_data.drop(
        columns=['text'])

    mayImgLoad = LoadImg(may_data, r"data/imgs/")
    mayData = mayImgLoad.data_filter('imgFile')
    mayImgList_rgb, mayiidList = mayImgLoad.load_image(color_mode="RGB", norm=0)

    mayImgFeature = ImgFeature(mayImgList_rgb, None, None)
    may_featureList_resnet = mayImgFeature.NN_feature("ResNet50")

    # get data from bayes classifier
    nlp = spacy.load("de_core_news_sm")
    may_resnet_feature = np.array(feature_reshape(may_featureList_resnet))
    may_text_feature = mayData[feature_labels].values
    may_feature = np.hstack((may_text_feature, may_resnet_feature))

    # label
    labelList = get_labels(image_labels, iidList)
    mayLabelList = get_labels(image_labels_5, mayiidList)

    all_labels = mayLabelList + labelList
    all_labels_onehot = word2vec(all_labels)

    X_label_onehot = all_labels_onehot[len(mayLabelList):]
    X_label_onehot_may = all_labels_onehot[:len(mayLabelList)]

    # title
    data = mayData["title"].tolist() + completeData["title"].tolist()

    # lemmazation
    data_lemma = []
    for line in data:
        doc = nlp(line)
        result = ' '.join([token.lemma_ for token in nlp(line)])
        data_lemma.append(result)

    X_title_onehot_all = word2vec(data_lemma)

    X_title_onehot = X_title_onehot_all[len(mayLabelList):]
    X_title_onehot_may = X_title_onehot_all[:len(mayLabelList)]

    X_title_label_onehot_may = np.hstack((X_label_onehot_may, X_title_onehot_may))
    X_title_label_onehot = np.hstack((X_label_onehot, X_title_onehot))

    print('sample size of classifier may data', len(X_title_label_onehot_may), '\n',
          'sample size of classifier training data', len(X_title_label_onehot))

    # state may data features
    X_reg_may = may_feature
    # may data
    X_bayes_may = X_title_label_onehot_may
    # training data
    X_bayes = X_title_label_onehot
    y_bayes = completeData_new['binary_ctr'].tolist()

    # Bayes-Regression Model
    y_reg['normed_nReads'] = y_reg['normed_nReads'].fillna(0)
    pre_ctr, pre_nReads, weighted_ctr = bayes_regression_prediction(X_bayes, y_bayes, X_reg, y_reg, X_bayes_may,
                                                                    X_reg_may, 'bernoulli', 'split')

    resultdf = pd.DataFrame(pre_ctr, columns=['pre_ctr'])
    resultdf["pre_nReads"] = pre_nReads
    resultdf["weighted_ctr"] = weighted_ctr
    resultdf.to_csv('batch05_prediction_result.csv', index=False)
