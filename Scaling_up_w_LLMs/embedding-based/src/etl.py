# Author: Yifei Ning (Couson)
# Last Update: 3/18/2022

import pymongo
import pprint
import re
import string
import json
import pandas as pd

from datetime import datetime, timedelta
from dateutil.relativedelta import *

# Data cleaning (etl.py)
# Filter valid applications
# Handle cancelled application / claims
# Format texts
# EDA (pending)
# Output: csv, tables with needed features as columns (abstracts, claims, appnum, 102labels, and more) => saved in data4/xuanyu/data_2021/full_data.csv

def generate_firstInventorToFileIndicator(data):
    try:
        out = data['firstInventorToFileIndicator'].lower()
    except:
        out = 'null'
    return out


def generate_groupArtUnitNumber(data):
    try:
        out = data['groupArtUnitNumber'].lower()
    except:
        out = 'null'
    return out


def generate_prosecutionHistory_descriptor(data):
    try:
        out = str(len(data['prosecutionHistoryData']))
    except:
        out = '0'
    return out


def generate_applicationTypeCategory(data):
    try:
        out = data['applicationTypeCategory']
    except:
        out = 'null'
    return out

def generate_termAdjustmentTotalQuantity(data):
    try:
        out = str(data['termAdjustmentTotalQuantity'])
    except:
        out = '0'
    return out

def generate_publicationKind(data):
    try:
        out = data['publications'][0]['kind']
    except:
        out = 'null'
    return out


def get_label(data):
    try:
        lab = str(data['simpleLabelFromStatus'])
    except:
        return 'null'
    return lab

def generate_relatedDocumentData(date):
    try:
        out = str(len(date['relatedDocumentData']))
    except:
        return '0'
    return out

def generate_matching_dates_feature(data):
    allowedClaimIndicator = 'null'
    claimNumberArrayDocument = '0'
    publication_date = 'null'

    try:
        publication_date = data['publications'][0]['date']
    except KeyError:
        return allowedClaimIndicator, claimNumberArrayDocument

    publication_window = publication_date + relativedelta(months=-4)

    min_diff = timedelta(days = 999999999)
    closest_date = 'null'

    for oa_rejection in data['oa_rejections']:
        submission_date = oa_rejection['submissionDate']
        date_diff = abs(publication_window - submission_date)

        if date_diff < min_diff:
            min_diff = date_diff
            closest_date = submission_date


    try:
        allowedClaimIndicator = str(oa_rejection['allowedClaimIndicator']).lower()
    except:
        allowedClaimIndicator = 'null'

    try:
        claimNumberArrayDocument = str(len(oa_rejection['claimNumberArrayDocument']))
    except:
        claimNumberArrayDocument = '0'

    return datetime.strftime(publication_date, '%Y-%m'), allowedClaimIndicator, claimNumberArrayDocument


def generate_claimLevelInfo(data):
    claims_lst = []
    claim_num_lst = []
    label_binary_lst = []
    label_101_lst = []
    label_102_lst = []
    label_103_lst = []
    label_112_lst = []

    app_num = data['applicationNumber']
    claims_arr = data['publications'][0]['fullText']['claims']

    try:
        label_101_dict = data['publications'][0]['claimLabels101']
    except:
        label_101_dict = {}

    try:
        label_102_dict = data['publications'][0]['claimLabels102']
    except:
        label_102_dict = {}

    try:
        label_103_dict = data['publications'][0]['claimLabels103']
    except:
        label_103_dict = {}

    try:
        label_112_dict = data['publications'][0]['claimLabels112']
    except:
        label_112_dict = {}


    for item in claims_arr:
        try:
            claim_txt = re.sub('\t', ' ', item['claim'].strip())
            claim_num = re.findall('^[a-zA-z]?(\d+)', claim_txt)[0]
            claim = re.sub('^[a-zA-z]?\d+\s[\.|\:|\)]\s', '', claim_txt)

            # filter cancelled claims
            if len(claim) > 20:
                try:
                    label_101 = label_101_dict[claim_num]
                except: label_101 = -1
                try:
                    label_102 = label_102_dict[claim_num]
                except: label_102 = -1
                try:
                    label_103 = label_103_dict[claim_num]
                except: label_103 = -1
                try:
                    label_112 = label_112_dict[claim_num]
                except: label_112 = -1

                claims_lst.append(claim)
                claim_num_lst.append(claim_num)
                label_101_lst.append(label_101)
                label_102_lst.append(label_102)
                label_103_lst.append(label_103)
                label_112_lst.append(label_112)

        except:
            logging(claim_txt + '\t', "/data4/xuanyu/logs/logs.etl.txt")
            continue

    claims_lst = json.dumps(claims_lst)
    claim_num_lst = json.dumps(claim_num_lst)
    label_101_lst = json.dumps(label_101_lst)
    label_102_lst = json.dumps(label_102_lst)
    label_103_lst = json.dumps(label_103_lst)
    label_112_lst = json.dumps(label_112_lst)

    return [claims_lst, claim_num_lst, label_101_lst, label_102_lst, label_103_lst, label_112_lst]


def generate_abstract(data):
    try:
        abstract = data['publications'][0]['fullText']['abstract']
        abstract = ' '.join(abstract)
        out = re.sub('\t', ' ', abstract)
        # print(abstract)
    except:
        out = 'null'

    return out


def generate_applicationNumber(data):
    try:
        out = data['applicationNumber']
    except:
        out = 'null'
    return out

def generate_patentClassification(data):

    try:
        patentClassification = data['patentClassification'][0]
        nationalClass = patentClassification['mainNationalClassification']['nationalClass']
        nationalSubclass = patentClassification['mainNationalClassification']['nationalSubclass']
    except:
        nationalClass = 'null'
        nationalSubclass = 'null'
    out = json.dumps([nationalClass, nationalSubclass])
    return out


def generate_applicantCitedExaminerReferenceIndicatorCount(data):
    out = -1
    if 'citations' in data.keys():
        out = 0
        field = data['citations']
        for item in field:
            try:
                check = item['applicantCitedExaminerReferenceIndicator']
                if isinstance(check, bool):
                    out += (check == True)
                elif isinstance(check, str):
                    out += (check == '1')
                    out += (check.lower() == 'true')
                elif isinstance(check, int):
                    out += (check == 1)
            except KeyError:
                continue
    out = str(out)
    return out


def generate_filingDate(data):
    try:
        out = datetime.strftime(data['filingDate'], '%Y-%m-%d')
    except:
        out = 'null'
    return out

# def genereate_grants_claimLevelInfo(data):
#     claims_lst = []
#
#     try:
#         claims_arr = data['fullText']['claims']
# #         label_dict = data['claimLabels']
#     except:
#         return json.dumps(claims_lst)
#
#     for item in claims_arr:
#         try:
#             cleaned_item = item['claim'].replace('\n', '').replace(',', '').strip()
#             to_append = re.findall('\d+\.(.*)', cleaned_item) # ‘1. txt’
#             claims_lst.append([to_append[0].lower(), '1'])
#
#         except:
#             continue
#     return json.dumps(claims_lst)
#
#
# def generate_grants_abstract(data):
#     try:
#         out = data['fullText']['abstract'][0].lower()
#     except:
#         out = 'null'
#     return out

def record_features(feature_dict, path, sample_data, mode = 'a', buffer_size = 100):
    NUM_FEAT = 10
    MAX_SIZE = NUM_FEAT * buffer_size

    counter = 0

    # write header first
    with open(path, 'w') as file:
        new_header = '\tpublicationDate\tallowedClaimIndicator\tclaimNumberArrayDocument\tclaims\tclaims_idx\tlabel_101\tlabel_102\tlabel_103\tlabel_112\tabstract'
        header = '\t'.join(feature_dict.keys()) + new_header +'\n'
        file.write(header)

    to_write = ''

    # while sample_data is alive
    while (sample_data.alive):
        sample = sample_data.next()
        cols =[func(sample) for func in feature_dict.values()]

        # extract features from the sample object
        other_feats = generate_matching_dates_feature(sample)
        claim_feats = generate_claimLevelInfo(sample)

        # extract abstract features
        abstract_feats = [generate_abstract(sample)]

        # concat features together
        cols += other_feats
        cols += claim_feats
        cols += abstract_feats

        # handle samples by batch of size MAX_SIZE
        if counter < MAX_SIZE:
            counter += 1
            to_write += '\t'.join(cols) + '\n'

        # write into files
        elif counter == MAX_SIZE:
            with open(path, mode) as file:
                file.write(to_write)
            to_write = ''
            counter = 0
    with open(path, mode) as file:
        file.write(to_write)

# def record_grants_features(feature_dict, path, sample_data, mode = 'a', buffer_size = 100):
#     NUM_FEAT = 10
#     MAX_SIZE = NUM_FEAT * buffer_size
#
#     counter = 0
#
#     # write header first
#     with open(path, 'w') as file:
#         header = '\t'.join(feature_dict.keys()) +'\n'
#         file.write(header)
#
#     to_write = ''
#     while (sample_data.alive):
#         sample = sample_data.next()
#         cols =[func(sample) for func in feature_dict.values()]
#         if counter < MAX_SIZE:
#             counter += 1
#             to_write += '\t'.join(cols) + '\n'
#
#         elif counter == MAX_SIZE:
#             with open(path, mode) as file:
#                 file.write(to_write)
#             to_write = ''
#             counter = 0
#     with open(path, mode) as file:
#         file.write(to_write)

def dataset_temporal_split(data_path, temporal_split):
    df = pd.read_csv(data_path, sep = '\t', low_memory =True)
    split_field = temporal_split['field']
    split_date = temporal_split['date']

    selector = pd.to_datetime(df.loc[:, split_field], format='%Y-%m')
    thres_date = datetime.strptime(split_date, '%Y-%m')
    selector = selector < thres_date

    training_set = df.loc[selector, :].reset_index(drop = True)
    testing_set = df.loc[(selector == 0), :].reset_index(drop = True)

    val_set = training_set.sample(int(len(training_set) // 8), replace = False, random_state = 24).reset_index(drop = True)
    training_set = training_set.drop(val_set.index, axis = 0).reset_index(drop = True)

    train_path = data_path.replace('.csv', '.train.csv')
    test_path = data_path.replace('.csv', '.test.csv')
    val_path = data_path.replace('.csv', '.val.csv')

    training_set.to_csv(train_path, sep = '\t')
    testing_set.to_csv(test_path, sep = '\t')
    val_set.to_csv(val_path, sep = '\t')
    # return



def logging(to_write, write_path):
    with open(write_path, 'a') as f:
        f.write(to_write)



def load_params(fp):
    with open(fp) as fh:
        param = json.load(fh)
    return param


def main():
    # assign file path
    DATABASE_PARAMS = '../config/DATABASE.json'
    ETL_PARAMS = '../config/etl.json'

    # define the treatment of each feature
    feature_dict = {'applicationNumber': generate_applicationNumber,
                'groupArtUnitNumber': generate_groupArtUnitNumber,
                'applicationTypeCategory': generate_applicationTypeCategory,
                'relatedDocumentData': generate_relatedDocumentData,
                'patentClassification': generate_patentClassification,
                'firstInventorToFileIndicator': generate_firstInventorToFileIndicator,
                'applicantCitedExaminerReferenceIndicatorCount': generate_applicantCitedExaminerReferenceIndicatorCount,
                'filingDate': generate_filingDate
               }

    # connect to database
    db_settings = load_params(DATABASE_PARAMS)
    db = pymongo.MongoClient(**db_settings)

    # cleaning dataset
    cleaning_settings = load_params(ETL_PARAMS)
    set_name = cleaning_settings['set_name']
    output_path = cleaning_settings['output_path'].replace('.csv', '.%s.csv' % set_name)
    # temporal_split = cleaning_settings['temporal_split']

    collection = db['utility_patents_full'][set_name]

    # handle collection by specified data set name
    if set_name == 'expanded':
        data = collection.aggregate(cleaning_settings[set_name])
    elif set_name == 'main':
        data = collection.find(cleaning_settings[set_name])

    # print('---> spliting the data by time threshold')
    # dataset_temporal_split(output_path, temporal_split)
    # print('---> suceess! set split by time threshold %s ' % temporal_split['date'])



if __name__ == "__main__":
    main()

    # generate a whole clean fixed df with everything, keep logging
    # generate filted data set

    # train/test split

    # resolve 罗马数字claim num 和 带字幕的 claim num；需要check label dict 来决定是否保留
