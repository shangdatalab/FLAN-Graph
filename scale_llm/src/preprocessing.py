# Author: Zhaoyi Hou (Joey), Yifei Ning (Couson)
# Last Update: 3/18/2022

import re
import scipy
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, RandomSampler, SequentialSampler
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import string
import numpy as np
import pandas as pd

######## DataFrame Exploding Functions #########
def parse_labels_from_string(l):
    claim_labels_lst = [int(i.strip()) for i in l[1:-1].split(',') if i.strip() != '']
    if len(set(np.unique(claim_labels_lst)) - {0, 1}) > 0:
        return
    else:
        return claim_labels_lst
    return claim_labels_lst

def parse_claims_from_string(claims):
    claims = claims[1:-1]
    c = []
    for tmp_claim in re.findall('\"[^\"]+\"', claims):
        c.append(tmp_claim.replace('"', ''))
    return c

def parse_claims_idx_from_string(claim_idx):
    claim_idx = claim_idx[1:-1]
    c = []
    for tmp_idx in re.findall('\"[^\"]+\"', claim_idx):
        c.append(int(tmp_idx.replace('"', '')))
    return c

def explode(raw_df):
    '''
    - raw data
        - File path: “/data4/xuanyu/full_data_2021/raw/”
            - Features:
                - "dataset": "train" or "val" or "test" or “train_expanded”
                - "application_number": application number (int)
    '''
    assert 'claims' in raw_df, '  => Must have [claims] column!'
    assert 'claims_idx' in raw_df, '  => Must have [claims_idx] column!'

    # Dropping non-needed columns
    if 'allowedClaimIndicator' in raw_df.columns:
        raw_df = raw_df.drop(columns = ['allowedClaimIndicator'])

    # Drop NaN
    prev_shape = raw_df.shape[0]
    raw_df = raw_df.dropna().reset_index(drop = True)
    curr_shape = raw_df.shape[0]
    print('  => Number of NaN rows dropped', prev_shape - curr_shape)

    original_label_lst = ['label_101', 'label_102', 'label_103', 'label_112']
    # Parse labels
    for label in original_label_lst:
        if label in raw_df.columns:
            raw_df['claim_' + label] = raw_df[label].apply(parse_labels_from_string)
            prev_shape = raw_df.shape[0]
            raw_df = raw_df.dropna().reset_index(drop = True)
            curr_shape = raw_df.shape[0]
            print('  => Number of invalid row dropped because of', label , ':', prev_shape - curr_shape)

    # Pase claims
    all_claims = []
    for row_idx in tqdm(range(raw_df.shape[0])):
        curr_app = raw_df.iloc[row_idx]
        curr_claims = parse_claims_from_string(curr_app.claims)
        curr_claims_idx = parse_claims_idx_from_string(curr_app.claims_idx)
        for i in range(len(curr_claims)):
            a_row = {
                'claim_input': curr_claims[i],
                'claim_idx': curr_claims_idx[i]
            }
            for col in raw_df.columns:

                # Skip original labels
                if col in original_label_lst:
                    continue

                # Handle claim labels with care
                if 'claim_label' in col:
                    a_row[col] = curr_app[col][i]
                elif col != 'claims' and col != 'claims_idx':
                    a_row[col] = curr_app[col]
            all_claims.append(a_row)
    return pd.DataFrame(all_claims).reset_index(drop = True)

def filter_by_label(df, filters):
    for filter in filters:
        df = df.query(filter)
    return df.reset_index(drop = True)

def split_dataset(raw_df, cut_offs):
    '''
        0 < x < cut_off_1: extended train;
        cut_off_1 < x < cut_off_2: train;
        cut_off_2 < x < cut_off_3: val;
        cut_off_3 < x < 0: val;
    '''
    assert 'percentile' in raw_df.columns, '  => Must have [percentile] column!'
    assert len(cut_offs) == 3, '  => Must have [4] elements in cut_offs!'
    dataset = []
    for p in raw_df['percentile'].values:
        if p < cut_offs[0]:
            dataset.append('train_ext')
        elif p < cut_offs[1]:
            dataset.append('train')
        elif p < cut_offs[2]:
            dataset.append('val')
        else:
            dataset.append('test')
    raw_df['dataset'] = dataset
    return raw_df

######## Feature Generation Functions #########
def clean_transcript(transcript):
    transcript = transcript.lower()
    transcript = re.sub(r'[^a-zA-Z0-9\s]', ' ', transcript)
    transcript = re.sub(r'\b\d+?\b', '', transcript)
    transcript = re.sub(r'\s+', ' ', transcript)
    return ' '.join(transcript.split())

# Calculate perplexity score given a sentence and a bigram model
def sentence_perplexity(sentence, model, vocab_size, word2freq):
    '''
    Perplexity score of a sentence:
        ps = ((1/p1) * (1/p2) * ... * (1/pn-1)) ^ (1/(N - 1))
        - pi: probability of i-th bigram
        - N: number of tokens in the sentence
    '''
    # sentence = clean_transcript(sentence)
    sentence = sentence.split()
    perplexity = 1

    # Check len(sentence)
    if len(sentence) == 1:
        return perplexity
    for i in range(len(sentence) - 1):
        try:
            numerator = model[sentence[i]][sentence[i + 1]] + 1
        except:
            numerator = 1
        try:
            denominator = word2freq[sentence[i]] + vocab_size
        except:
            denominator = vocab_size
        prob = numerator / denominator
        perplexity = perplexity * (1 / prob)
    perplexity = pow(perplexity, 1 / float(len(sentence) - 1))
    return perplexity

def boilerplate_feature_generator(text, cfreq_2gram, vocab_size, word2freq, threshold, all_ppl_dict):
    claim = clean_transcript(text)
    words = [w for w in word_tokenize(claim.lower().replace('\\n', '')) if w not in string.punctuation]
    if len(words) < 4:
        return [], [], -1, -1

    result = {}
    # Four-grams
    four_grams = [' '.join([words[i], words[i+1], words[i+2], words[i+3]]) for i in range(len(words)-3)]
    boilerplate_counter = 0
    ppl_lst_4 = []
    gram_lst_4 = []
    for i in range(len(four_grams)):
        gram = four_grams[i]
        if gram in all_ppl_dict.keys():
            ppl = all_ppl_dict[gram]
        else:
            ppl = sentence_perplexity(gram, cfreq_2gram, vocab_size, word2freq)
            all_ppl_dict[gram] = ppl
        ppl_lst_4.append(ppl)
        gram_lst_4.append(clean_transcript(gram))
        if ppl < threshold:
            boilerplate_counter += 1
    boilerplate_r_4 = boilerplate_counter / len(four_grams)
    ppl_dist_4 = []
    if len(ppl_lst_4) > 0:
        ppl_lst_4 = np.array(ppl_lst_4)
        ppl_dist_4 = [np.percentile(ppl_lst_4, 25), np.percentile(ppl_lst_4, 50), np.percentile(ppl_lst_4, 75), \
                      ppl_lst_4.std(), scipy.stats.iqr(ppl_lst_4), scipy.stats.skew(ppl_lst_4)]
    else:
        return [], [], -1, -1

    # Tri-grams
    tri_grams = [' '.join([words[i], words[i+1], words[i+2]]) for i in range(len(words)-2)]
    boilerplate_counter = 0
    ppl_lst_3 = []
    gram_lst_3 = []
    for i in range(len(tri_grams)):
        gram = tri_grams[i]
        if gram in all_ppl_dict.keys():
            ppl = all_ppl_dict[gram]
        else:
            ppl = sentence_perplexity(gram, cfreq_2gram, vocab_size, word2freq)
            all_ppl_dict[gram] = ppl
        ppl_lst_3.append(ppl)
        gram_lst_3.append(clean_transcript(gram))
        if ppl < threshold:
            boilerplate_counter += 1
    boilerplate_r_3 = boilerplate_counter / len(tri_grams)
    ppl_dist_3 = []
    if len(ppl_lst_3) > 0:
        ppl_lst_3 = np.array(ppl_lst_3)
        ppl_dist_3 = [np.percentile(ppl_lst_3, 25), np.percentile(ppl_lst_3, 50), np.percentile(ppl_lst_3, 75),\
                      ppl_lst_3.std(), scipy.stats.iqr(ppl_lst_3), scipy.stats.skew(ppl_lst_3)]
    else:
        return [], [], -1, -1
    return ppl_dist_4, ppl_dist_3, boilerplate_r_4, boilerplate_r_3

def add_boilerplate_feature(df_in, cfreq_2gram, vocab_size, word2freq):
    df = df_in.copy()
    df['bp_4_25'] = -1
    df['bp_4_50'] = -1
    df['bp_4_75'] = -1
    df['bp_4_std'] = -1
    df['bp_4_iqr'] = -1
    df['bp_4_ratio'] = -1
    df['bp_4_skewness'] = -1

    df['bp_3_25'] = -1
    df['bp_3_50'] = -1
    df['bp_3_75'] = -1
    df['bp_3_std'] = -1
    df['bp_3_iqr'] = -1
    df['bp_3_ratio'] = -1
    df['bp_3_skewness'] = -1

    all_ppl_dict = {}
    for row_i in tqdm(df.index):
        claim = df.loc[row_i, 'claim_input']
        threshold = 5000
        ppl_dist_4, ppl_dist_3, boilerplate_r_4, boilerplate_r_3 = \
            boilerplate_feature_generator(claim, cfreq_2gram, vocab_size, word2freq, threshold, all_ppl_dict)
        try:
            if len(ppl_dist_4) == 0:
                continue
        except:
            continue
        df.loc[row_i, 'bp_4_25'] = ppl_dist_4[0]
        df.loc[row_i, 'bp_4_50'] = ppl_dist_4[1]
        df.loc[row_i, 'bp_4_75'] = ppl_dist_4[2]
        df.loc[row_i, 'bp_4_std'] = ppl_dist_4[3]
        df.loc[row_i, 'bp_4_iqr'] = ppl_dist_4[4]
        df.loc[row_i, 'bp_4_ratio'] = boilerplate_r_4
        df.loc[row_i, 'bp_4_skewness'] = ppl_dist_4[5]

        df.loc[row_i, 'bp_3_25'] = ppl_dist_3[0]
        df.loc[row_i, 'bp_3_50'] = ppl_dist_3[1]
        df.loc[row_i, 'bp_3_75'] = ppl_dist_3[2]
        df.loc[row_i, 'bp_3_std'] = ppl_dist_3[3]
        df.loc[row_i, 'bp_3_iqr'] = ppl_dist_3[4]
        df.loc[row_i, 'bp_3_ratio'] = boilerplate_r_3
        df.loc[row_i, 'bp_3_skewness'] = ppl_dist_3[5]
    return df

def cal_lexical_diversity(text):
    tokenized = word_tokenize(text)
    out = len(tokenized) / len(set(tokenized))
    return round(out, 5)

def handle_patentClassification(pc):
    try:
        pc =  pc[1: -1].replace('"', '').split(',')[0].strip()
        if pc == 'null':
            return
    except:
        return
    return pc

######## Transformer Functions #########
def tokenize(sentences, labels, tokenizer, max_length, app_nums, claim_idx, app_features = None, use_pad_token=False):
    assert len(sentences) == len(claim_idx), '[ERROR!] Should have same length for claim texts and claim_idx!'
    assert len(sentences) == len(app_nums), '[ERROR!] Should have same length for claim texts and app_nums!'
    input_ids = []
    attention_masks = []
    if use_pad_token: 
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    for sent in tqdm(sentences):
        encoded_dict = tokenizer.encode_plus(
            sent,
            truncation = True,
            add_special_tokens = True,
            max_length = max_length,
            padding='max_length',
            return_attention_mask = True,
            return_tensors = 'pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)

    if labels.dtype != 'f8':
        labels = labels.astype('int64')
    labels = torch.tensor(labels)

    app_nums = torch.tensor(app_nums)
    claim_idx = torch.tensor(claim_idx)

    if app_features is not None:
        if app_features.dtype != 'f8':
            app_features = app_features.type(torch.FloatTensor)
        app_features = torch.tensor(app_features)
    else:
        return TensorDataset(input_ids, attention_masks, labels, app_nums, claim_idx)
    return TensorDataset(input_ids, attention_masks, labels, app_features, app_nums, claim_idx)

def prepare_dataloader(dataset, sampler, batch_size):
    dataloader = DataLoader(dataset, sampler = sampler(dataset), batch_size = batch_size)
    return dataloader
