# Corey B. Holstege
# Western Governors University
# Masters of Science in Data Analytics
# D214 Performance Assessment (NKM2)
# file name: D214_NKM2_Submission_Holstege.py
# %% [1] IMPORT LIBRARIES
# import os  # https://docs.python.org/3/library/os.html
import re
import sys  # https://docs.python.org/3/library/sys.html
import pandas as pd  # https://pandas.pydata.org/docs/index.html
import numpy as np  # https://numpy.org/doc/stable/
import missingno as msno  # https://github.com/ResidentMario/missingno
import string  # for punctuation
import nltk  # natural language tookkit

from sklearn.model_selection import train_test_split  # to split the dataset into test and train
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences  # to pad the numbers with zeros
from tensorflow.keras.preprocessing.text import Tokenizer  # convert text to tokens (numbers)
from tensorflow.keras.callbacks import EarlyStopping  # stop training at a threshold
from wordcloud import WordCloud
from wordcloud import STOPWORDS

# nltk.download('stopwords')
nltk.download('punkt')

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid", palette="colorblind")  # https://seaborn.pydata.org/generated/seaborn.set_theme.html
import warnings
warnings.filterwarnings('ignore')


# %%%[1.1] CHECK VERSIONS
print('Python version: ', sys.version, '\n')
print('Pandas version: ', pd.__version__, '\n')
print('Numpy version: ', np.__version__, '\n')
print('Missingno version: ', msno.__version__, '\n')
print('Matplotlib version: ', matplotlib.__version__, '\n')
print('Seaborn version: ', sns.__version__, '\n')
print('Tensorflow version', tf.__version__, '\n')

# %% [2] GET THE DATA: IMPORT CSV FILE
# error: UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 1432-1433: invalid continuation byte
# citation: https://stackoverflow.com/a/51763708
file_encoding = 'utf8'
input_fd = open(r'C:\Users\K2Admin\OneDrive\Documents\WGUMSDA\D214\PA\McDonald_s_Reviews.csv', encoding=file_encoding, errors='backslashreplace')
df = pd.read_csv(input_fd)

# %% [3] INITIAL DATA EXAMINATION
print('Begin section: Data Set Examination \n')

# show size of the dataset
print('Number of rows/columns: ' + str(df.shape), '\n')
# notes: 33,396 rows, 10 columns

# count how many times each data type is present in the datset
print('Count of each datatype: \n', pd.value_counts(df.dtypes), '\n')
# notes: object:7; float64: 2; int64: 1

# set the display width to be larger so we can read the text
# pd.set_option('display.max_colwidth', 5000)
# pd.set_option('display.max_columns', None)
print('View first seven rows of the dataframe: \n', df.head(7), '\n')
print('View last seven rows of the dataframe: \n', df.tail(7), '\n')
# notes:

# %%%[3.1] EXAMINE COLUMNS
print('View column info: \n', df.info(), '\n')
# notes: lat and long are missing values

# %%%[3.2] VIEW STATISTICAL INFO
df_describe = df.describe(include='all')
print('View statistical info: \n', df_describe, '\n')
# notes:
# citation: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html
# print dataframe description

# %%%[3.3] CHECK FOR MISSING VALUES
missing_any = df.isnull().any()  # missing value: boolean value
missing_sum = df.isnull().sum()  # count of missing values
missing_df = pd.concat([missing_any, missing_sum], axis=1)  # concatenate the two series into a dataframe
missing_df.reset_index(inplace=True)  # reset the index
missing_df.columns = ['column_name', 'any_missing', 'total_missing']  # rename the columns
print(missing_df)

# notes: lat and long are both missing 660 values

msno.bar(df, labels=True)
# notes: visualization confirms no missing values

# %%[4] EDA: NOT REVIEW COLUMN
# Dictionary to store frequency tables
freq_tables = {}


def eda_analysis(df, column_name):
    """
    """
    # Print column name, number of unique values, and datatype
    print('-' * 75)
    print('Begin ' + column_name + ':')
    print('Number of unique values: ', df[column_name].nunique(), '\n')
    print('Datatype of the column: ', df[column_name].dtypes, '\n')

    # Print summary statistics for column
    print('Describe ' + column_name)
    print(df[column_name].describe(include=all), '\n')

    # Create frequency table
    freq_table = pd.concat([df[column_name].value_counts(dropna=False),
                            (df[column_name].value_counts(dropna=False, normalize=True) * 100).round(2)],
                           axis=1,
                           keys=['Count', 'Percentages'])
    # Add frequency table to dictionary
    freq_tables[column_name] = freq_table

    # Create countplot for column
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.countplot(y=column_name, data=df, order=df[column_name].value_counts().index)
    plt.xlabel('Count', size=14)
    plt.ylabel(column_name, size=14)
    plt.title('Count of ' + column_name, size=18)
    plt.tight_layout()
    plt.show()
    plt.close()  # Close the figure

    # Print frequency table
    print(' ')
    print('Frequency table for variable ' + column_name, ':\n', freq_table, '\n')
    print('End ' + column_name + '\n')


print(eda_analysis(df, 'store_name'))
print(eda_analysis(df, 'category'))
print(eda_analysis(df, 'store_address'))
print(eda_analysis(df, 'latitude '))
print(eda_analysis(df, 'longitude'))
print(eda_analysis(df, 'rating_count'))
print(eda_analysis(df, 'review_time'))
print(eda_analysis(df, 'rating'))

# notes:
    # store_name: only two values, one of those has randome characters at the front
    # category: only has one value
    # store_address: 40 unique; 2476 Kal has random characters for the remainder (660 values)
    # latitude: 39 unique values (but store address has 40 so there should be 40 here); 660 are NaN
    # longtitude: 39 unique values (but store address has 40 so there should be 40 here); 660 are NaN
    # rating_count: object, not an integer, 51 unique #'s. shouldn't there be a unique count per store?
    # review_time: object, not datetime, most reviews are 2-6 years ago
    # rating: object, not integer, mostreviews 1 or 5 stars

# print('Datatype of the column: ', df['latitude'].dtypes, '\n')
# print('Number of unique values: ', df['latitude'].nunique(), '\n')
# print(list(df.columns))

# %%[5] EDA: REVIEW
# %%%[5.1] WORDCLOUDS
# citation: https://medium.com/mlearning-ai/sentiment-analysis-using-lstm-21767a130857
def generate_wordcloud(data, title):
    wordcloud = WordCloud(
        stopwords=STOPWORDS,
        max_words=100,
        max_font_size=40,
        scale=4).generate(str(data))
    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.title(title)
    plt.show()


generate_wordcloud(df['review'], "All Reviews")

# generate a wordcloud for the positive sentiment and wordcloud for negative sentiment?
df_positive = df[['rating', 'review']].loc[(df['rating'] == '4 stars') | (df['rating'] == '5 stars')]
df_negative = df[['rating', 'review']].loc[(df['rating'] == '1 star') | (df['rating'] == '2 stars') | (df['rating'] == '3 stars')]

generate_wordcloud(df_positive['review'], "Positive Reviews")
generate_wordcloud(df_negative['review'], "Negative Reviews")

# %%%[5.2] CHECK FOR SPECIAL CHARACTERS
pd.set_option('display.max_colwidth', 5000)
print('View random 21 rows of the dataframe: \n', df['review'].sample(21), '\n')
# notes: upper case, lower clase, punctation,

# to-do: update this function to use the bar chart formatting code from section 4
# define a function to plot a bar chart of character counts
def character_chart(df, column_x, column_y):
    """Return a bar chart of chararacter counts."""
    plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
    plt.bar(df[column_x], df[column_y])
    #plt.barh(df[column_x], df[column_y])
    plt.xlabel('Character')
    plt.ylabel('Count')
    plt.title('Character Counts')
    plt.show()
    plt.close()


# print(character_chart.__doc__)  # print function doc string


def extract_unique_characters(df, column_name):
    """
    # Citation Dr.Ellah Festus D213 Task 2 Cohort Webinar (converted into a function)
    # function to input a dataframe and column name and output a list of unique characters found in the specified column
    """
    list_of_characters = []
    for review in df[column_name]:
        for character in review:
            if character not in list_of_characters:
                list_of_characters.append(character)
    return list_of_characters


# review list of unique characters
print(extract_unique_characters(df, 'review'))
# notes: special characters: '?', '\n'',','.','/', '*', "'",'�','\\',':','#', '"','&', '-', '!','(', ')','%', '+', '$',';', '[', '_','~', '@', '=', '<', ']', '>', '{', '}', '^'

# what am i doing with these??
special_char_1 = string.punctuation
special_char_2 = extract_unique_characters(df, 'review')

# print just special characters
print(string.ascii_letters)  # citation: https://docs.python.org/3/library/string.html


def extract_special_character_counts_df(df, column_name):
    """extract all special characters and a count of how many times the speical character appears
       store in a dataframe
    """
    special_character_counts = {}
    for review in df[column_name]:
        for character in review:
            if character not in string.ascii_letters:
                if character not in special_character_counts:
                    special_character_counts[character] = 1
                else:
                    special_character_counts[character] += 1

    # Convert the dictionary to a DataFrame
    df_special_character_counts = pd.DataFrame(list(special_character_counts.items()), columns=['character', 'count'])
    return df_special_character_counts


df_special_character_counts = extract_special_character_counts_df(df, 'review')  # store in a dataframe
df_special_character_counts = df_special_character_counts.sort_values(by='count', ascending=False)  # sort the values in the dataframe
character_chart(df_special_character_counts, 'character', 'count')  # generate the bar chart

# extract the special characters and store in a list for later
list_special_characters = df_special_character_counts['character'].tolist()
print(list_special_characters)

# notes: over 700k spaces, what does it look like when excluding spaces?
df_special_character_counts = df_special_character_counts.drop(index=0)  # drop the row for space
character_chart(df_special_character_counts, 'character', 'count')  # generate the bar chart

# notes: over 80k weird triangle/square, what does it look like when excluding this?
df_special_character_counts = df_special_character_counts.drop(index=8)  # drop the row for weird character
character_chart(df_special_character_counts, 'character', 'count')  # generate the bar chart

# %%%[5.3] EXTRACT CHARACTER COUNTS


def extract_character_counts_df(df, column_name):
    """extract all characters and a count of how many times the character appears
       store in a dataframe
    """
    character_counts = {}
    for review in df[column_name]:
        for character in review:
            if character in character_counts:
                character_counts[character] += 1
            else:
                character_counts[character] = 1

    # Convert the dictionary to a DataFrame
    df_character_counts = pd.DataFrame(list(character_counts.items()), columns=['character', 'count'])
    return df_character_counts


df_character_counts = extract_character_counts_df(df, 'review')  # store in a dataframe
df_character_counts = df_character_counts.sort_values(by='count', ascending=False)  # sort the values in the dataframe
# print(df_character_counts.sort_values(by='count', ascending=False))
print(df_character_counts)

character_chart(df_character_counts, 'character', 'count')  # generate the bar chart
# notes: the number of spaces are having an outsized effect on the chart
df_character_counts = df_character_counts.drop(index=3)  # spaces are in index row 3, drop this from the dataframe
character_chart(df_character_counts, 'character', 'count')  # replot the bar chart

# %%%[5.4] EXTRACT LETTERS


def extract_alphabet_counts_df(df, column_name):
    """extract all alphabet characters and a count of how many times the letter appears
       store in a dataframe
    """
    character_counts = {}
    for review in df[column_name]:
        for character in review:
            if character in string.ascii_letters:
                if character in character_counts:
                    character_counts[character] += 1
                else:
                    character_counts[character] = 1

    df_alphabet_counts = pd.DataFrame(list(character_counts.items()), columns=['character', 'count'])
    return df_alphabet_counts


df_alphabet_counts = extract_alphabet_counts_df(df, 'review')  # store in a dataframe
df_alphabet_counts = df_alphabet_counts.sort_values(by='count', ascending=False)  # sort the values in the dataframe
character_chart(df_alphabet_counts, 'character', 'count')  # generate the bar chart

# %%%[5.5] EXTRACT NUMBERS


def extract_numbers_counts_df(df, column_name):
    """extract all numbers and a count of how many times the number appears
       store in a dataframe
    """
    number_counts = {}
    for review in df[column_name]:
        for number in review:
            if number in string.digits:
                if number in number_counts:
                    number_counts[number] += 1
                else:
                    number_counts[number] = 1

    df_number_counts = pd.DataFrame(list(number_counts.items()), columns=['character', 'count'])
    return df_number_counts


df_number_counts = extract_numbers_counts_df(df, 'review')  # store in a dataframe
df_number_counts = df_number_counts.sort_values(by='count', ascending=False)  # sort the values in the dataframe
character_chart(df_number_counts, 'character', 'count')  # generate the bar chart

# %%%[5.6] EXTRACT STOP WORDS
print(STOPWORDS)


def extract_stopwords_counts_df(df, column_name):
    """extract all stop words and a count of how many times the stop word appears
       store in a dataframe
    """
    stop_word_count = {}
    for review in df[column_name]:
        words = review.split()
        for word in words:
            if word in STOPWORDS:
                if word in stop_word_count:
                    stop_word_count[word] += 1
                else:
                    stop_word_count[word] = 1

    df_stopword_counts = pd.DataFrame(list(stop_word_count.items()), columns=['character', 'count'])
    return df_stopword_counts


df_stopword_counts = extract_stopwords_counts_df(df, 'review')  # store in a dataframe
df_stopword_counts = df_stopword_counts.sort_values(by='count', ascending=False)  # sort the values in the dataframe
character_chart(df_stopword_counts, 'character', 'count')  # generate the bar chart. better as a barh chart
print('There are', df_stopword_counts['count'].sum(), 'stopwords.')

# %%%[5.7] EXTRACT CONTRACTIONS
# count how many contractions there are
# see how many of these contractions are in STOPWORDS


def extract_apostrophe_counts_df(df, column_name):
    apostrophe_count = {}

    for review in df[column_name]:
        words = review.split()  # Tokenize the text into words

        for word in words:
            if "'" in word:  # Check if the word contains an apostrophe
                if word in apostrophe_count:
                    apostrophe_count[word] += 1
                else:
                    apostrophe_count[word] = 1

    df_apostrophe_counts = pd.DataFrame(list(apostrophe_count.items()), columns=['word', 'count'])
    return df_apostrophe_counts


df_apostrophe_counts = extract_apostrophe_counts_df(df, 'review')  # store in a dataframe
df_apostrophe_counts = df_apostrophe_counts.sort_values(by='count', ascending=False)  # sort the values in the dataframe

# %%%[5.8] VOCAB SIZE
word_tokenizer = Tokenizer()
# review_length = []

# need to set variables outside the function for review_max, review_min, and review_median so they can be used
def vocab_size_sequence_length(df, column_name, header):
    """
    """
    word_tokenizer.fit_on_texts(df[column_name])
    vocab_size = len(word_tokenizer.word_index) + 1  # how many unique words

    review_length = []
    for char_len in df[column_name]:
        review_length.append(len(char_len.split(' ')))

    review_max = np.max(review_length)
    review_min = np.min(review_length)
    review_median = np.median(review_length)

    return print(header), print('Vocab size (unique words) is: ', vocab_size), print('Longest review has', review_max, 'words.'), print('Shortest review has', review_min, 'words.'), print('Average review has', review_median, 'words.')


print(vocab_size_sequence_length(df, 'review', 'Review dirty and not preprocessed.'))
# notes:
# vocab size (unique words): 15,114
# longest review has 584 words
# shortest review has 1 word
# average # of words per review is 11


# %%[6] SPLIT DATAFRAME DF

df_store_info = df[['store_address', 'latitude ', 'longitude', 'rating_count']].copy()
df_reviews = df[['reviewer_id', 'review_time', 'rating', 'review']].copy()

# %%[7.0] CLEAN AND PREPROCESS
# %%%[7.1] CLEAN DF_STORE_INTO
# clean up the store master data
# citation: https://www.geeksforgeeks.org/delete-duplicates-in-a-pandas-dataframe-based-on-two-columns/
df_store_info = df_store_info.drop_duplicates(subset=['store_address', 'latitude ', 'longitude', 'rating_count'], keep='first').reset_index(drop=True)
# some address are in the list twice with different rating counts - the rating count if off by a small number
df_store_info = df_store_info.drop_duplicates(subset=['store_address'], keep='first').reset_index(drop=True)

print(df_store_info.info())
# notes: rating_count column is a number with integer, lat and long at each missing 1

# Remove commas and convert to integer
df_store_info['rating_count'] = df_store_info['rating_count'].str.replace(',', '', regex=True).astype(int)

# double check
print(df_store_info.info())

# to-do:
# didnt create a linkage between the two df's for store master data

# %%%[7.2] CLEAN DF_REVIEWS
# %%%%[7.2.1] PREPROCESS COLUMN: RATING
# create a dictionary to map to reivews ratings to 0 (negative sentiment) or 1 (positive sentiment)
rating_mapping = {'1 star': 0,
                  '2 stars': 0,
                  '3 stars': 0,
                  '4 stars': 1,
                  '5 stars': 1}
df_reviews['rating'] = df_reviews['rating'].map(rating_mapping)  # update the column in df_reviews
print(eda_analysis(df_reviews, 'rating'))


# %%%%[7.2.2] PREPROCESS COLUMN: REVIEW_TIME

print(eda_analysis(df_reviews, 'review_time'))

# define a dictionary mapping for time units to dates
time_mapping = {'review_time':
                {'6 hours ago': '2023-10-01',
                    '8 hours ago': '2023-10-01',
                    '20 hours ago': '2023-10-01',
                    '21 hours ago': '2023-10-01',
                    '22 hours ago': '2023-10-01',
                    '23 hours ago': '2023-10-01',
                    'a day ago': '2023-10-01',
                    '2 days ago': '2023-10-01',
                    '3 days ago': '2023-10-01',
                    '4 days ago': '2023-10-01',
                    '5 days ago': '2023-10-01',
                    '6 days ago': '2023-10-01',
                    'a week ago': '2023-10-01',
                    '2 weeks ago': '2023-10-01',
                    '3 weeks ago': '2023-10-01',
                    '4 weeks ago': '2023-10-01',
                    'a month ago': '2023-09-01',
                    '2 months ago': '2023-08-01',
                    '3 months ago': '2023-07-01',
                    '4 months ago': '2023-06-01',
                    '5 months ago': '2023-05-01',
                    '6 months ago': '2023-04-01',
                    '7 months ago': '2023-03-01',
                    '8 months ago': '2023-02-01',
                    '9 months ago': '2023-01-01',
                    '10 months ago': '2022-12-01',
                    '11 months ago': '2022-11-01',
                    'a year ago': '2022-10-01',
                    '2 years ago': '2021-10-01',
                    '3 years ago': '2020-10-01',
                    '4 years ago': '2019-10-01',
                    '5 years ago': '2018-10-01',
                    '6 years ago': '2017-10-01',
                    '7 years ago': '2016-10-01',
                    '8 years ago': '2015-10-01',
                    '9 years ago': '2014-10-01',
                    '10 years ago': '2013-10-01',
                    '11 years ago': '2012-10-01',
                    '12 years ago': '2011-10-01'}}

df_reviews.replace(time_mapping, inplace=True)  # replace the values in the column using the dictionary defined above
df_reviews['review_time'] = pd.to_datetime(df_reviews['review_time'])  # change the datatype of the column
print(eda_analysis(df_reviews, 'review_time'))

# %%%%[7.2.3] PREPROCESS COLUMN: REVIEW


def clean_review(df, column_name, punct_list):
    """
    """

    df['review_lowercase'] = df[column_name].str.lower()  # convert text to lowercase
    df['review_no_punct'] = df['review_lowercase'].apply(lambda text: ''.join([' ' if char in punct_list else char for char in text]))  # remove punctation
    df['review_no_stopwords'] = df['review_no_punct'].apply(lambda x: ' '.join([word for word in x.split() if word not in (STOPWORDS)]))  # remove stopwords
    df['review_no_xbf_xef'] = df['review_no_stopwords'].str.replace(r'xbf|xef|xfd', '', regex=True)  # remove xbf and xef
    df['review_tokenized'] = word_tokenizer.texts_to_sequences(df['review_no_xbf_xef'])  # convert sentences to numeric counterpart

    review_length = []
    for char_len in df['review_no_xbf_xef']:
        review_length.append(len(char_len.split(' ')))

    review_max = np.max(review_length)
    review_min = np.min(review_length)
    review_median = int(np.median(review_length))

    padded_sequences_max = pad_sequences(df['review_tokenized'], padding='post', maxlen=review_max)  # add zeros to the end to set them all to the same length
    df['review_padded_max'] = pd.DataFrame({'review_padded_max': padded_sequences_max.tolist()})

    padded_sequences_median = pad_sequences(df['review_tokenized'], padding='post', maxlen=review_median)  # add zeros to the end to set them all to the same length
    df['review_padded_median'] = pd.DataFrame({'review_padded_max': padded_sequences_median.tolist()})

    df['review_len'] = df[column_name].str.len()  # original review length
    df['review_no_stopwords_len'] = df['review_no_stopwords'].str.len()  # length after stop word removal
    df['review_no_xbf_xef_len'] = df['review_no_xbf_xef'].str.len()  # length after removing xbf and xef
    df['review_tokenized_len'] = df['review_tokenized'].str.len()
    df['review_padded_max_len'] = df['review_padded_max'].str.len()
    df['review_padded_median_len'] = df['review_padded_median'].str.len()
    # df['len_diff'] = df['review_len'].sub(df['review_no_stopwords_len'])  # https://www.tutorialspoint.com/how-to-subtract-two-columns-in-pandas-dataframe

    return print('Longest review has', review_max, 'words.'), print('Shortest review has', review_min, 'words.'), print('Average review has', review_median, 'words.'), print('all done')


clean_review(df_reviews, 'review', list_special_characters)
print(df_reviews.info())


df_reviews['review_padded_max_str'] = df_reviews['review_padded_max']  # create a new column that is a copy of review_padded_max
df_reviews['review_padded_max_str'] = df_reviews['review_padded_max_str'].astype(str)  # type the new column as a str for the function


def has_all_zeros(value):
    """check to see if all values are zero
    """
    pattern = r'^\[0+(?:,\s*0+)*\]$'
    return bool(re.match(pattern, value))


print(df_reviews.shape)
df_blank_rows = df_reviews[df_reviews['review_padded_max_str'].apply(has_all_zeros)]  # copy the rows with zeros to a new dataframe
df_reviews_indicies_to_drop = df_blank_rows.index  # create a list of the index to drop (of the rows that were moved)
df_reviews = df_reviews.drop(df_reviews_indicies_to_drop)  # drop these rows from df_reviews
print(df_reviews.shape)

# notes:
# longest review is: 273 words
# shortest review is: 1 word
# average review is: 6 words

# while building the clean_review function these values were notices. these lines hlped determine if occurance was significant enough for removal
# Count the occurrences of 'xbf' or 'xef' in the 'reviews' column
count_xbf_pre = df_reviews['review'].str.count(r'xbf').sum()
count_xef_pre = df_reviews['review'].str.count(r'xef').sum()
count_xfd_pre = df_reviews['review'].str.count(r'xfd').sum()

print("Total count of 'xbf' in the 'reviews' column before cleaning:", count_xbf_pre)
print("Total count of 'xef' in the 'reviews' column before cleaning:", count_xef_pre)
print("Total count of 'xfd' in the 'reviews' column before cleaning:", count_xfd_pre)

count_xbf_post = df_reviews['review_no_xbf_xef'].str.count(r'xbf').sum()
count_xef_post = df_reviews['review_no_xbf_xef'].str.count(r'xef').sum()
count_xfd_post = df_reviews['review_no_xbf_xef'].str.count(r'xfd').sum()

print("Total count of 'xbf' in the 'reviews' column after cleaning:", count_xbf_post)
print("Total count of 'xef' in the 'reviews' column before cleaning:", count_xef_post)
print("Total count of 'xfd' in the 'reviews' column before cleaning:", count_xfd_post)

print(vocab_size_sequence_length(df_reviews, 'review_no_xbf_xef', 'Review preprocessed and clean.'))
# vocabsize = 15232

embedding_dimension = int(round(np.sqrt(np.sqrt(15232)), 0))  # 4th squart root of the vocab size
print(f'embedding dimension {embedding_dimension}')
# 11

# %%[8] TRAIN TEST SPLIT
# %%%[8.1] MAX LENGTH
# set X and y for train/test split using review padded to 273 tokens
X = df_reviews['review_padded_max']
y = df_reviews['rating']

# split the data set into train and test sets with 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=36, stratify=y)

print('Training size: ', X_train.shape, '\n')
print('Test size: ', X_test.shape, '\n')

# how can we make this a function?
print('y_test sentiment counts: ', y_test.value_counts())
y_test_counts = y_test.value_counts()
plt.bar(y_test_counts.index, y_test_counts.values)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Count of 0\'s and 1\'s in y_test')
plt.xticks(y_test_counts.index, ['0', '1'])
plt.show()
plt.close()

print('y_train sentiment counts', y_train.value_counts())
y_train_counts = y_train.value_counts()
plt.bar(y_train_counts.index, y_train_counts.values)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Count of 0\'s and 1\'s in y_train')
plt.xticks(y_train_counts.index, ['0', '1'])
plt.show()
plt.close()

# %%%[8.2] MEDIAN LENGTH
# set X and y for train/test split using review padded to 6 tokens
X_median = df_reviews['review_padded_median']
y_median = df_reviews['rating']

# split the data set into train and test sets with 80/20 split
X_train_median, X_test_median, y_train_median, y_test_median = train_test_split(X_median, y_median, test_size=0.20, random_state=36, stratify=y)

print('Training size: ', X_train_median.shape, '\n')
print('Test size: ', X_test_median.shape, '\n')

print('y_test_median sentiment counts: ', y_test_median.value_counts())
y_test_median_counts = y_test_median.value_counts()
plt.bar(y_test_median_counts.index, y_test_median_counts.values)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Count of 0\'s and 1\'s in y_test')
plt.xticks(y_test_median_counts.index, ['0', '1'])
plt.show()
plt.close()

print('y_train_median sentiment counts', y_train_median.value_counts())
y_train_median_counts = y_train_median.value_counts()
plt.bar(y_train_median_counts.index, y_train_median_counts.values)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Count of 0\'s and 1\'s in y_train')
plt.xticks(y_train_median_counts.index, ['0', '1'])
plt.show()
plt.close()

# %%[9] CREATE THE MODEL
# %%%[9.1]CREATE SOME THINGS THAT WILL BE USED
# %%%%[9.1.1] LEARNING CURVE FUNCTION
# plot training and validatoin accuracy scores
# citation: https://westerngovernorsuniversity-my.sharepoint.com/personal/william_sewell_wgu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fwilliam_sewell_wgu_edu%2FDocuments%2FDocuments%2FD213%2FWebinars%2FSentiment_Analysis_Tensorflow_2.html&parent=%2Fpersonal%2Fwilliam_sewell_wgu_edu%2FDocuments%2FDocuments%2FD213%2FWebinars&ga=1
def plot_learningCurve(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


# %%%%[9.1.2] CREATE DATAFRAME TO HOLD METRICS

df_model_metrics = pd.DataFrame({
    'model_name': [],
    'model_description': [],
    'test_loss': [],
    'test_accuracy': [],
    'epoch_stopped_at': []
    })


# %%%[9.2] DEFINE EARLY STOPPING MONITOR
# Early Stopping Monitor
early_stopping_monitor = EarlyStopping(patience=2)  # model will stop after 2 epochs of no improvement

# %%%[9.3] MODEL 1 MAX LENGTH
# citation: https://stackoverflow.com/a/77054189
X_train = np.array(X_train.tolist())
X_test = np.array(X_test.tolist())

# print(X_train.info())
# print([i for i, x in enumerate(X_train) if len(x) != 273])  # citation: https://stackoverflow.com/a/49284700, check length of arrays

model_1 = Sequential()
model_1.add(Embedding(input_dim=15232, output_dim=11, input_length=273))
model_1.add(Flatten())  # https://keras.io/api/layers/reshaping_layers/flatten/
model_1.add(Dense(100, activation='relu'))
model_1.add(Dense(50, activation='relu'))
model_1.add(Dense(2, activation='softmax'))
model_1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model_1.summary())

model_1_history = model_1.fit(X_train, y_train, epochs=20, batch_size=32, callbacks=[early_stopping_monitor], verbose=True, validation_data=(X_test, y_test))
print(model_1_history.history)
# show the plot
plot_learningCurve(model_1_history)

model_1_score = model_1.evaluate(X_train, y_train, verbose=0)
print(f'Training Set: Test Loss: {model_1_score[0]} / Test Accuracy: {model_1_score[1]}')

# evaluate model against the test dataset
model_1_evaluation = model_1.evaluate(X_test, y_test)
print(f'Test Set: Test Loss: {model_1_evaluation[0]} / Test Accuracy: {model_1_evaluation[1]}')

# add metrics to metrics dataframe
df_model_1_metrics = pd.DataFrame({
    'model_name': ['model_1'],
    'model_description': ['2 layers max length'],
    'test_loss': [model_1_evaluation[0]],
    'test_accuracy': [model_1_evaluation[1]],
    'epoch_stopped_at': ['4']
})

df_model_metrics = pd.concat([df_model_metrics, df_model_1_metrics], ignore_index=True)
print(df_model_metrics)

# %%%[9.4] MODEL 2 MEDIAN LENGTH
# citation: https://stackoverflow.com/a/77054189
X_train_median = np.array(X_train_median.tolist())
X_test_median = np.array(X_test_median.tolist())

model_2 = Sequential()
model_2.add(Embedding(input_dim=15232, output_dim=11, input_length=6))
model_2.add(Flatten())  # https://keras.io/api/layers/reshaping_layers/flatten/
model_2.add(Dense(100, activation='relu'))
model_2.add(Dense(50, activation='relu'))
model_2.add(Dense(2, activation='softmax'))
model_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model_2.summary())

model_2_history = model_2.fit(X_train_median, y_train_median, epochs=20, batch_size=32, callbacks=[early_stopping_monitor], verbose=True, validation_data=(X_test_median, y_test_median))
print(model_2_history.history)
# show the plot
plot_learningCurve(model_2_history)

model_2_score = model_2.evaluate(X_train_median, y_train_median, verbose=0)
print(f'Training Set: Test Loss: {model_2_score[0]} / Test Accuracy: {model_2_score[1]}')

# evaluate model against the test dataset
model_2_evaluation = model_2.evaluate(X_test_median, y_test_median)
print(f'Test Set: Test Loss: {model_2_evaluation[0]} / Test Accuracy: {model_2_evaluation[1]}')

# add metrics to metrics dataframe
df_model_2_metrics = pd.DataFrame({
    'model_name': ['model_2'],
    'model_description': ['2 layers median length'],
    'test_loss': [model_2_evaluation[0]],
    'test_accuracy': [model_2_evaluation[1]],
    'epoch_stopped_at': ['3']
})

df_model_metrics = pd.concat([df_model_metrics, df_model_2_metrics], ignore_index=True)
print(df_model_metrics)


# %%%[9.5] MODEL 3: MAX LENGTH & ADD LAYER
model_3 = Sequential()
model_3.add(Embedding(input_dim=15232, output_dim=11, input_length=273))
model_3.add(Flatten())  # https://keras.io/api/layers/reshaping_layers/flatten/
model_3.add(Dense(100, activation='relu'))
model_3.add(Dense(50, activation='relu'))
model_3.add(Dense(25, activation='relu'))
model_3.add(Dense(2, activation='softmax'))
model_3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model_3.summary())

model_3_history = model_3.fit(X_train, y_train, epochs=20, batch_size=32, callbacks=[early_stopping_monitor], verbose=True, validation_data=(X_test, y_test))
print(model_3_history.history)
# show the plot
plot_learningCurve(model_3_history)

model_3_score = model_3.evaluate(X_train, y_train, verbose=0)
print(f'Training Set: Test Loss: {model_3_score[0]} / Test Accuracy: {model_3_score[1]}')

# evaluate model against the test dataset
model_3_evaluation = model_3.evaluate(X_test, y_test)
print(f'Test Set: Test Loss: {model_3_evaluation[0]} / Test Accuracy: {model_3_evaluation[1]}')

# add metrics to metrics dataframe
df_model_3_metrics = pd.DataFrame({
    'model_name': ['model_3'],
    'model_description': ['3 layers max length'],
    'test_loss': [model_3_evaluation[0]],
    'test_accuracy': [model_3_evaluation[1]],
    'epoch_stopped_at': ['3']
})

df_model_metrics = pd.concat([df_model_metrics, df_model_3_metrics], ignore_index=True)
print(df_model_metrics)


# %%%[9.6] MODEL 4: MEDIAN LENGTH & ADD LAYER
model_4 = Sequential()
model_4.add(Embedding(input_dim=15232, output_dim=11, input_length=6))
model_4.add(Flatten())  # https://keras.io/api/layers/reshaping_layers/flatten/
model_4.add(Dense(100, activation='relu'))
model_4.add(Dense(50, activation='relu'))
model_4.add(Dense(2, activation='softmax'))
model_4.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model_4.summary())

model_4_history = model_4.fit(X_train_median, y_train_median, epochs=20, batch_size=32, callbacks=[early_stopping_monitor], verbose=True, validation_data=(X_test_median, y_test_median))
print(model_4_history.history)
# show the plot
plot_learningCurve(model_4_history)

model_4_score = model_4.evaluate(X_train_median, y_train_median, verbose=0)
print(f'Training Set: Test Loss: {model_4_score[0]} / Test Accuracy: {model_4_score[1]}')

# evaluate model against the test dataset
model_4_evaluation = model_4.evaluate(X_test_median, y_test_median)
print(f'Test Set: Test Loss: {model_4_evaluation[0]} / Test Accuracy: {model_4_evaluation[1]}')

# add metrics to metrics dataframe
df_model_4_metrics = pd.DataFrame({
    'model_name': ['model_4'],
    'model_description': ['3 layers median length'],
    'test_loss': [model_4_evaluation[0]],
    'test_accuracy': [model_4_evaluation[1]],
    'epoch_stopped_at': ['3']
})

df_model_metrics = pd.concat([df_model_metrics, df_model_4_metrics], ignore_index=True)
print(df_model_metrics)


# print(model_4.layers[3].get_weights()[0])  # print weights for 3rd layer
# print(model_4.layers[3].get_weights()[1])  # print biases for 3rd layer

# save model: https://datascience.stackexchange.com/questions/27343/output-trained-parameters-of-keras-model

# %%%[9.7] FIRST SIX WORDS

df_reviews['first_six_words'] = df_reviews['review_no_xbf_xef'].str.split().apply(lambda x: ' '.join(x[:6]))
generate_wordcloud(df_reviews['first_six_words'], 'Cleaned Data First Six Words')


# %%[10] EVALUATE THE MODEL ON SOCIAL MEDIA DATA

input_fd_sm = open(r'C:\Users\K2Admin\OneDrive\Documents\WGUMSDA\D214\PA\McDonalds_reviews_social_media.csv', encoding=file_encoding, errors='backslashreplace')
df_sm = pd.read_csv(input_fd_sm)

df_sm.info()

print(eda_analysis(df_sm, 'source'))
print(eda_analysis(df_sm, 'date'))
generate_wordcloud(df_sm['review'], "All Reviews")

clean_review(df_sm, 'review', list_special_characters)
# Longest review has 67 words.
# Shortest review has 2 words.
# Average review has 11 words.
generate_wordcloud(df_sm['review_no_xbf_xef'], "All Reviews")
print(df_sm.info())

df_sm_padded_max = pad_sequences(df_sm['review_tokenized'], padding='post', maxlen=273)  # need to pad to 273 as that is what the model was trained on
df_sm['review_padded_max_max'] = pd.DataFrame({'review_padded_max_max': df_sm_padded_max.tolist()})  # add the new padd back to the same df to keep everything together

# execute model 3 on df_sm and predict sentiment
df_sm_predictions = model_3.predict(df_sm_padded_max)
df_sm_predictions_analysis = pd.DataFrame(df_sm_predictions)  # convert to a dataframe


def max_column(row):
    """ function to look at value in two columns, and return which column has the greater value
    """
    if row[0] > row[1]:
        return 0
    else:
        return 1


# add column to the df_sm with the predicted rating (sentiment)
df_sm['model_3_prediction'] = df_sm_predictions_analysis.apply(max_column, axis=1)


num_index = 9
print('Origional review', df_sm['review'][num_index], '\n')
print('Predicted:', 'Negative' if df_sm_predictions[num_index][0] >= 0.5 else 'Positive', 'review')


# %%[11] CUSTOMER SATISFACTION SCORE
# This metric is calculated by taking a count of all of reviews where the company was rated a 4 or 5 (satisfied or very satisfied) divided by the total number of reviews, multiplied by 100 to turn it into a percent. The benchmark for fast food restaurants for CSAT is 76% (SurveyMonkey).

csat_satisfied_df_reviews = sum(df_reviews['rating'] == 1)
csat_satisfied_df_sm = sum(df_sm['model_3_prediction'] == 1)

csat_all_df_reviews = len(df_reviews['rating'])
csat_all_df_sm = len(df_sm['model_3_prediction'])

csat = (csat_satisfied_df_reviews + csat_satisfied_df_sm) / (csat_all_df_reviews + csat_all_df_sm) * 100

print(f'CSAT score: {round(csat,2)}%')

# %%[17] END OF SCRIPT
print(' ')
print("End of script!")
