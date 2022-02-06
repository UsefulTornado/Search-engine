import re
from tqdm import tqdm

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from joblib import Parallel, delayed


def create_pattern(data, from_df=True, cols=None):
    unclear_symbols = set()

    if from_df:
        for col in cols:
            for row in data[col]:
                unclear_symbols.update(re.findall(r'[^A-Za-z \-\+\*\.\\\/\'\]\|]', row))
    else:
        unclear_symbols.update(re.findall(r'[^A-Za-z \-\+\*\.\\\/\'\]\|]', data))

    pattern = r'['

    for s in unclear_symbols:
        pattern += s

    pattern += '\-\+\*\.\\\/\'\]\|]'

    return pattern


def get_wordnet_pos(treebank_tag):
    pos_dict = {
        'J': 'a', # wordnet.ADJ
        'V': 'v', # wordnet.VERB,
        'N': 'n', # wordnet.NOUN,
        'R': 'r', # wordnet.ADV,
    }

    for key, item in pos_dict.items():
        if treebank_tag.startswith(key):
            return item

    return 'n' # wordnet.NOUN


def lemmatize(sent, pattern, stop_words):
    lemmatizer = WordNetLemmatizer()

    merged_sent = " ǁ ".join(sent)

    tokenized_sent = re.sub(pattern, ' ', merged_sent.lower()).split()

    tokens = [word for word in tokenized_sent if word not in stop_words]

    pos_tagged = [(word, get_wordnet_pos(tag))
                 for word, tag in pos_tag(tokens)]

    lemmatized_sent = ' '.join([lemmatizer.lemmatize(word, tag)
                                for word, tag in pos_tagged])

    res = [doc.strip() for doc in lemmatized_sent.split('ǁ')]

    return res


def lemmatize_column(df, col, batch_size=1000):
    """Applies lemmatization to a column of DataFrame.

    Deletes all non-alphabetic (English) symbols and stop-words
    from given column and applies lemmatization with WordNetLemmatizer().
    To speed up calculations func separates rows in column into batches
    of batch_size length and applies lemmatization to batch
    with paralleling using multiprocessing.

    Args:
        df: DataFrame that contains passed column.
        col: Column to apply lemmatization.
        batch_size: size of batch.

    Returns:
        list of column's lemmatized rows.

    """

    pattern = create_pattern(data=df, cols=[col])
    sw_eng = stopwords.words('english')

    texts = df[col].values
    text_batches = [texts[i: i + batch_size] for i in range(0, len(texts), batch_size)]

    processed_texts = Parallel(n_jobs=-1)(delayed(lemmatize)(t, pattern, sw_eng) for t in tqdm(text_batches))

    data_lemmatized = []

    for lst in processed_texts:
        data_lemmatized.extend(lst)

    return data_lemmatized


def lemmatize_sentence(sent):
    """Applies lemmatization to a string.

    Deletes all non-alphabetic (English) symbols and stop-words
    from given sent and applies lemmatization with WordNetLemmatizer().

    Args:
        sent: a string to lemmatize.

    Returns:
        lemmatized sent string.

    """

    pattern = create_pattern(data=sent, from_df=False)
    sw_eng = stopwords.words('english')

    return lemmatize([sent], pattern, sw_eng)[0]