import pickle
import numpy as np
from lemmatization import lemmatize_sentence

class Document(object):
    """Class that represents document.

    In this case Document class represents quotes.

    Attributes:
        id: identifier of document.
        quote: text of quote.
        author: quote's author.
        title: the source where was the quote taken from.
        quote_lemmatized: lemmatized quote attribute.
        author_lemmatized: lemmatized author attribute.
        title_lemmatized: lemmatized title attribute.

    """

    def __init__(self, id, quote, author, title, quote_lemmatized, author_lemmatized, title_lemmatizd):
        """Inits Document class with all attributes."""
        self.id = id
        self.quote = quote
        self.author = author
        self.title = title
        self.quote_lemmatized = quote_lemmatized
        self.author_lemmatized = author_lemmatized
        self.title_lemmatized = title_lemmatizd

    
    def format(self):
        """Returns a header-text pair formatted for the request."""
        return [self.author + ', ' + self.title, self.quote]


class Storage(object):
    """Data storage class.
    
    This class designed to store documents database and
    to perform information retrieval operations such as
    searching documents by query and scoring them to
    rank while showing search engine result page.

    Attributes:
        index: list of all loaded documents.
        quotes_inv_index: inverted index, which is a dictionary that maps words
            from quotes to the IDs of the documents in which they occur.
        titles_inv_index: inverted index, which is a dictionary that maps words
            from titles to the IDs of the documents in which they occur.
        DOCS_NUMBER: number of all documents.
        AVG_WORDS_QUOTE: average number of words in lemmatized quote's text.
        AVG_WORDS_TITLE: average number of words in lemmatized title's text.

    """

    def __init__(self):
        """Inits all attributes of Storage class to empty values."""
        self.index = None
        self.quotes_inv_index = None
        self.titles_inv_index = None
        self.DOCS_NUMBER = None
        self.AVG_WORDS_QUOTE = None
        self.AVG_WORDS_TITLE = None


    def load_index(self, filename):
        """Loads index.
        
        Loads builded index with all documents from the file,
        calculates and initializes DOCS_NUMBER, AVG_WORDS_QUOTE,
        AVG_WORDS_TITLE class fields.

        Args:
            filename: name of file which contains builded index.

        Returns:
            None

        """

        with open(filename, 'rb') as f:
            self.index = pickle.load(f)

        self.DOCS_NUMBER = len(self.index)

        words_quote = words_title = 0

        for doc in self.index:
            words_quote += len(doc.quote_lemmatized.split())
            words_title += len(doc.title_lemmatized.split())

        self.AVG_WORDS_QUOTE = words_quote / self.DOCS_NUMBER
        self.AVG_WORDS_TITLE = words_title / self.DOCS_NUMBER


    def load_inverted_indices(self, quotes_inv_index_filename,
                                    titles_inv_index_filename):
        """Loads quotes and titles inverted indices.
        
        Args:
            quotes_inv_index_filename: name of file which
                contains builded quotes inverted index.
            titles_inv_index_filename: name of file which
                contains builded titles inverted index.

        Returns:
            None

        """

        with open(quotes_inv_index_filename, 'rb') as f:
            self.quotes_inv_index = pickle.load(f)

        with open(titles_inv_index_filename, 'rb') as f:
            self.titles_inv_index = pickle.load(f)


    def retrieve(self, query):
        """Retrieves documents by query.

        Lemmatizes received query and finds documents, which
        title or quote's text contains all words in lemmatized query.

        Args:
            query: query received from the user.

        Returns:
            list of Document objects.
        
        """

        query_lemmatized = lemmatize_sentence(query)
        keywords = query_lemmatized.split()

        if not keywords:
            return []

        docs_by_quotes = set(self.quotes_inv_index[keywords[0]])
        docs_by_titles = set(self.titles_inv_index[keywords[0]])

        for word in keywords[1:]:
            docs_by_quotes = docs_by_quotes.intersection(set(self.quotes_inv_index[word]))
            docs_by_titles = docs_by_titles.intersection(set(self.titles_inv_index[word]))

        return [self.index[i] for i in docs_by_titles.union(docs_by_quotes)][:300]


    def score(self, query, document):
        """Scores query-document pair.
        
        Evaluates score of document for given query
        with calculating score by each term of lemmatized query
        depending on title and quote.

        Args:
            query: query received from the user.
            document: document to evaluate score by query.

        Returns:
            a float that is similar to relevance.

        """

        doc_quote_words = document.quote_lemmatized.split()
        doc_title_words = document.title_lemmatized.split()

        k = 5
        b = 0.7

        l1 = len(doc_quote_words)
        l2 = len(doc_title_words)

        def score_by_term(term):
            tf_quote = doc_quote_words.count(term)
            tf_title = doc_title_words.count(term)

            df_quote = len(self.quotes_inv_index[term])
            df_title = len(self.titles_inv_index[term])

            quote_score = (((tf_quote*(k+1)) / (k*(1 - b + b * l1 / self.AVG_WORDS_QUOTE) + tf_quote)) *
                            np.log(self.DOCS_NUMBER / (1 + df_quote)))
                            
            title_score = (((tf_title*(k+1)) / (k*(1 - b + b * l2 / self.AVG_WORDS_TITLE) + tf_title)) *
                            np.log(self.DOCS_NUMBER / (1 + df_title)))
            
            return quote_score * 0.4 + title_score * 0.6

        query_lemmatized = lemmatize_sentence(query)

        sum = 0
        
        for term in query_lemmatized.split():
            sum += score_by_term(term)

        return sum


    def search(self, query):
        """Searches the most relevant documents for query.
        
        Retrieves documents for given query and sorts them by score method.

        Args:
            query: query received from the user.

        Returns:
            list of doc + score(query, doc) tuples.

        """

        documents = self.retrieve(query)
        
        scored = [(doc, self.score(query, doc)) for doc in documents]
        scored = sorted(scored, key=lambda doc: -doc[1])

        return scored[:30]
