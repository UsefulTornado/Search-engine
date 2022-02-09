# Search engine - find a quote
Information retrieval ML model that performs search by quotes of celebrities/famous writers/etc.

> ***Model was trained on two datasets that were taken from kaggle: [Quotes- 500k](https://www.kaggle.com/manann/quotes-500k?select=quotes.csv) and [Goodreads Quotes](https://www.kaggle.com/faellielupe/goodreads-quotes?select=quotes.sqlite).***

# ![image](https://user-images.githubusercontent.com/77489392/153223814-3e161f12-d07c-4ac0-820f-db813bb29fd5.png)

# Description
There are all documents (stored in the *index*) and inverted indices of quotes and titles (allow us to quickly find all documents that contain some word). Server (implemented on **flask**) receives the query, then accesses to database in order to find relevant quotes. It finds documents that contain each word of lemmatized query either in title or quote (using inverted indices and index as well), sorts them by relevance and returns to server. Then server formats received documents and creates page with quotes showing to user.

# Getting started
1. Clone repository.
2. Download datasets from kaggle (links above). I used this files with **quotes1.csv** and **quotes2.csv** filenames.
Second dataset is a **.sqlite** database. You can simply save it to csv file:

```python
import pandas as pd
import sqlite3

conn = sqlite3.connect('quotes.sqlite')

df = pd.read_sql('SELECT * from quotes', conn)
df.to_csv('quotes2.csv', index=False)
```

3. Run all cells in **preparations.ipynb** to generate auxiliary files.
4. Run **server.py**.
5. Now you can use search engine on localhost (http://localhost:1111/ by default).
