from flask import Flask, render_template, request
from search import Storage
from time import time


app = Flask(__name__, template_folder='.')

storage = Storage()
storage.load_index('index.pickle')
storage.load_inverted_indices(quotes_inv_index_filename='quotes_inv_index.pickle',
                              titles_inv_index_filename='titles_inv_index.pickle')


@app.route('/', methods=['GET'])
def index():
    start_time = time()

    query = request.args.get('query')

    if query is None:
        query = ''

    scored_documents = storage.search(query)
    results = [doc.format()+['%.2f' % scr] for doc, scr in scored_documents]

    return render_template(
        'index.html',
        time="%.2f" % (time()-start_time),
        query=query,
        search_engine_name='Quotex',
        results=results
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1111)