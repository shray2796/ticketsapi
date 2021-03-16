from flask import Flask, render_template, url_for, redirect, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import marisa_trie


app = Flask(__name__)



# hack to store vocabulary in MARISA Trie
class _MarisaVocabularyMixin(object):

    def fit_transform(self, raw_documents, y=None):
        super(_MarisaVocabularyMixin, self).fit_transform(raw_documents)
        self._freeze_vocabulary()
        return super(_MarisaVocabularyMixin, self).fit_transform(raw_documents, y)
        
    def _freeze_vocabulary(self):
        if not self.fixed_vocabulary_:
            self.vocabulary_ = marisa_trie.Trie(self.vocabulary_.keys())
            self.fixed_vocabulary_ = True
            del self.stop_words_
            

class MarisaTfidfVectorizer(_MarisaVocabularyMixin, TfidfVectorizer):
    def fit(self, raw_documents, y=None):
        super(MarisaTfidfVectorizer, self).fit(raw_documents)
        self._freeze_vocabulary()
        return super(MarisaTfidfVectorizer, self).fit(raw_documents, y)

# load the model from disk
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

data = pickle.load(open('dataframe.sav','rb'))

id_to_category = pickle.load(open('idcategory.sav','rb'))

tfidf = MarisaTfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(data.text).toarray()

@app.route('/tickets', methods=['POST'])
def json_example():
    texts = []
    request_data = request.get_json()
    texts.append(request.form.get('ticket', ''))
    if(texts[0]==''):
        data_response = {'status':'400'}
        return jsonify(data_response)
    else:
        text_features = tfidf.transform(texts)
        predictions = loaded_model.predict(text_features)
        for text, predicted in zip(texts, predictions):
            print('"{}"'.format(text))
            print("  - Predicted as: '{}'".format(id_to_category[predicted]))
            print("")
            response = id_to_category[predicted]

        data_response = {'status':'200','category':response}

        return jsonify(data_response)




if __name__ == '__main__':
    app.run(host="localhost", debug=True)