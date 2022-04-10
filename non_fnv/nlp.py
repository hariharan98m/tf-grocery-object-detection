from pathlib import Path
from scipy.special import softmax
import re
import difflib
import json
import numpy as np
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import nltk
import difflib
from nltk.corpus import stopwords

from difflib import get_close_matches

stopwords = stopwords.words('english')

def filter_stopwords(tokens):
    filtered = [w for w in tokens if not (w in stopwords)]
    return filtered

products = {}
for x in Path('grocery-db/').rglob('*.json'):
    obj = json.load(open(str(x)))
    products[obj['product']] = obj

# vocabulary selection.
seen_vocab = set()
corpus = []
for p, obj in products.items():
    product_text = obj['text']
    toks = set(filter_stopwords(word_tokenize(product_text.lower())))
    corpus.append(' '.join(list(toks)))
    seen_vocab = seen_vocab.union(toks)

from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer(tokenizer = lambda x: x.split(' '), vocabulary= seen_vocab)
tf_idf.fit(corpus)

idfs = list(tf_idf.idf_)
vocab_ = list(tf_idf.get_feature_names())
selected_vocab = []
for idf, v in zip(idfs, vocab_):
    if idf > 1.5: selected_vocab.append(v)
    # else: print(v)
# selected_vocab

for product, obj in products.items():
    product_tokens = set(filter_stopwords(word_tokenize(obj['text'].lower())))
    product_tokens = [tok for tok in product_tokens if re.match('\W+', tok) is None and tok in selected_vocab]
    products[product]['product_tokens'] = product_tokens

product_list = list(products)

def match_vals(args):
    text_tokens, p = args

    # keyword alone matching
    keywords = products[p]['keywords']
    weights = {tok: val for tok, val in zip(keywords, np.arange(1, 1- len(keywords)/20, -0.05)) }
    keyword_scoring = 0.
    for tok in text_tokens:
        close_matches = get_close_matches(tok, keywords, cutoff = 0.7)
        if len(close_matches) != 0:
            scores = [difflib.SequenceMatcher(None, tok, close_match).ratio() for close_match in close_matches]
            closest_token = close_matches[np.argmax(scores)]
            keyword_scoring += max(scores) * weights[closest_token]

    # entire text matching
    product_tokens = products[p]['product_tokens']
    sims = set()
    text_scoring = 0
    for tok in text_tokens:
        close_matches = set(get_close_matches(tok, product_tokens, cutoff = 0.7))
        sims = sims.union(close_matches)

        if len(close_matches)>0:
            scores = []
            for close_match in close_matches:
                score = difflib.SequenceMatcher(None, tok, close_match).ratio()
                scores.append(score)
            text_scoring+= np.max(scores)

    return keyword_scoring, text_scoring

from vision_utils import get_bounds, FeatureType

def find_product(selected_boxes, documents):
    selected_boxes = selected_boxes.tolist()
    found_products = []
    variants = []
    for i, (document, selected_box) in enumerate(zip(documents, selected_boxes)):
        ymin, xmin, ymax, xmax = selected_box
        bounds, texts = get_bounds(document, FeatureType.BLOCK)
        if texts == [] or len(texts) < 2:
            selected_boxes.pop(i)
            documents.pop(i)
            continue

        # input data
        text = ''.join(texts)
        text_tokens = set(filter_stopwords(word_tokenize(text.lower())))

        # output
        final_product = None
        variant_unknown = False

        keyword_scores, text_scores = list(zip(*[match_vals((text_tokens, p)) for p in product_list]))
        softmax_text_scores = softmax(list(text_scores))
        index = np.argmax(softmax_text_scores)
        if softmax_text_scores[index] > 0.98:
            sorted_text_scores = sorted(text_scores, reverse = True)
            observed = sorted_text_scores[0]
            immediate = sorted_text_scores[1]
            if (observed - immediate)/immediate > 0.5:
                final_product = product_list[index]

        if final_product is None:
            if len(text_tokens) < 13:
                ks = sorted([x for x in keyword_scores if x!=0], reverse = True)
                if len(ks) > 0:
                    observed = ks[0]
                    if observed > 1:
                        if len(ks) > 1:
                            immediate = ks[1]
                            if (observed - immediate)/immediate > 0.5:
                                final_product = product_list[keyword_scores.index(observed)]
                            else:
                                if immediate.split('_', 1)[0] == observed.split('_', 1)[0]:
                                    final_product = immediate.split('_', 1)[0]
                                    variant_unknown = True
                        else:
                            final_product = product_list[keyword_scores.index(observed)]

        print('-'*100)
        print(selected_box)
        print('OCR detected texts:', text_tokens)
        print('-'*100)
        print('Keyword scores:' , keyword_scores)
        print('Text scores:', text_scores)
        print('Softmax Text scores:', softmax_text_scores)
        print()
        print('Final product:', final_product)
        print('\n\n')
        found_products.append(final_product)
        variants.append(variant_unknown)
    selected_boxes = np.array(selected_boxes)
    return selected_boxes, found_products, variants