"""
    Preprocess reference https://www.kaggle.com/awadhi123/text-preprocessing-using-nltk
"""
from os import truncate
from nltk import stem
# module for stop words that come with NLTK
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings
import re
import string
import transformers
import torch


def clean_text(raw_text):
    """
    Removes tweet syntactics of given sentence
    """

    # remove old style retweet text "RT"
    t = re.sub(r'^RT[\s]+', '', raw_text)

    # remove hyperlinks
    t = re.sub(r'https?:\/\/.*[\r\n]*', '', t)

    # remove hashtags
    t = re.sub(r'#', '', t)

    return t


def preprocess_sample_bert(raw_text):
    """
    returns a dict {"input_ids": [int], "token_type_ids": [int], "attention_mask": [int]}.
    The returned dictionary should be fed directly into a BERT model
    """
    cleaned = clean_text(raw_text)
    return tokenizer(cleaned, padding='max_length', max_length=40, truncation=True)


def preprocess_sample(raw_text, tokenizer, stemmer, remove_stop_words, remove_punctuation, stemming=False):
    """
    Given raw text, output tokenized list of tokens
    """
    t = clean_text(raw_text)
    tokens = tokenizer.tokenize(t)

    if remove_stop_words:
        tokens = filter(lambda t: t not in stopwords.words('English'), tokens)
    if remove_punctuation:
        tokens = filter(lambda t: t not in string.punctuation, tokens)
    if stemming:
        tokens = [stemmer.stem(t) for t in tokens]

    return list(tokens)


def preprocess_batch_avgemb(batch):
    tokenizer = TweetTokenizer(
        preserve_case=False, strip_handles=True, reduce_len=True)
    stemmer = PorterStemmer()
    return [preprocess_sample(s, tokenizer, stemmer, remove_stop_words=True, remove_punctuation=True) for s in batch]


def preprocess_batch_bert(sentences):
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(
        'bert-base-uncased')

    # sample_sentences = [preprocess_sample(s, tokenizer, stemmer, remove_stop_words=False, remove_punctuation=False) for s in sentences]
    sample_tokens = [bert_tokenizer.tokenize(clean_text(s)) for s in sentences]
    full_tokens = [['[CLS]'] + st + ['[SEP]'] for st in sample_tokens]
    sent_lengths = [len(tokens) for tokens in full_tokens]
    print(" Tokens are \n {} ".format(full_tokens[:5]))
    print("Average token length: {}".format(
        sum([len(ft) for ft in full_tokens]) / len(full_tokens)))
    max_width = min(40, max([len(ft) for ft in full_tokens]))
    print("Max token length: {}".format(max_width))

    padded_tokens = [
        ft[:max_width] + ['[PAD]' for _ in range(max_width-len(ft))] for ft in full_tokens]
    print("Padded tokens are \n {} ".format(padded_tokens[:5]))
    attn_mask = [[1 if token != '[PAD]' else 0 for token in pt]
                 for pt in padded_tokens]
    print("Attention Mask are \n {} ".format(attn_mask[:5]))

    seg_ids = [[0 for _ in range(len(pt))] for pt in padded_tokens]
    print("Segment Tokens are \n {}".format(seg_ids[:5]))

    sent_ids = [bert_tokenizer.convert_tokens_to_ids(
        pt) for pt in padded_tokens]
    print("sentence indexes \n {} ".format(sent_ids[:5]))
    vocab = set()
    for sent in sent_ids:
        for w in sent:
            vocab.add(w)
    print("Size of the vocabulary is: {}".format(len(vocab)))
    token_ids = torch.tensor(sent_ids)
    attn_mask = torch.tensor(attn_mask)
    seg_ids = torch.tensor(seg_ids)

    # print(token_ids.shape)
    # print(attn_mask.shape)
    # print(seg_ids.shape)

    return token_ids, attn_mask, seg_ids, sent_lengths
