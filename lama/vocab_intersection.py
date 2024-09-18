# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from lama.modules import build_model_by_name
from tqdm import tqdm
import argparse
import spacy
import lama.modules.base_connector as base


LOWERCASED_MODELS = [
#  {
#    # "BERT BASE UNCASED"
#    "lm": "bert",
#    "bert_model_name": "bert-base-uncased",
#    "bert_model_dir": None,
#    "bert_vocab_name": "vocab.txt"
#  },
   {
      "lm": "roberta",
      "roberta_model_name": "pytorch_model.bin",
      "roberta_model_dir": "./pre-trained_language_models/roberta-base/",
      "roberta_vocab_name": "vocab.json"
   }
#  {
#    # "BERT LARGE UNCASED"
#    "lm": "bert",
#    "bert_model_name": "bert-large-uncased",
#    "bert_model_dir": None,
#    "bert_vocab_name": "vocab.txt"
#  },
]

LOWERCASED_COMMON_VOCAB_FILENAME = "pre-trained_language_models/common_vocab_lowercased.txt"


def __vocab_intersection(models, filename):
    vocabularies = []

    for arg_dict in models:
        args = argparse.Namespace(**arg_dict)
        print(args)
        model = build_model_by_name(args.lm, args)
        vocabularies.append(model.vocab)
        print(len(model.vocab))
        print(type(model.vocab))

    if len(vocabularies) > 0:
        common_vocab = set(vocabularies[0])
        for vocab in vocabularies:
            common_vocab = common_vocab.intersection(set(vocab))

        # no special symbols in common_vocab
        for symbol in base.SPECIAL_SYMBOLS:
            if symbol in common_vocab:
                common_vocab.remove(symbol)

        # remove stop words
        from spacy.lang.en.stop_words import STOP_WORDS
        for stop_word in STOP_WORDS:
            if stop_word in common_vocab:
                print(stop_word)
                common_vocab.remove(stop_word)

        common_vocab = list(common_vocab)

        # remove punctuation and symbols
        nlp = spacy.load('en')
        manual_punctuation = ['(', ')', '.', ',']
        special_tokens = {"<s>", "</s>", "<unk>", "<pad>"}  # RoBERTa special tokens

        new_common_vocab = []
        for i in tqdm(range(len(common_vocab))):
            word = common_vocab[i]

            # Skip RoBERTa special tokens
            if word in special_tokens:
                print(f"Skipping special token: {word}")
                continue

            doc = nlp(word)
            token = doc[0]

            if len(doc) != 1:
                print(f"Word with len(doc) != 1: {word}")
                for idx, tok in enumerate(doc):
                    print(f"{idx} - {tok}")
            elif word in manual_punctuation:
                continue
            elif token.pos_ == "PUNCT":
                continue
            elif token.pos_ == "SYM":
                continue
            else:
                new_common_vocab.append(word)

        common_vocab = new_common_vocab

        # store common_vocab on file
        with open(filename, 'w') as f:
            for item in sorted(common_vocab):
                f.write(f"{item}\n")



def main():
    # cased version
    # __vocab_intersection(CASED_MODELS, CASED_COMMON_VOCAB_FILENAME)
    # lowercased version
    __vocab_intersection(LOWERCASED_MODELS, LOWERCASED_COMMON_VOCAB_FILENAME)


if __name__ == '__main__':
    main()
