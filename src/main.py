# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 12:01:39 2021

@author: worka
"""
import os
import spacy
from torchtext import datasets
import matplotlib.pyplot as plt
from torchtext.legacy import data

from train import *
from model import *

if __name__ == '__main__':

    import spacy
    spacy_fr = spacy.load('fr_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    def tokenize_en(sentence):
        return [tok.text for tok in spacy_en.tokenizer(sentence)]
    
    def tokenize_fr(sentence):
        return [tok.text for tok in spacy_fr.tokenizer(sentence)]
    
    EN_TEXT = data.Field(tokenize=tokenize_en)
    FR_TEXT = data.Field(tokenize=tokenize_fr, init_token = "<sos>", eos_token = "<eos>")
    
    YOUR_PATH = '../downloads'
    europarl_en = open(f"{YOUR_PATH}/europarl-v7.fr-en.en", encoding='utf-8').read().split('\n')
    europarl_fr = open(f"{YOUR_PATH}/europarl-v7.fr-en.fr", encoding='utf-8').read().split('\n')

    if os.path.exists('./data/train.csv') and os.path.exists('./data/train.csv'):
        print('exists')
    else:
        import pandas as pd
        raw_data = {'English' : [line for line in europarl_en], 'French': [line for line in europarl_fr]}
        df = pd.DataFrame(raw_data, columns=["English", "French"])
        # remove very long sentences and sentences where translations are 
        # not of roughly equal length
        df['eng_len'] = df['English'].str.count(' ')
        df['fr_len'] = df['French'].str.count(' ')
        df = df.query('fr_len < 80 & eng_len < 80')
        df = df.query('fr_len < eng_len * 1.5 & fr_len * 1.5 > eng_len')
        from sklearn.model_selection import train_test_split
        # create train and validation set 
        train, val = train_test_split(df, test_size=0.1)
        train.to_csv("./data/train.csv", index=False)
        val.to_csv("./data/val.csv", index=False)
    
    # associate the text in the 'English' column with the EN_TEXT field, # and 'French' with FR_TEXT
    print('creating dataset')
    data_fields = [('src', EN_TEXT), ('trg', FR_TEXT)]
    train,val = data.TabularDataset.splits(path='./data', train='train_.csv', 
                                           validation='val_.csv', format='csv', fields=data_fields)

    FR_TEXT.build_vocab(train, val)
    EN_TEXT.build_vocab(train, val)
    print(EN_TEXT.vocab.stoi['the'])
    print(EN_TEXT.vocab.itos[11])
    
    train_iter = data.BucketIterator(train, batch_size=20, 
                                     sort_key=lambda x: len(x.French), shuffle=True)
    
    pad_idx = FR_TEXT.vocab.stoi["<blank>"]
    model = make_model(len(EN_TEXT.vocab), len(FR_TEXT.vocab), N=6)
    model
    criterion = LabelSmoothing(size=len(FR_TEXT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion
    BATCH_SIZE = 12000
    train_iter = MyIterator(train, batch_size=BATCH_SIZE,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    model_par = model#nn.DataParallel(model)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), 
                  model_par, 
                  SimpleLossCompute(model.generator, criterion, model_opt))
                  #MultiGPULossCompute(model.generator, criterion, 
                   #                   devices=devices, opt=model_opt))
        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
                          model_par, 
                         SimpleLossCompute(model.generator, criterion, model_opt))
#                           MultiGPULossCompute(model.generator, criterion, 
#                           devices=devices, opt=None))
        print(loss)