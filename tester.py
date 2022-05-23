#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import pandas as pd
import glob 
import numpy

import sys
import time
import importlib
import numpy as np
from os.path import expanduser


def read_dataset(path) :
    dataset = {'train' : {'pos' : [open(f).read() for f in glob.glob('%s/train/pos/*.txt'%path)],
                          'neg' : [open(f).read() for f in glob.glob('%s/train/neg/*.txt'%path)]
                         } , 
               'test' : {'pos' : [open(f).read() for f in glob.glob('%s/test/pos/*.txt'%path)], 
                         'neg' : [open(f).read() for f in glob.glob('%s/test/neg/*.txt'%path)]
                        } ,
               '' :{'pos' : [open(f).read() for f in glob.glob('%s/pos/*.txt'%path)], 
                    'neg' : [open(f).read() for f in glob.glob('%s/neg/*.txt'%path)]
               }
              }

    full_data = None
    for category in dataset:
        dfpos = pd.DataFrame({'text' : dataset[category]['pos'][:500], 'kind' : 'pos'})
        dfneg = pd.DataFrame({'text' : dataset[category]['neg'][:500], 'kind' : 'neg'})
        df = dfpos.append(dfneg, ignore_index=True)
        if full_data is None :
            full_data = df
        else :
            full_data = full_data.append(df, ignore_index=True)
    
    return full_data





def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler) 
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    except TimeoutError as exc:
        result = default
    finally:
        signal.alarm(0)

    return result


def test_predictor(filename,df) :
    
    importTime = 0 
    predictTime = 0
    acc = 0

    try :
        
        start = time.time()
        student = importlib.import_module(filename)
        importTime = time.time() - start
        print("Import time:", importTime)
        print('Result for a negative:', student.predict_sentiment("This is a bad movie. The worst I even seen."))
        print('Result for a positive:', student.predict_sentiment("Best movie ever. Wonderful. Love it."))
        
        all_run = []
        for f in range(1) :
            start = time.time()
            for x in list(df['text'])[:1000] :
                (student.predict_sentiment(x))
            all_run.append(time.time() - start)
        predictTime =  np.average(all_run)
        print("Prediction time:", predictTime)
    
        acc = sum(df['kind'] == df['text'].apply(lambda x : student.predict_sentiment(x)))/len(df)
        print("Accuracy:", acc)
    except BaseException as e:
        print("Error:" , e)
        
    return (importTime, predictTime, acc)




if __name__ == "__main__" :    
    dataset = read_dataset (expanduser("./small/")  )
    
    try:
        res = timeout(test_predictor, args = ["sentiment_predict",dataset],  timeout_duration=17200)
        print(res)
    except BaseException as e:
        print("Exception:",e)
        raise e


