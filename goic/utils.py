import os
import json
import gzip
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s')


def line_iterator(filename):
    if filename.endswith(".gz"):
        f = gzip.GzipFile(filename)
    else:
        f = open(filename, encoding='utf8')

    while True:
        line = f.readline()
        # Test the type of the line and encode it if neccesary...
        if type(line) is bytes:
            line = str(line, encoding='utf8')

        # If the line is empty, we are done...
        if len(line) == 0:
            break

        line = line.strip()
        # If line is empty, jump to next...
        if len(line) == 0:
            continue

        yield line

    # Close the file...
    f.close()


def tweet_iterator(filename):
    for line in line_iterator(filename):
        yield json.loads(line)


TEXT = os.environ.get("TEXT", 'text')
KLASS = os.environ.get("KLASS", 'klass')


def read_data_labels(filename, get_tweet=TEXT,
                     get_klass=KLASS, maxitems=1e100):
    data, labels = [], []
    count = 0
    for tweet in tweet_iterator(filename):
        count += 1
        try:
            x = get_tweet(tweet) if callable(get_tweet) else tweet[get_tweet]
            y = get_klass(tweet) if callable(get_klass) else tweet[get_klass]
            data.append(x)
            labels.append(str(y))
            if count == maxitems:
                break
        except KeyError as e:
            logging.warn("error at line {0}, input: {1}".format(count, tweet))
            raise e

    return data, labels


def read_data(filename, get_tweet=TEXT, maxitems=1e100):
    data = []
    count = 0
    for tweet in tweet_iterator(filename):
        count += 1
        x = get_tweet(tweet) if callable(get_tweet) else tweet[get_tweet]
        data.append(x)
        if count == maxitems:
            break

    return data
