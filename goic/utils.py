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


def item_iterator(filename):
    for line in line_iterator(filename):
        yield json.loads(line)


NAME = os.environ.get("name", 'name')
KLASS = os.environ.get("klass", 'klass')


def read_data_labels(filename, get_name=NAME,
                     get_klass=KLASS, maxitems=1e100):
    data, labels = [], []
    count = 0
    for item in item_iterator(filename):
        count += 1
        try:
            x = get_name(item) if callable(get_name) else item[get_name]
            y = get_klass(item) if callable(get_klass) else item[get_klass]
            data.append(x)
            labels.append(str(y))
            if count == maxitems:
                break
        except KeyError as e:
            logging.warn("error at line {0}, input: {1}".format(count, item))
            raise e

    return data, labels


def read_data(filename, get_name=NAME, maxitems=1e100):
    data = []
    count = 0
    for item in item_iterator(filename):
        count += 1
        x = get_name(item) if callable(get_name) else item[get_name]
        data.append(x)
        if count == maxitems:
            break

    return data
