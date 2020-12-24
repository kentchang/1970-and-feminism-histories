
# 1970-and-feminism-histories

Code to support "1970 and Feminism's Histories" (Post-45/CA). This code was intended to be used in HathiTrust Data Capsule. For more information, see [http://analytics.hathitrust.org/](http://analytics.hathitrust.org/).

Github: [https://github.com/kentchang/1970-and-feminism-histories.git](https://github.com/kentchang/1970-and-feminism-histories.git).


## Setup

```
pip install nltk
pip install sklearn

python setup.py metadata.csv
htrc download -c -o $PWD/CORPUS hathi_ids.txt
python setup.py metadata.csv
python sort_raw_texts.py metadata.csv
```


## Collocation stats

`collocation_stats_viewer` is a preliminary implementation of [AntConc](https://www.laurenceanthony.net/software/antconc/)'s collocation tab.

```
python collocation_stats_viewer METADATA_FILENAME CLASS_LABEL LEFT_BOUNDARY RIGHT_BOUNDARY FREQUENCY_THRESHOLD IS_AGGREGATE TERMS_FILE
```

Example:

```
python collocation_stats_viewer.py 5 5 10 True FEM interested_terms.txt
```

## NER and counts

Download [https://nlp.stanford.edu/software/CRF-NER.html#Download](https://nlp.stanford.edu/software/CRF-NER.html#Download) and store `stanford-ner.jar` and `english.all.3class.distsim.crf.ser.gz` in `LIB`.

```
python ner_counts.py metadata.csv WOMEN
```


```python

```
