# Larger-Context
codes for Larger-Context Tagging

## 1.Aggregators
* `bow_seq-aggregator`: codes of the bow and seq aggregators.
* `cPre-seq-aggregator`: codes of the cPre-seq aggregator.
* `graph-aggregator`: codes of the graph aggregator.
* `codes4analysis`: codes for the analysis of this paper.

## 2.Requirements

-  `python3`
-  `pytorch==1.2`
-  `pip3 install -r requirements.txt`

## 3.Run
Take `bow_seq-aggregator` as an example.
- Put the data in the `bow_seq-aggregator/data` and set the train, dev, and test set position by setting the `--train`, `--dev`, `--test`.
- Setting the number of sentence for the current model. 


Put the data in data/kairos with train, dev, and test



## 4.Datasets

The datasets utilized in our paper including:

### (1) Named Entity Recognition (NER)
- CoNLL-2003 (in this repository.)
- OntoNotes 5.0 (The domains we utilized in the paper: WB, MZ, BC, BN, TC, NW.) (Yor can download from [LDC](https://catalog.ldc.upenn.edu/LDC2013T19) )

### (2) Chinese Word Segmentation (CWS)
- CITYU 
- PKU
- NCC
- SXU

### (3) Chunk
- CoNLL-2000

### (4) Part-of-Speech (POS)
- Penn Treebank (PTB) III


