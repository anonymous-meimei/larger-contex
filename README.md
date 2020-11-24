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
- Put the data in the `bow_seq-aggregator/data` and set the train, dev, and test set location by setting the `--train`, `--dev`, and `--test` respectively.
- Put the Non-contextualized pre-trained embeddings into the `bow_seq-aggregator/emb`, and set the location of the embeddings by setting the `--emb-fn`.
- Set the number of sentence for the current aggregator model by setting the `--value`.
- After running `./run.sh`, the best model and the result of best model achieved will be saved in the `bow_seq-aggregator/models` and `bow_seq-aggregator/results` respectively.


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


