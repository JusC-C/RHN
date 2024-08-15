# RHN
The code of "Knowledge Graph Completion with Relation-based Hard Negative Sampling".

There are six folders in the main directory. `Model` contains the pre-trained model, and `data` is the dataset. Please copy the `data` folder into the remaining four folders, each of which represents one of the four methods described in the paper.

**for WN18RR**

```1. bash scripts/preprocess.sh WN18RR ```

```2. OUTPUT_DIR=./checkpoint/wn18rr/ bash scripts/train_wn.sh ```

```3. bash scripts/eval.sh ./checkpoint/wn18rr/model_last.mdl WN18RR ```

**for FB15k-237**

```1. bash scripts/preprocess.sh FB15k237 ```

```2. OUTPUT_DIR=./checkpoint/fb15k237/ bash scripts/train_fb.sh ```

```3. bash scripts/eval.sh ./checkpoint/fb15k237/model_last.mdl FB15k237 ```

**for Wikidata5M**

```1. bash scripts/preprocess.sh wiki5m_trans ```

```2. OUTPUT_DIR=./checkpoint/wiki5m_trans/ bash scripts/train_wiki.sh wiki5m_trans ```

```3. bash scripts/eval_wiki5m_trans.sh ./checkpoint/wiki5m_trans/model_last.mdl ```
