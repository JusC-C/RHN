# RHN
The code of "Knowledge Graph Completion with Relation-based Hard Negative Sampling".

In the main directory, there are four folders, each corresponding to one of the four methods mentioned in the paper. [From this link](https://www.alipan.com/s/t129eDRStse), download the `Model` folder and the `data` folder. Place the `Model` folder in the main directory, and copy the `data` folder into each of the four method folders. 

**for WN18RR**

```1. bash scripts/preprocess.sh WN18RR ```

```2. OUTPUT_DIR=./checkpoint/wn18rr/ bash scripts/train_wn.sh ```

```3. bash scripts/eval.sh ./checkpoint/wn18rr/model_last.mdl WN18RR ```

**for FB15k-237**

```1. bash scripts/preprocess.sh FB15k237 ```

```2. OUTPUT_DIR=./checkpoint/fb15k237/ bash scripts/train_fb.sh ```

```3. bash scripts/eval.sh ./checkpoint/fb15k237/model_last.mdl FB15k237 ```

**for Wikidata5M**

```0. bash ./scripts/download_wikidata5m.sh```

Move the files from `data\shard` to `data\wiki5m_trans`.

```1. bash scripts/preprocess.sh wiki5m_trans ```

```2. OUTPUT_DIR=./checkpoint/wiki5m_trans/ bash scripts/train_wiki.sh wiki5m_trans ```

```3. bash scripts/eval_wiki5m_trans.sh ./checkpoint/wiki5m_trans/model_last.mdl ```
