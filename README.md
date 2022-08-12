# Fold3D

## Usage

We've provided several scripts for pretraining BERT, GPT, CPM, T5 and Turing-NLG in [`examples`](./examples) directory.

## Data Preprocessing
The training data requires preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:
<pre>
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
</pre>

The name of the `text` field of the json can be changed by using the `--json-key` flag in [`preprocess_data.py`](./tools/preprocess_data.py) The other metadata are optional and are not used in training.

The loose json is then processed into a binary format for training. To convert the json into mmap, cached index file, or the lazy loader format use `preprocess_data.py`. Set the `--dataset-impl` flag to `mmap`, `cached`, or `lazy`, respectively (default is `mmap`). An example script to prepare data for BERT training is:
<pre>
python tools/preprocess_data.py \
       --input my-corpus.json \
       --output-prefix my-bert \
       --vocab bert-vocab.txt \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences
</pre>

The output will be two files named, in this case, `my-bert_text_sentence.bin` and `my-bert_text_sentence.idx`. The `--data-path` specified in later BERT training is the full path and new filename, but without the file extension.

For T5 use the same preprocessing as BERT, perhaps renaming it to:
<pre>
       --output-prefix my-t5 \
</pre>

Some minor modifications are required for GPT data preprocessing, namely, the addition of a merge table, an end-of-document token, removal of sentence splitting, and a change to the tokenizer type:
<pre>
python tools/preprocess_data.py \
       --input my-corpus.json \
       --output-prefix my-gpt2 \
       --vocab gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --append-eod
</pre>

Here the output files are named `my-gpt2_text_document.bin` and `my-gpt2_text_document.idx`. As before, in GPT training, use the longer name without the extension as `--data-path`.

Further command line arguments are described in the source file [`preprocess_data.py`](./tools/preprocess_data.py).
