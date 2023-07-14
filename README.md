### Chat Ion (WIP)

Chatbot written in Pytorch using transformers \
Leverages six types of training
* conversational (with conversation history)
* random word masking (same as BERT)
* sentence to sentence generation
* sentence to paragraph generation
* paragraph to sentence generation
* paragraph to paragraph generation

### Datasets Used:

 * Open Web text 2 - https://the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar
 * Cornell Movie Corpus - https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
 * en_comprehensive from this github issue - https://github.com/nltk/nltk_data/issues/62
 * nltk's words & wordnet corpuses 


### Training
 run `python3 train.py` for the default configuration which is very similar to BERT \
 to use custom hyper-parameters see `python3 train.py --help` for more details

# (WIP)
### Evaluation & Visualization

run `python3 eval.py` to test the model on user input \
run `python3 visualize.py` to run a TensorBoard session for more information
* 3D Embeddings
* Model Architecture
* Example Data