## IWAN

Code for IEEE Multimedia Paper **"Modeling Incongruity between Modalities for Multimodal Sarcasm Detection."**

### Clone
Clone this repo and install requirements(python 3.8, torch, sklearn).

### Download Pre-processed Dataset
Download [dataset files](https://cowtransfer.com/s/31a630e5ab3f40)(about 640 MB), extract them and put them in the root directory

Contained files (15 files)
- TAttention_Mask_Relu_Score_bert_context_dataset.pickle **[Pre-processed Dataset]**
- TAttention_Mask_Relu_Score_dataset.pickle **[Pre-processed Dataset]**
- context_bert_all.dataset **[Context Textual Feature]**
- context_resnet.pickle **[Context Visual Feature]**
- context_audio.pickle **[Context Acoustic Feature]**
- context_audio_mask.pickle **[Used Context Acoustic Feature]**
- bert_global_embedding.pickle **[Utterance-level Textual Feature]**
- sarcasm_resnet_utterance.pickle **[Utterance-level Visual Feature]**
- sarcasm_opensmile_utterance.dataset **[Utterance-level Acoustic Feature]**
- audio_mask.pickle **[Used Utterance-level Acoustic Feature]** 
- sarcasm_vat_ferplus.dataset **[Word-level Visual and Acoustic Feature]** 
- sarcasm_last_bert.dataset **[Word-level Textual Feature]** 
- SentiWords_1.1.txt **[Sentiment Lexicon]**
- sarcasm_dataset_indices.pickle **[Official Dataset Splits]**
- sarcasm_data.json **[Official Dataset Annoation]**

### Some Details
- We extract the word-level, utterance-level and context features from the mutlimodal inputs. 
- We perform [feature selection](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-using-selectfrommodel) for the audio features to reduce the feature dimension.  
- We first train the model and then extract the features before the classification layer. The extraced features are feed into the **SVM** to obtian the final results. 


### Usage
If you only want to reproduce our experimental results, you can run the code like this.

```python
# Word-Level + Utterance-Level
python svm_classification.py --name TAttention_Mask_Relu_Score

# Word-Level + Utterance-Level + Context
python svm_classification_context.py --name TAttention_Mask_Relu_Score_bert_context
```

If you want to train our model from scratch, you can first run the main*.py to train the model.

```python
# Using context features
python main_context.py --model TAttention_Mask_Relu_Score --name TAttention_Mask_Relu_Score_bert_context
# Do not using context features
python main_batch.py --model TAttention_Mask_Relu_Score --name TAttention_Mask_Relu_Score
```
And then you can run the feature_extraction*.py

```python
# Using context features
python feature_extraction_context.py --name TAttention_Mask_Relu_Score_bert_context
# Do not using context features
python feature_extraction.py --name TAttention_Mask_Relu_Score
```
Finally, you can run the svm_classification*.py to evaluate the model.


### Citation
@ARTICLE{9387561,  author={Wu, Yang and Zhao, Yanyan and Lu, Xin and Qin, Bing and Wu, Yin and Sheng, Jian and Li, Jinlong},  journal={IEEE MultiMedia},   title={Modeling Incongruity between Modalities for Multimodal Sarcasm Detection},   year={2021},  volume={28},  number={2},  pages={86-95},  doi={10.1109/MMUL.2021.3069097}}
