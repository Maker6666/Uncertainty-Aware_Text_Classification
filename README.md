# Uncertainty-Aware Sentence Embedding in Text Classification
The text classification task is implemented based on the pre-training model like BERT and ST5, and the uncertainty analysis of feature embedding is carried out to improve the classification performance
## [SentEval](https://github.com/facebookresearch/SentEval) 
SentEval is a library for evaluating the quality of sentence embeddings. We apply our method to accomplish the classification task.
### Download datasets
To get all the transfer tasks datasets, run (in senteval/data/downstream/):   
```javascript copy
./get_transfer_data.bash
```
To train and reproduce the results in our article, run (in senteval/examples/):
```javascript copy
python bert_variation.py
```
```javascript copy
python st5_enc_variation.py
```

## Multi-label text classification
Our pre-trained models are available: 

1.Toxic Comment 

2.GoEmotions-original 

3.AAPD 

* [Toxic Comments dataset](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)  
  The Toxic Comment dataset is a large-scale dataset available on Kaggle. It consists of numerous online comments 
  labeled for toxicity, including 6 categories such as toxic, severe toxic, obscene, threat, insult, and identity hate.

  In 
* GoEmotions dataset  
  Number of examples: 58,009, Number of labels: 27 + Neutral.
  raw dataset can be retrieved by running:
  ```javascript copy
  wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
  wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
  wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv
  ```
* [AAPD dataset](https://git.uwaterloo.ca/jimmylin/Castor-data/tree/master/datasets/AAPD)
  
  

## References
[1] A. Conneau, D. Kiela, SentEval: An Evaluation Toolkit for Universal Sentence Representations
```
@article{conneau2018senteval,
  title={SentEval: An Evaluation Toolkit for Universal Sentence Representations},
  author={Conneau, Alexis and Kiela, Douwe},
  journal={arXiv preprint arXiv:1803.05449},
  year={2018}
}
```
[2] MAGNET: Multi-Label Text Classification using Attention-based Graph Neural Network  
[3] Demszky, Dorottya and Movshovitz-Attias, GoEmotions: A Dataset of Fine-Grained Emotions
```
@inproceedings{demszky2020goemotions,
  author = {Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and 
   Ravi, Sujith},
  booktitle = {58th Annual Meeting of the Association for Computational Linguistics (ACL)},
  title = {{GoEmotions: A Dataset of Fine-Grained Emotions}},
  year = {2020}
}
```
[4] Pengcheng Yang , SunSGM: Sequence Generation Model for Multi-label Classification
```
@inproceedings{YangCOLING2018,
  author    = {Pengcheng Yang and Xu Sun and Wei Li and Shuming Ma and Wei Wu andHoufeng Wang},
  title     = {{SGM:} Sequence Generation Model for Multi-label Classification},
  booktitle = {Proceedings of the 27th International Conference on Computational
               Linguistics, {COLING} 2018, Santa Fe, New Mexico, USA, August 20-26,
               2018},
  pages     = {3915--3926},
  year      = {2018}
}
```

