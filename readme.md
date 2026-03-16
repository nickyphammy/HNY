# Team HNY - Sentiment Classfication on Game Reviews
UCI COMPSCI175 Winter 2026
Team Members:
- Qian Ying Wong, 49411619, qywong@uci.edu
- Nicholas Pham, 18778261, nlpham1@uci.edu
- Helena Sun, 87415451, helenays@uci.edu

## Libraries and Packages Used
- HuggingFace Transformers (https://huggingface.co/docs/transformers/index)
- HuggingFace Datasets (https://huggingface.co/docs/datasets/index)
- Matplotlib (https://matplotlib.org/)
- NumPy (https://numpy.org/)
- PyTorch (https://pytorch.org/)
- Scikit-learn (https://scikit-learn.org/stable/)
- Seaborn (https://seaborn.pydata.org/)
- Wordcloud (https://amueller.github.io/word_cloud/)

## Code
### Publicly Available Code
- HuggingFace Transformers sequence example (https://huggingface.co/docs/transformers/tasks/sequence_classification/)： modified around 70 lines of code
- Sentiment Classification using BERT examples + Wordcloud (https://www.geeksforgeeks.org/nlp/sentiment-classification-using-bert/): modified around 50 lines of code
- Qwen usage examples (https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/): added and mofified 100+ lines of code

### Original Code
- BERT_FineTuned_more_analysis.ipynb: Loads and processes datasets, tokenizes reviews, fine-tunes a pretrained BERT model, and evaluates performance (~500 lines)
- BERT_large_FineTuned.ipynb: similar to above but runs the "bert-large-cased" version, which has more layers and is much more computationally intensive (~275 lines)
- Qwen_benchmark.ipynb: Loads dataset, formats prompts of the language model, generates predictions, and evaluates performance (~175 lines)


## Proposals and Reports
- [Project Proposal](https://docs.google.com/document/d/1JQeRbgVZoeeWs-vfXFODf35sWo_yMrJy02VVgvN1l2o/edit?tab=t.0)
- [Weekly Report](https://docs.google.com/document/d/1gB0Q73awFsgA8-8XRjMNActGP-1Vy8saouBG0fRiXTw/edit?tab=t.0)
- [Progress Report](https://docs.google.com/document/d/1FrJaUMIe8TUsx5eEiiuDMewd5Hdc5jt29ipr6e71QMA/edit?tab=t.0)
- [Final Report](https://docs.google.com/document/d/1fMFvXQDifmDEiDFyWAVEA7P5vAKRFoDq/edit)

## Datasets
- [Kaggle - Steam Review&Games Dataset](https://www.kaggle.com/datasets/filipkin/steam-reviews)
- [Mendeley - Steam Games Metadata and Player Reviews](https://data.mendeley.com/datasets/jxy85cr3th/2)
