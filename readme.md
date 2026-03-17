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
- NLTK (https://www.nltk.org/)
- Gensim (https://radimrehurek.com/gensim/)
- Autocorrect (https://pypi.org/project/autocorrect/)
- Langdetect (https://pypi.org/project/langdetect/)

## Code
### Publicly Available Code
- HuggingFace Transformers sequence example (https://huggingface.co/docs/transformers/tasks/sequence_classification/)： modified around 70 lines of code
- Sentiment Classification using BERT examples + Wordcloud (https://www.geeksforgeeks.org/nlp/sentiment-classification-using-bert/): modified around 50 lines of code
- Qwen usage examples (https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/): added and mofified 100+ lines of code

### Original Code
- data_scripts.ipynb: Cleans and preprocesses both Kaggle and Mendeley datasets by filtering non-English reviews, removing invalid entries, and balancing class labels (~200 lines)
- cleaned_data_scripts.ipynb: Performs further exploration and analysis on the cleaned Mendeley and Kaggle datasets (~100 lines)
- tdidf_logistic_regression.ipynb: Trains a TF-IDF logistic regression model on Mendeley data and evaluates cross-dataset generalization on Kaggle data (~150 lines)
- BOW_kaggle.ipynb: Spell checks and generates token-frequency BOW JSON of the Kaggle dataset (~150 lines)
- BOW_mendeley.ipynb: Spell checks and generates token-frequency BOW JSON of the Mendeley dataset (~170 lines)
- BOW_spellcorrect.ipynb: Performs spell checks and generates spell corrected version of both Kaggle and Mendeley datasets (~100 lines)
- BOW_logistic_reg.ipynb: Trains a BOW logistic regression model on Mendeley data and evaluates cross-dataset generalizatoin on Kaggle data (~230 lines)
- weighted_BOW_logistic_reg.ipynb: Trains a weighted BOW logistic regression model on Mendeley data and evaluates cross-dataset generalization on Kaggle data (~230 lines)
- static_embed_logistic_reg.ipynb: Trains a word2vec skip-gram logistic regression model on Mendeley data and evlauates cross-dataset generalization on Kaggle data (~280 lines)
- confusion_comparison.ipynb: Compares false positive and negative reviews across models (~150 lines)
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
