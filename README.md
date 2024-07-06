# Examination of the Evolution of Language Among Dark Web Users

## Project Description

The objective of this project was to analyze the evolution of language among Dark Web users using a series of Natural Language Processing (NLP) models. Several NLP models were developed and trained, including TF-IDF, LDA, BERT, and LSTM, to understand the context, sentiment, and thematic elements of forum discussions. After thorough analysis, the BERT model was chosen as the most effective.

For more details on the project, you can read the Medium article [here](#).


## Installation

To install the project, follow these simple steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/D0men1c0/GSoC
    ```
2. Install the dependencies using `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```


## Repository Structure

The structure of the repository is as follows:

```
├───Analyze_files
│   ├───CombiningAnalysisCompleteDataset
│   │   ├───ContentAnalysis
│   │   │   ├───DatasetsContentBERTopic
│   │   │   ├───ModelsContent
│   │   │   │   └───topic_model_all-MiniLM-L6-v2_190_20n_8dim_safetensors
│   │   │   ├───PreProcessFiles
│   │   │   └───ZeroShotClassificationResultsContent
│   │   │       └───all-MiniLM-L6-v2_190_20n_8dim
│   │   └───ThreadAnalysis
│   │       ├───DatasetsThreadBERTopic
│   │       ├───Models
│   │       │   ├───topic_model_all-MiniLM-L6-v2_150_20n_safetensors
│   │       │   ├───topic_model_all-MiniLM-L6-v2_200_safetensors
│   │       │   └───topic_model_all-MiniLM-L6-v2_400_safetensors
│   │       ├───OtherFilesPreviousApproach
│   │       ├───ResultsCluster
│   │       ├───ResultsGridSearchBERTopic
│   │       └───ZeroShotClassificationResults
│   │           ├───all-MiniLM-L6-v2_150_20n
│   │           ├───all-MiniLM-L6-v2_200
│   │           └───all-MiniLM-L6-v2_400
│   └───SingleDatasetsAnalysis
├───Datasets
│   ├───CleanedData
│   ├───FeatureEngineeringData
│   │   ├───Boards
│   │   ├───Members
│   │   └───Threads
│   ├───IntentCrime
│   └───RawData
├───Img
│   ├───Content
│   └───Thread
├───MergedModelBERT
├───MLModelsBERT
│   └───Thread
│       └───SavedModels
├───ShowModelsBaselineBERT
│   ├───Content
│   └───Thread
├───ShowResultsHTML
└───Util
```


## Datasets

The CrimeBB dataset was used, specifically the "dread-2023-06-21" data scraped from Dread, updated on 2023-06-21 (for more details see the Citations section).<br> 
Dread is a popular forum on the Dark Web where users exchange opinions and reviews on various topics, including drug sales, seller reviews, password and bitcoin transactions, as well as passports and fake IDs.

For more details, here is a table structure:

<img src="structure_tables.png" alt="structure_tables" width="500" height="600"/>

The following tables were analyzed to extract the topics:
- **Post:** Contains 290k records.
- **Thread:** Contains 75k records.


## Summary of Work Done

Several NLP models were developed and trained, including TF-IDF, LDA, BERT, and Machine Learning algorithm. Through analysis, the BERT model was chosen as the most effective model for understanding the context, sentiment, and thematic elements of discussions on the Dread forum.


## Results

Here are the main results obtained from the analysis:

| Metric              | Value   |
|---------------------|---------|
| Accuracy            | 92.3%   |
| Precision           | 91.8%   |
| Recall              | 90.7%   |
| F1 Score            | 91.2%   |



## Usage

Before executing the code, navigate to the directory where the saved BERTopic models are located:

```bash
cd Analyze_files/CombiningAnalysisCompleteDataset/ContentAnalysis
```

To perform topic prediction using the BERTopic model, follow these steps:

```python
import sys
sys.path.append('../../../Util')
import BERTopicUtils as btu
from bertopic import BERTopic

# Load the BERTopic model
topic_model = BERTopic.load("ModelsContent/topic_model_all-MiniLM-L6-v2_190_20n_8dim")

# Define a sentence for prediction
sentence = ['recently closed Samsara market']

# Perform topic prediction
btu.predict_topic(topic_model, sentence, custom_labels=True)
```

The BERTopic model used is one example, there are several. Just go to the ModelsContent subdirectory to see how many more there are.<br>
Or also use the templates in the thread section on Models:

```bash
cd ../ThreadAnalysis/Models
```

By default, the top 5 labels are set, but just set the `num_classes` parameter with the number of topics desired.

### Results
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Topic</th>
      <th>Probability</th>
      <th>Label</th>
      <th>Words</th>
      <th>Sentence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[(samsara, 0.11383384349850058), (market, 0.01...</td>
      <td>1.0</td>
      <td>samsara market</td>
      <td>[samsara, market, samsara market, sam, dream, ...</td>
      <td>recently closed Samsara market</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[(subdread, 0.047141772290036604), (sub, 0.018...</td>
      <td>0.0</td>
      <td>subdread - sub - post</td>
      <td>[subdread, sub, post, subdreads, create, dread...</td>
      <td>recently closed Samsara market</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[(empire, 0.10987278200488068), (nightmare, 0....</td>
      <td>0.0</td>
      <td>empire - dread</td>
      <td>[empire, nightmare, empire empire, find empire...</td>
      <td>recently closed Samsara market</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[(onion, 0.09360299836020991), (dot onion, 0.0...</td>
      <td>0.0</td>
      <td>onion link</td>
      <td>[onion, dot onion, dot, onion link, onion site...</td>
      <td>recently closed Samsara market</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[(det, 0.04309335455807283), (er, 0.0412459466...</td>
      <td>0.0</td>
      <td>er - det - og</td>
      <td>[det, er, og, har, jeg, som, ikke, til, en, med]</td>
      <td>recently closed Samsara market</td>
    </tr>
  </tbody>
</table>
</div>

## Acknowledgements


I would like to extend my heartfelt gratitude to my supervisors Andrea Underhill, Jane Daquin, and the head of the organization Sergei Gleyzer for their unwavering support and guidance during the Google Summer of Code (GSoC) 2024 with HumanAI. Their mentorship provided me with the opportunity to fully express my creativity and push the boundaries of my capabilities.<br> This project was made possible thanks to the Google Summer of Code: [Google Summer of Code 2024](https://summerofcode.withgoogle.com/) and HumanAI: [HumanAI Foundation](https://humanai.foundation/).


## Citation

Dataset used CrimeBB: dread-2023-06-21

```
@inproceedings{10.1145/3178876.3186178,
    title = {CrimeBB: Enabling Cybercrime Research on Underground Forums at Scale},
    author = {Pastrana, Sergio and Thomas, Daniel R. and Hutchings, Alice and Clayton, Richard},
    year = {2018},
    isbn = {9781450356398},
    publisher = {International World Wide Web Conferences Steering Committee},
    address = {Republic and Canton of Geneva, CHE},
    url = {https://doi.org/10.1145/3178876.3186178},
    doi = {10.1145/3178876.3186178},
    booktitle = {Proceedings of the 2018 World Wide Web Conference},
    pages = {1845–1854},
    numpages = {10},
    location = {Lyon, France},
    series = {WWW '18}
}
```

BERTopic for experiments

```
@article{grootendorst2022bertopic,
  title={BERTopic: Neural topic modeling with a class-based TF-IDF procedure},
  author={Grootendorst, Maarten},
  journal={arXiv preprint arXiv:2203.05794},
  year={2022}
}
```