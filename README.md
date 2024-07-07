# Examination of the Evolution of Language Among Dark Web Users

## Project Description

The objective of this project was to analyze the evolution of language among Dark Web users using a series of Natural Language Processing (NLP) models. Several NLP models were developed and trained, including TF-IDF, LDA, BERT, and LSTM, to understand the context, sentiment, and thematic elements of forum discussions. After thorough analysis, the BERT model was chosen as the most effective.

For more details on the project, you can read the Medium article [here](#).

## Table of Contents

- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets Used](#datasets-used)
- [Summary of Work Done](#summary-of-work-done)
- [Results](#results)
- [Future work](#future-work)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)


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

The roles of the different folders are detailed below:

1. `Analyze_files`
Contains files and scripts for data analysis.

    1. `CombiningAnalysisCompleteDataset`
    Folder containing the analysis of tables (boards, threads, members, posts) merged into one

        1. `ContentAnalysis`
        Focuses on analysing the content of table posts using BERT models to extract topics.

            - `DatasetsContentBERTopic`: Contains datasets with the original content field with information extracted from BERT.
            - `ModelsContent`: Holds the specific BERTopic models for content analysis.
            - `PreProcessFiles`: Contains the pre-processing files of the content field.
            - `ZeroShotClassificationResultsContent`: Stores the results of the zero shot classification on custom topic names.

        2. `ThreadAnalysis`
        Focuses on analysing the content of table thread using BERT models to extract topics.

            - `DatasetsThreadBERTopic`: Same role as the content, focusing instead on the thread 
            - `OtherFilesPreviousApproach`: Contains files from previous approaches.
            - `ResultsCluster`: Dataset on cluster analysis of a previous approach.
            - `ResultsGridSearchBERTopic`: Contains the results of the grid search of BERTopic.
            - `ZeroShotClassificationResults`: Same role as the content, focusing instead on the thread.

    2. `SingleDatasetsAnalysis`
    Contains analyses for separate individual tables (boards, discussions, members, messages).

2. `Datasets`
Stores tables datasets used in the project.

    - `CleanedData`: Contains cleaned data ready for analysis.
    - `FeatureEngineeringData`: Contains data prepared for feature engineering.
    - `IntentCrime`: Specific dataset related to crime intent used for Zero-shot classification.
    - `RawData`: Contains raw, unprocessed data tables.

3. `Img`
    Stores images used for topic representation graphics found with BERTopic.

    - `Content`: Images related to content analysis.
    - `Thread`: Images related to thread analysis.

4. `MergedModelBERT`
Contains script merged BERT models.

5. `MLModelsBERT`
    Contains Machine Learning scripts and models used to validate datasets obtained from BERT approaches on content and threads.

6. `ShowModelsBaselineBERT`
    Displays baseline BERT models.

    - `Content`: Content-related models.
    - `Thread`: Thread-related models.

7. `ShowResultsHTML`
Contains HTML files showing results of the analysis.

8. `Util`
Utility scripts and auxiliary functions used throughout the project to process data and evaluate models.


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

Clearly the custom predict_topic function is being used, alternatively one could directly use BERTopic's fit_transform without importing anything else.

### Example Results
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


## Datasets Used

The CrimeBB dataset was used, specifically the "dread-2023-06-21" data scraped from Dread, updated on 2023-06-21 (for more details see the Citations section).<br> 
Dread is a popular forum on the Dark Web where users exchange opinions and reviews on various topics, including drug sales, seller reviews, password and bitcoin transactions, as well as passports and fake IDs.

For more details, here is a tables structure:

<img src="structure_tables.png" alt="structure_tables" width="500" height="600"/>

<br>The following tables were analyzed to extract the topics:
- **Post:** Contains 290k records.
- **Thread:** Contains 75k records.

## Summary of Work Done

The objective of this project was to analyze the evolution of language among Dark Web users using a series of Natural Language Processing (NLP) models. Initially, each table in the dataset was examined, reducing the number of features and eliminating outliers and duplicates. The focus was specifically on two tables: threads and posts, analyzing the fields `thread` and `content` respectively.

To gain an overview of the topics discussed, the `thread` field was analyzed first. Various approaches such as TF-IDF, LDA, and Zero-shot classification were used, and a class was developed that combined Sentence Transformer with t-SNE, PCA, and k-means. However, these methods did not yield optimal results.<br> Therefore, the BERTopic library was employed, which, thanks to its modularity, enabled the analysis of text by applying different strategies including Sentence BERT, c-TFIDF, KeyBERTInspired, introducing diversity with custom classes, and using UMAP for dimensionality reduction and HDBSCAN for clustering. Subsequently, Zero-shot classification was applied to the topic names.

This comprehensive approach resulted in two baselines:

- One with 7 macro clusters (general topics)
- Another with 68 clusters (aiming to identify as many relevant topics as possible while avoiding micro clusters)

These results were validated using metrics and distribution graphs provided by BERTopic, which also allowed for the analysis of topic distribution over time.

To further validate the results from an accuracy metrics perspective, a LightGBM model was trained using the embeddings as the input features (X) and the topics identified by BERTopic as the target variable. This was done to validate the BERTopic results in a classification task.

Subsequently, the `content` field in the posts was analyzed to verify if the topics identified matched those in the threads, which they did, resulting in 120 clusters.


## Results

These are the results of the cluster analysis produced by BERT on both threads and content. For further graphical analysis (distance between clusters, hierarchy, distribution in space), please consult the html directory: ShowResultsHTML.

Thread:

| Metric                                  | Value (68 Topics) | Value (7 Topics) |
|-----------------------------------------|-------------------|------------------|
| Coherence Score                         | 0.57              | 0.40             |
| Silhouette Score                        | 0.50              | 0.51             |
| Davies Bouldin Score                    | 0.87              | 0.76             |
| Dos Score (diversity overlapped Score)  | 0.06              | 0.20             |
| % Outliers                              | 0.30              | 0.42             |

Content:

| Metric                                  | Value (121 Topics)|
|-----------------------------------------|-------------------|
| Coherence Score                         | 0.69              |
| Silhouette Score                        | 0.60              |
| Davies Bouldin Score                    | 0.46              |
| Dos Score (diversity overlapped Score)  | 0.24              |
| % Outliers                              | 0.35              |

These are the results obtained by LightGBM using the embedding and classes obtained by BERT for the thread field in a classification task:

Thread:

| Metric              | Value (68 Topics) | Value (7 Topics) |
|---------------------|-------------------|------------------|
| Accuracy            | 0.89              | 0.96             |
| Precision           | 0.91              | 0.96             |
| Recall              | 0.84              | 0.95             |
| F1 Score            | 0.87              | 0.96             |

## Future work

- Merge Baselines: The next steps involve merging the baselines obtained from the Thread and Content sections into a single model to integrate both representations, with Thread providing general topics and Content extending them.
- Validation: Validate the results using clustering metrics and machine learning models. Given the data volume, LightGBM may not be sufficient, so exploring neural networks could be beneficial.
- Topic Refinement: Conduct a final review of the topics, potentially integrating specific expressions that need to be highlighted (e.g., a particular abbreviation for a drug that wasn't identified).
- Deployment on HuggingFace: Push both the "general" model with 7 topics and the more specific model resulting from merging the Thread and Content sub-models to HuggingFace.
- Temporal Validation: If time permits, consider using LSTM to validate the results over time (currently done with BERT), which could provide additional insights.
- Multimodal Model: Finally, consider making the model multimodal by incorporating both text and images. This would require more in-depth development.

## Acknowledgements


I would like to extend my heartfelt gratitude to my supervisors Jane Daquin, Andrea Underhill and the head of the organization Sergei Gleyzer for their unwavering support and guidance during the Google Summer of Code (GSoC) 2024 with HumanAI. Their mentorship provided me with the opportunity to fully express my creativity and push the boundaries of my capabilities.<br> This project was made possible thanks to the Google Summer of Code: [Google Summer of Code 2024](https://summerofcode.withgoogle.com/) and HumanAI: [HumanAI Foundation](https://humanai.foundation/).


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