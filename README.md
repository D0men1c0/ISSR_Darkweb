# Examination of the Evolution of Language Among Dark Web Users

<p align="center">
    <img src="Img/GSoC@HumanAI.png" alt="GSoC@HumanAI" width="800" height="300"/>
</p>

## Table of Contents

- [Project Description](#project-description)
- [Utility of Dark Web Language Analysis](#utility-of-dark-web-language-analysis)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
  - [Running on Google Colab](#running-on-google-colab)
- [Usage](#usage)
  - [Safetensors and Hugging Face](#safetensors-and-hugging-face)
- [Summary of the Analysis](#summary-of-the-analysis)
- [Datasets Used](#datasets-used)
- [Summary of Work Done](#summary-of-work-done)
- [Results](#results)
  - [Cluster Validation using LightGBM and LSTM](#cluster-validation-using-lightgbm-and-lstm)
- [Multimodal Model](#multimodal-model)
- [Future work](#future-work)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Project Description

The objective of this project was to analyze the evolution of language among Dark Web users using a series of Natural Language Processing (NLP) models. Several NLP models were developed and trained, including TF-IDF, LDA, BERT, and LSTM, to understand the context, sentiment, and thematic elements of forum discussions. After thorough analysis, the BERT model was chosen as the most effective.

For more details on the project, refer to the Medium article [here](https://medium.com/@domenicolacavalla8/examination-of-the-evolution-of-language-among-dark-web-users-67fd3397e0fb).

## Utility of Dark Web Language Analysis

This analysis is highly useful because it allows for the examination of large volumes of data to identify the main topics discussed, along with the nuances of associated words. Understanding these elements can provide insights into the nature of criminal activities, the evolution of their language, and the connections between various slang terms and specific illicit activities. This information can be invaluable for law enforcement, cybersecurity professionals, and researchers studying the dynamics of underground online communities.

However, only one forum has been analyzed so far. By combining multiple models with data from different forums over several years, a comprehensive picture of dark web language can be obtained. This is important because there may be other forums discussing topics not covered on Dread, such as red rooms, child pornography, hired killers, cannibalism, etc.

Unlike other approaches based on a few datasets, this method allows for the analysis of a vast number of conversations, greatly enhancing the breadth and depth of insights. In this way, a unique, highly comprehensive model could be developed (requiring multiple datasets), encompassing the most important topics discussed across various forums with a broader timeline. For example, insights could be drawn from forums like SilkRoad, which was very active and where certain words were used and then fell out of use.

Additionally, associating this data with images could lead to more precise identification. The current results are already highly valuable, demonstrating the potential of this approach.

## Repository Structure

The structure of the repository is as follows:

```
├───Analyze_files
│   ├───CombiningAnalysisCompleteDataset
│   │   ├───ContentAnalysis
│   │   │   ├───DatasetsContentBERTopic
│   │   │   ├───LLAMA
│   │   │   ├───ModelsContent
│   │   │   │   ├───topic_model_all-MiniLM-L6-v2_150_150n_10dim_white_nation_safetensors
│   │   │   │   └───topic_model_all-MiniLM-L6-v2_190_20n_8dim_safetensors
│   │   │   ├───PreProcessFiles
│   │   │   └───ZeroShotClassificationResultsContent
│   │   │       ├───all-MiniLM-L6-v2_150_150n_10dim
│   │   │       └───all-MiniLM-L6-v2_190_20n_8dim
│   │   └───ThreadAnalysis
│   │       ├───DatasetsThreadBERTopic
│   │       ├───LLAMA
│   │       ├───Models
│   │       │   ├───topic_model_0.50Sil300_safetensors
│   │       │   ├───topic_model_0.64SilNew_safetensors
│   │       │   ├───topic_model_all-MiniLM-L6-v2_150_150n_10dim_raid_safetensors
│   │       │   ├───topic_model_all-MiniLM-L6-v2_150_20n_safetensors
│   │       │   ├───topic_model_all-MiniLM-L6-v2_200_safetensors
│   │       │   └───topic_model_all-MiniLM-L6-v2_400_safetensors
│   │       ├───OtherFilesPreviousApproach
│   │       ├───PreProcessFiles
│   │       ├───ResultsCluster
│   │       ├───ResultsGridSearchBERTopic
│   │       └───ZeroShotClassificationResults
│   │           ├───all-MiniLM-L6-v2_150_20n
│   │           ├───all-MiniLM-L6-v2_200
│   │           ├───all-MiniLM-L6-v2_400
│   │           └───distiluse_7cluster
│   └───SingleDatasetsAnalysis
├───ComputerVision
│   ├───Datasets
│   │   └───RawData
│   │       ├───test
│   │       ├───train
│   │       └───valid
│   ├───Embeddings
│   └───Models
│       └───topic_visual_model_safetensors
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
│   ├───Merged
│   ├───Thread
│   └───Visual
├───MergedModelBERT
│   ├───Merged_Models_safetensors
│   ├───Merged_Models_White_Nations_Raid_safetensors
│   └───Merged_Models_White_Nations_safetensors
├───MLModelsBERT
│   ├───Content
│   │   └───SavedModels
│   └───Thread
│       └───SavedModels
├───ShowModelsBaselineBERT
│   ├───Content
│   │   ├───CSV121Topic
│   │   └───CSV31TopicWhiteNation
│   └───Thread
│       ├───26TopicRaid
│       │   └───CSV26TopicRaid
│       ├───68Topic
│       │   └───CSV68Topic
│       └───7Topic
│           └───CSV7Topic
├───ShowResultsHTML
│   ├───ReproducibilityResults
│   └───ShowFinalAnalysisBaselines
└───Util
```

The roles of the different folders are detailed below:

1. `Analyze_files`
  Contains files and scripts for data analysis.

    1. `CombiningAnalysisCompleteDataset`
      Folder containing the analysis of tables (boards, threads, members, posts) merged into one

        1. `ContentAnalysis`
          Focuses on analysing the content of table posts using BERT models to extract topics. Contains notebook analysis (and their folders) on `White Nations` and `Dread` posts.

            - `DatasetsContentBERTopic`: Contains datasets with the original content field with information extracted from BERT.
            - `LLAMA`: Stores the results of the use of Mistral on custom topic names.
            - `ModelsContent`: Holds the specific BERTopic models for content analysis (both pickle and safetensors).
            - `PreProcessFiles`: Contains the pre-processing files of the content field.
            - `ZeroShotClassificationResultsContent`: Stores the results of the zero shot classification on custom topic names.

        2. `ThreadAnalysis`
          Focuses on analysing the content of table thread using BERT models to extract topics. Contains notebook analysis (and their folders) on `Raid Forums` and `Dread` threads.

            - `DatasetsThreadBERTopic`: Same role as the content, focusing instead on the thread
            - `LLAMA`: Same role as the content, focusing instead on the thread.
            - `Models`: Same role as the content, focusing instead on the thread.
            - `OtherFilesPreviousApproach`: Contains files from previous approaches.
            - `PreProcessFiles`: Same role as the content, focusing instead on the thread.
            - `ResultsCluster`: Dataset on cluster analysis of a previous approach.
            - `ResultsGridSearchBERTopic`: Contains the results of the grid search of BERTopic.
            - `ZeroShotClassificationResults`: Same role as the content, focusing instead on the thread.

    2. `SingleDatasetsAnalysis`
      Contains analyses for separate individual tables (boards, posts, members, threads) for the Dread dataset, as well as a threads table for Raid Forums and a posts table for White Nations.

2. `Datasets`
    Stores tables datasets used in the project.

    - `CleanedData`: Contains cleaned data ready for analysis.
    - `FeatureEngineeringData`: Contains data prepared for feature engineering.
    - `IntentCrime`: Specific dataset related to crime intent used for Zero-shot classification.
    - `RawData`: Contains raw, unprocessed data tables.

3. `ComputerVision`
    Contains resources for making the model multimodal by integrating computer vision components.<br>
    Due to the lack of direct image data from the dark web, a public dataset was used. <br>
    The dataset is sourced from [Roboflow's Drug Detection Project](https://universe.roboflow.com/freakinggojo-er8b5/drug_detection_project/dataset/9) and includes three folders: train, test, and validation, featuring images related to drugs, guns, people, and robbers. <br>
    The dataset comprises approximately 3700 images with corresponding annotations.

    - `Datasets`: 
      - `RawData`: Contains the dataset in three subfolders:
        - *test*: Images for testing
        - *train*: Images for training
        - *valid*: Images for validation

    - `Embeddings`: Contains embeddings derived from the images.

    - `Models`: Includes the trained visual model.

    Additionally, there is a notebook file that demonstrates the visual model's performance in identifying the four topics with corresponding predictions.

4. `Img`
    Stores images used for topic representation graphics found with BERTopic.

    - `Content`: Images related to content analysis.
    - `Merged`: Images related to merged models analysis.
    - `Thread`: Images related to thread analysis.
    - `Visual`: Images related to visual model.

5. `MergedModelBERT`
    Contains scripts for the merged BERT models based on the baselines. Specifically, there are three new merged models: one for Dread content and Dread threads, one combining the previous model with White Nations, and a final model that merges everything—Dread content, Dread threads, White Nations, and Raid Forums—resulting in a total of `173` topics. <br> 
    This was done to create a robust model capable of covering as many topics as possible. <br> 
    This folder also contains the resulting models (pickle and safetensors).

6. `MLModelsBERT`
    Contains Machine Learning scripts and models used to validate datasets obtained from BERT approaches on content and threads including LightGBM and LSTM.

7. `ShowModelsBaselineBERT`
    Provides summaries of the baseline BERT models (using notebooks) and includes additional analyses with CSV files. <br>
    hese analyses cover topic descriptions, original documents with topic associations, sentiment analysis for each document and topic, the evolution of topics over time, and example of predictions.

    - `Content`: 
      - *Dread Posts*: 121 topics
      - *White Nation Posts*: 31 topics

    - `Thread`: 
      - *Dread Threads*: 7 topics and 68 topics
      - *Raid Forums Threads*: 26 topics

    In addition to summarizing the models, this section also presents insights and analyses using various charts. <br> 
    These include visualizations of the topics, documents, topic evolution over time, and sentiment analysis for each topic, offering a comprehensive view beyond just the model summaries.

8. `ShowResultsHTML`
    Contains HTML files that display the results of the final analysis and the reproducibility of the results. <br>

    This folder has two subdirectories:
    - `ReproducibilityResults`: Documents the entire process of training, evaluating, and testing all baseline models, including LSTM and LightGBM.
    - `ShowFinalAnalysisBaselines`: Presents the analyses from section 7, offering a comprehensive and consolidated view of the results.

9. `Util`
    Utility scripts and auxiliary functions used throughout the project to process data and evaluate models.<br>
    **Note:** If using Google Colab and wanting to import these modules, see point 4 of this section: [Running on Google Colab](#running-on-google-colab)

## Installation

To install the project, follow these simple steps. **Note:** that this project uses Python `3.12.2`

1. Clone the repository:
    ```bash
    git clone https://github.com/D0men1c0/ISSR_Darkweb
    ```
2. Install the dependencies using `requirements.txt`:

    Using `venv` (Virtual Environment):
    ```bash
    # Ensure you are using Python 3.12
    python --version

    # Create and activate a virtual environment (optional but recommended)
    python3.12 -m venv myenv
    source myenv/bin/activate   # On Windows: myenv\Scripts\activate

    # Install the dependencies using requirements.txt
    pip install -r requirements.txt
    ```

    Using `conda`:
    ```bash
    # Ensure you are using Python 3.12
    python --version

    # Create and activate a conda environment (optional but recommended)
    conda create --name myenv python=3.12
    conda activate myenv

    # Install the dependencies using requirements.txt
    pip install -r requirements.txt
    ```
3. (Optional) Run the `download_files.py` script to download the additional files from Google Drive. This script handles retrieving files and placing them in the appropriate directories within the repository.<br>
Note: being quite heavy files (7 GB in total) choose carefully which models and files to download.
    ```bash
    python download_files.py
    ```
4. (Optional) Run the `llama_download.py` script to download the llama quantized model library, which is used for assigning labels. This file is approximately 4.2GB and is not required for running the examples but is necessary for reproducibility of results. 
    ```bash
    python llama_download.py
    ```
    Note: with Google Colab, just run this:
      ```bash
      !wget https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf
      ```

### Running on Google Colab

To run the notebooks on Google Colab, follow these steps:

1. Upload the repository to your Google Drive either manually or by cloning it from GitHub. <br>You can refer to this [guide](https://www.geeksforgeeks.org/how-to-clone-github-repository-and-push-changes-in-colaboratory/)

2. Mount your Google Drive in Colab:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3. Change directory to the location of the current `.ipynb` file. <br>Replace `/content/drive/MyDrive/GSoC/Analyze_files/CombiningAnalysisCompleteDataset/ContentAnalysis/` with the actual path to your current `.ipynb` file:
    ```python
    import os
    os.chdir('/content/drive/MyDrive/GSoC/Analyze_files/CombiningAnalysisCompleteDataset/ContentAnalysis/')
    ```
4. To import the `.py` files present in the `Util` module, insert the directory containing the utility modules into the system path. <br>Replace `/content/drive/MyDrive/GSoC/Util/` with the actual path to your utilities directory:
    ```python
    import sys
    sys.path.insert(0, '/content/drive/MyDrive/GSoC/Util/')
    ```
    This way, you can import the `.py` modules present in the `Util` directory.
4. Install the necessary libraries. For example:
    ```python
    !pip install bertopic
    !torch
    !pip install llama_cpp_python
    !pip install umap_learn
    !pip install hdbscan
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
topic_model = BERTopic.load("ModelsContent/topic_model_all-MiniLM-L6-v2_190_20n_8dim", embedding_model='all-MiniLM-L6-v2')

# Define a sentence for prediction
sentence = ['recently closed Samsara market']

# Perform topic prediction
btu.predict_topic(topic_model, sentence, custom_labels=True)
```

**Note:** we are using the `topic_model_all-MiniLM-L6-v2_190_20n_8dim`, which is not directly in the directory because it weighs 2.5GB but can be downloaded directly from the file `download_files.py`.
Alternatively, there is the version `topic_model_all-MiniLM-L6-v2_190_20n_8dim_safetensors` which is much smaller and is present in the directory, but the prediction results are much poorer. <br>
The BERTopic model used is one example, there are several. Just go to the ModelsContent subdirectory to see how many more there are.<br>
Or also use the models in the thread section on Models:

```bash
cd ../ThreadAnalysis/Models
```

By default, the top 5 labels are set, but just set the `num_classes` parameter with the number of topics desired.

Clearly the custom `predict_topic` function is being used, alternatively, one could use BERTopic's `transform' function [directly](https://maartengr.github.io/BERTopic/getting_started/serialization/serialization.html), without importing anything else.

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

Note: In the same `.ipynb` file there are other examples

### Safetensors and Hugging Face

If utilising safetensors or Hugging Face (see https://huggingface.co/D0men1c0) models use:
```python
from bertopic import BERTopic

# Load the BERTopic model
topic_model = BERTopic.load("D0men1c0/ISSR_Dark_Web_Merged_Models_Content_Thread")

# Define a sentence for prediction
sentence = ['recently closed Samsara market']

# Predict
results, _ = topic_model.transform(sentence)

# Get topic info
topic_model.get_topic_info(results[0])
```
because with safetensors Umap and Hdbscan are not reported, so the prediction must be made on embeddings and not on probabilities.

#### Example Results
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Topic</th>
      <th>Count</th>
      <th>Name</th>
      <th>CustomName</th>
      <th>Representation</th>
      <th>Representative_Docs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>113</td>
      <td>290</td>
      <td>113_samsara_market_samsara market_sam</td>
      <td>Samsara Market</td>
      <td>[samsara, market, samsara market, sam, dream, ...</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

## Summary of the Analysis

I recommend checking the `ShowResultsHTML/ShowFinalAnalysisBaselines` directory, which contains HTML files with all the results obtained so far for each baseline, including additional analyses, sentiment analysis, graphs, and examples of predictions.

## Datasets Used

The CrimeBB and ExtremeBB datasets (for more details, see the Citations section) were used, specifically the following data:
- **dread-2023-06-21** scraped from Dread, updated on 2023-06-21
- **extremebb-white-nations 2021-12-10** scraped from White Nations, updated on 2021-12-10
- **raidforums 2023-06-21** scraped from Raid Forums, updated on 2023-06-21

Dread is a popular forum on the Dark Web where users exchange opinions and reviews on various topics, including drug sales, seller reviews, password and bitcoin transactions, as well as passports and fake IDs.

ExtremeBB's "white-nations" dataset covers discussions related to conspiracies, climate change, elections, and racism, reflecting a broad range of socio-political issues.

RaidForums, on the other hand, focuses on leaked and cracked accounts from social media, games, Spotify, and other platforms, highlighting discussions around data breaches and unauthorized access.

For more details, here is a general tables structure:

<img src="Img/structure_tables.png" alt="structure_tables" width="500" height="600"/>

The following tables were analyzed to extract the topics:
- **Dread Posts:** Contains 290k records.
- **Dread Threads:** Contains 75k records.
- **White Nations Posts:** Contains 52k records.
- **Raid Forums Threads:** Contains 94k records

In addition, to make the model multimodal, as described in section [Repository Structure](#repository-structure), a public dataset from [Roboflow's Drug Detection Project](https://universe.roboflow.com/freakinggojo-er8b5/drug_detection_project/dataset/9) was used. <br>
This dataset includes approximately 3700 images, divided into training, testing, and validation sets, covering four topics: people, guns, drugs, and robbers.

## Summary of Work Done

The objective of this project was to analyze the evolution of language among Dark Web users using a series of Natural Language Processing (NLP) models. <br> Initially, each table in the dataset was examined, reducing the number of features and eliminating outliers and duplicates. The focus was specifically on two tables: threads and posts, analyzing the fields `thread` and `content` respectively.

To gain an overview of the topics discussed, the `thread` field was analyzed first. Various approaches such as TF-IDF, LDA, and Zero-shot classification were used, and a class was developed that combined Sentence Transformer with t-SNE, PCA, and k-means. However, these methods did not yield optimal results.<br> Therefore, the BERTopic library was employed, which, thanks to its modularity, enabled the analysis of text by applying different strategies including Sentence BERT, c-TFIDF, KeyBERTInspired, introducing diversity with custom classes, and using UMAP for dimensionality reduction and HDBSCAN for clustering. Subsequently, Zero-shot classification was applied to the topic names.

This comprehensive approach resulted in two baselines with Dread forums:

- One with 7 macro clusters (general topics)
- Another with 68 clusters (aiming to identify as many relevant topics as possible while avoiding micro clusters)

These results were validated using metrics and distribution graphs provided by BERTopic, which also allowed for the analysis of topic distribution over time.

To further validate the results from an accuracy metrics perspective, a LightGBM model was trained using the embeddings as the input features (X) and the topics identified by BERTopic as the target variable. This was done to validate the BERTopic results in a classification task.

Subsequently, the `content` field in the posts was analyzed to verify if the topics identified matched those in the threads, which they did, resulting in 121 clusters.

To build more robust models, it was decided to analyze two additional datasets with the same structure. 
The first dataset (**White Nations**) contained 31 topics derived from 52k data points, while the second dataset (**Raid Forums**) included 26 topics from 94k data points. <br>
The first dataset covered topics such as conspiracies, climate change, elections, and racism, whereas the second dataset focused on leaked and cracked accounts (social media, games, Spotify, etc.).

Finally, to achieve a more robust model, the four baselines found were combined: the one with 68 topics, the one with 121 topics, the one with 31 topics, and the one with 26 topics, resulting in a total of `173` topics (see https://huggingface.co/D0men1c0/ISSR_Dark_Web_Merged_Models_Content_White_Nations_Raid).

## Results

These are the results of the cluster analysis produced by BERT on both threads and content.<br> For further graphical analysis (distance between clusters, hierarchy, distribution in space), please consult the html directory: `ShowResultsHTML`.

**Dread Thread:**

| Metric                                  | Value (68 Topics) | Value (7 Topics) |
|-----------------------------------------|-------------------|------------------|
| Coherence Score                         | 0.57              | 0.40             |
| Silhouette Score                        | 0.50              | 0.51             |
| Davies Bouldin Score                    | 0.87              | 0.76             |
| Dos Score (Diversity Overlapped Score)  | 0.06              | 0.80             |
| Outliers                                | 0.30              | 0.42             |

**Dread Content:**

| Metric                                  | Value (121 Topics)|
|-----------------------------------------|-------------------|
| Coherence Score                         | 0.69              |
| Silhouette Score                        | 0.60              |
| Davies Bouldin Score                    | 0.46              |
| Dos Score (Diversity Overlapped Score)  | 0.24              |
| Outliers                                | 0.35              |

**Raid Forums Thread:**

| Metric                                  | Value (26 Topics)|
|-----------------------------------------|-------------------|
| Coherence Score                         | 0.51              |
| Silhouette Score                        | 0.62              |
| Davies Bouldin Score                    | 0.48              |
| Dos Score (Diversity Overlapped Score)  | 0.20              |
| Outliers                                | 0.38              |

**White Nations Content:**

| Metric                                  | Value (31 Topics)|
|-----------------------------------------|-------------------|
| Coherence Score                         | 0.46              |
| Silhouette Score                        | 0.60              |
| Davies Bouldin Score                    | 0.55              |
| Dos Score (Diversity Overlapped Score)  | 0.20              |
| Outliers                                | 0.32              |


There are many other graphs present representing topics and their distribution, for reasons of space only the `Dread Content` graph of the top 10 most frequent topics distributed over time will be shown

<div style="text-align: center;">
  <img src="Img/Content/120TimeSeries.png" alt="Distribution Topic" width="800" height="400"/>
</div>

### Cluster Validation using LightGBM and LSTM

These are the results obtained by LightGBM using the embedding and classes obtained by BERT for the thread field in a classification task:

**Dread Thread:**

| Metric              | Value (68 Topics) | Value (7 Topics) |
|---------------------|-------------------|------------------|
| Accuracy            | 0.89              | 0.96             |
| Precision           | 0.91              | 0.96             |
| Recall              | 0.84              | 0.95             |
| F1 Score            | 0.87              | 0.96             |

<br>

| ![Loss Train Val 68 Topics](Img/LossLightGBM68.png) | ![Loss Train Val 7 Topics](Img/LossLightGBM7.png) |
|:------------------------------------------------------:|:---------------------------------------------------:|
| Loss Train Val 68 Topics                             | Loss Train Val 7 Topics                            |

<br>Instead, a neural network using two inputs was chosen for `Dread Content`: a timestamp treated as LSTM and another input with embeddings obtained from BERT. In this way it is also possible to make predictions based on time and understand the evolution of the language.

**Dread Content:**

| Metric              | Value (121 Topics)|
|---------------------|-------------------|
| Accuracy            | 0.87              |
| Precision           | 0.87              |
| Recall              | 0.86              |
| F1 Score            | 0.86              |

<br>

| ![Loss Train Val 121 Topics](Img/LSTMLossContent.png) | ![Accuracy Train Val 121 Topics](Img/LSTMAccuracyContent.png) |
|:------------------------------------------------------:|:---------------------------------------------------------------:|
| Loss Train Val 121 Topics                             | Accuracy Train Val 121 Topics                                  |

## Multimodal Model

In addition to the work done so far, a multimodal approach has been developed to further enhance the model's capabilities. <br>
As described previously, this involves integrating visual data with the existing textual baseline models. <br>
Although the current visual model is based on a public dataset rather than images from the dark web, it serves as a proof of concept for how a multimodal model can be effectively utilized.

A BERTopic model was trained using images from the dataset provided by [Roboflow's Drug Detection Project](https://universe.roboflow.com/freakinggojo-er8b5/drug_detection_project/dataset/9). <br>
This dataset includes approximately 3700 images divided into training, testing, and validation sets, covering four topics: people, guns, drugs, and robbers. The approach involved:

1. **Embedding Extraction**: Using models to generate embeddings from the images, which were then associated with captions.
2. **Visual Model Training**: Training a visual model to obtain topic representations from these embeddings.

The model was trained using the training and testing sets and was evaluated on the validation set. <br>
It successfully identified the four relevant topics: drugs, guns, people, and robbers.

### Advantages of the Multimodal Model

The multimodal model provides several significant advantages:

- **Enhanced Topic Detection**: The integration of visual data allows for more precise identification of topics. For instance, the model can differentiate between specific types of drugs, such as yellow heroin versus brown heroin, based on visual cues.
- **Improved Context Understanding**: By combining text and image data, the model can provide a deeper understanding of the context. This is particularly useful for identifying and categorizing nuanced topics.
- **Versatility**: The model's ability to analyze both text and images means it can adapt to various types of content and contexts.<br> This capability will be crucial when images directly from the dark web become available, enabling the combination of baseline text models with visual data to create a robust multimodal model.
- **Predictive Accuracy**: By leveraging visual data, the model can achieve more accurate predictions and insights, which are beneficial for applications that require detailed topic analysis and classification.

This demonstration underscores the potential of integrating visual data with text-based models and sets the stage for future advancements when dark web images can be incorporated into the model.

## Future work

- [X] **Merge Baselines**: The next steps involve merging the baselines obtained from the Thread and Content sections into a single model to integrate both representations, with Thread providing general topics and Content extending them.
- [X] **Validation**: Validate the results using clustering metrics and machine learning models. Given the data volume, LightGBM may not be sufficient, so exploring neural networks could be beneficial.
- [X] **Topic Refinement**: Conduct a final review of the topics, potentially integrating specific expressions that need to be highlighted (e.g., a particular abbreviation for a drug that wasn't identified).
- [X] **Integration with GenAI**: Incorporate Generative AI to better explain and refine the topic labels. Mistral can provide more detailed and contextually relevant labels, enhancing the interpretability of the topics.
- [X] **Deployment on HuggingFace**: Push both the "general" model with 7 topics and the more specific model resulting from merging the Thread and Content sub-models to HuggingFace.
- [X] **Temporal Validation**: If time permits, validate the results over time using LSTM (currently done with BERT) to ensure temporal consistency and gain additional insights.
- [X] **Dataset Integration**: Explore the integration of additional datasets to make the merged model more robust and comprehensive. This would involve aligning different datasets to ensure consistency and leveraging diverse data sources to cover more topics and nuances.
- [X] **Sentiment Analysis**: Implement sentiment analysis on processed documents to evaluate the prevailing sentiment within each cluster, providing deeper insights into the emotional tone and audience perceptions related to specific topics.
- [X] **Multimodal Model**: Finally, consider making the model multimodal by incorporating both text and images. This would require more in-depth development.


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