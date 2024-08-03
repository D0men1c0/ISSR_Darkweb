import matplotlib.pyplot as plt
from bertopic import BERTopic
import pandas as pd
import seaborn as sns
import math
import re
import numpy as np
from wordcloud import WordCloud
from collections import defaultdict
import networkx as nx
from typing import Union
from tqdm import tqdm


def plot_topic_distribution(df: pd.DataFrame, figsize: tuple = (8, 6)) -> None:
    """
    Create a horizontal bar plot for topic distribution based on counts.
    :param df: DataFrame containing 'Count' and 'Custom_Name_GenAI' columns.
    """
    plt.figure(figsize=figsize)
    sns.barplot(x='Count', y='Custom_Name_GenAI', data=df, orient='h',
                order=df.sort_values('Count', ascending=False)['Custom_Name_GenAI'])
    
    plt.title('Topic Distribution')
    plt.xlabel('Count')
    plt.ylabel('Custom Name Gen AI')
    plt.tight_layout()
    plt.show()

def clean_probability_string(prob_string: str) -> list:
    """
    Clean and convert a probability string into a list of numbers.
    :param prob_string: String containing probabilities.
    :return: List of probabilities as floats.
    """
    # Remove unwanted characters such as '\n' and add commas between numbers
    cleaned_str = re.sub(r'\s+', ',', prob_string.strip())

    # Add a comma between numbers where it's missing
    cleaned_str = re.sub(r'(\d)\s+(\d)', r'\1,\2', cleaned_str)

    return eval(f'[{cleaned_str}]')

def extract_max_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the maximum probability for each record in the DataFrame.
    :param df: DataFrame containing 'Probability' column with probability strings.
    :return: DataFrame with an additional 'Max_Probability' column.
    """
    # Clean and convert the probability strings into lists of numbers
    df['Probability'] = df['Probability'].apply(clean_probability_string)

    # Calculate the maximum probability for each record
    df['Max_Probability'] = df['Probability'].apply(np.max)
    return df

def plot_topic_percentage_distribution(df: pd.DataFrame, figsize: tuple = (8, 6)) -> None:
    """
    Create a pie chart for the percentage distribution of topics based on counts.
    :param df: DataFrame containing 'Count' and 'Topic' columns.
    """
    plt.figure(figsize=figsize)
    plt.pie(df['Count'], labels=df['Topic'], autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired(range(len(df))))
    
    plt.title('Topic Percentage Distribution')
    plt.tight_layout()
    plt.show()

def plot_avg_prob_or_freq(df: pd.DataFrame, metric: str = 'Probability', figsize: tuple = (10, 6)) -> None:
    """
    Create a bar plot of average probability or average frequency by topic.
    :param df: DataFrame containing 'Topic' and either 'Probability' or 'Frequency' columns.
    :param metric: Metric to plot; should be 'Probability' or 'Frequency'.
    """
    if metric not in ['Probability', 'Frequency', 'Max_Probability']:
        raise ValueError("Metric should be 'Frequency' or 'Probability' or 'Max_Probability'")
    
    avg_metric = df.groupby('Topic')[metric].mean().reset_index()
    avg_metric = avg_metric.sort_values(metric)
    
    plt.figure(figsize=figsize)
    
    # Use seaborn for 'Probability' or matplotlib for 'Frequency'
    if metric == 'Probability' or metric == 'Max_Probability':
        sns.barplot(x='Topic', y=metric, data=avg_metric, order=avg_metric['Topic'])
    else:
        plt.bar(avg_metric['Topic'], avg_metric[metric], color='skyblue')
    
    plt.title(f'Average {metric} by Topic')
    plt.xlabel('Topic')
    plt.ylabel(f'Average {metric}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_boxplot(df: pd.DataFrame, metric: str = 'Probability', figsize: tuple = (10, 6)) -> None:
    """
    Create a box plot for either frequency or probability by topic.
    :param df: DataFrame with 'Topic' and 'Frequency' or 'Probability'.
    :param metric: 'Frequency' or 'Probability' to plot.
    """
    if metric not in ['Probability', 'Frequency', 'Max_Probability']:
        raise ValueError("Metric should be 'Frequency' or 'Probability' or 'Max_Probability'")
    
    plt.figure(figsize=figsize)
    
    sns.boxplot(data=df, x='Topic', y=metric, palette='Set2')
    plt.title(f'Box Plot of {metric} by Topic')
    plt.xlabel('Topic')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_probability_distribution(df: pd.DataFrame, feature: str = 'Probability', figsize: tuple = (10, 6)) -> None:
    """
    Create a histogram with KDE for the distribution of document probabilities.
    :param df: DataFrame containing a 'Probability' column.
    """
    plt.figure(figsize=figsize)
    sns.histplot(df[feature], bins=20, kde=True, color='skyblue')
    
    plt.title('Distribution of Document Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Number of Documents')
    plt.tight_layout()
    plt.show()

def create_wordclouds(data: Union[BERTopic, pd.DataFrame], num_topics: int, cols: int = 3, is_model: bool = True, width: int = 1200, height: int = 500) -> None:
    """
    Create word clouds for each topic based on either a BERTopic model or a DataFrame.
    :param data: BERTopic model or DataFrame containing 'Topic' and 'Document' columns.
    :param num_topics: Number of topics to create word clouds.
    :param cols: Number of columns in the plot.
    :param is_model: Boolean flag indicating if the data is a BERTopic model (True) or a DataFrame (False).
    """
    rows = math.ceil(num_topics / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5), sharex=True, sharey=True)
    axes = axes.flatten()
    
    if is_model:
        # Using BERTopic model
        for i in range(num_topics):
            ax = axes[i]
            text = {word: value for word, value in data.get_topic(i)}
            wc = WordCloud(background_color="black", max_words=1000, width=width, height=height)
            wc.generate_from_frequencies(text)
            
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f'Topic {i}', fontsize=16)
    else:
        # Using DataFrame
        text_by_topic = data.groupby('Topic')['Document'].apply(lambda x: ' '.join(x)).reset_index()
        
        for i in range(num_topics):
            if i < len(text_by_topic):
                ax = axes[i]
                document_text = text_by_topic.loc[text_by_topic['Topic'] == i, 'Document'].values[0]
                wc = WordCloud(background_color="black", max_words=1000, width=width, height=height)
                wc.generate(document_text)
                
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                ax.set_title(f'Topic {i}', fontsize=16)
            else:
                axes[i].axis('off')
    
    # Hide any empty subplots
    for j in range(num_topics, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def create_topic_network_graphs(df: pd.DataFrame, window_size: int = 5, min_occurrences: int = 1, cols: int = 2, max_words: int = 100, width: int = 24, height: int = 8) -> None:
    """
    Create network graphs for each topic visualizing relationships between topics and documents.
    :param df: DataFrame containing 'Topic', 'Document', and 'BERTopic_Name' columns.
    :param window_size: The size of the sliding window to calculate co-occurrences.
    :param min_occurrences: Minimum co-occurrence count to include an edge in the graph.
    :param cols: Number of columns in the subplot grid.
    :param max_words: Maximum number of words to include in the graph for each topic.
    """
    topics = df['Topic'].unique()
    num_topics = len(topics)
    rows = math.ceil(num_topics / cols)
    
    # Increase the figure size for larger graphs
    fig, axes = plt.subplots(rows, cols, figsize=(width, rows * height), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, topic in tqdm(enumerate(topics), total=num_topics, desc='Processing Topics'):
        ax = axes[idx]
        
        # Create a subnetwork for the current topic
        G = nx.Graph()
        
        # Filter documents related to the current topic
        topic_docs = df[df['Topic'] == topic]
        
        # Get the topic name
        topic_name = df.loc[df['Topic'] == topic, 'BERTopic_Name'].values[0]
        
        # Tokenize the documents for the current topic
        text = ' '.join(topic_docs['Document'].tolist()).lower()
        tokens = text.split()
        
        # Create a set of unique words including the topic
        top_words = set([topic_name]).union(set(tokens))
        
        # Limit the number of words
        if len(top_words) > max_words:
            word_freq = defaultdict(int)
            for word in tokens:
                if word in top_words:
                    word_freq[word] += 1
            top_words = set(sorted(word_freq, key=word_freq.get, reverse=True)[:max_words])
        
        # Calculate co-occurrences
        co_occurrence = defaultdict(int)
        
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i:i + window_size]
            for i, word1 in enumerate(window):
                if word1 in top_words:
                    for j, word2 in enumerate(window):
                        if i != j and word2 in top_words:
                            pair = tuple(sorted([word1, word2]))
                            co_occurrence[pair] += 1
        
        # Add nodes and edges to the graph
        for (word1, word2), count in co_occurrence.items():
            if count >= min_occurrences:
                G.add_edge(word1, word2, weight=count)
        
        # Draw the network graph
        pos = nx.spring_layout(G, k=0.2, iterations=50)  # Adjust layout parameters
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='skyblue', ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=2.0, edge_color=weights, edge_cmap=plt.cm.Blues, ax=ax)
        nx.draw_networkx_labels(G, pos, labels={n: n for n in G.nodes()}, font_size=12, ax=ax)
        
        ax.set_title(f'Topic: {topic_name}', fontsize=18)
    
    # Hide any empty subplots
    for j in range(num_topics, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()