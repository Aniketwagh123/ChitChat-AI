##############################################################################################################################################
# Preprocessing of text 
##############################################################################################################################################
import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('punkt')

# Download the Reuters dataset from nltk
nltk.download('reuters')

# Load the Reuters dataset
from nltk.corpus import reuters
documents = reuters.fileids()
data = {'excerpt': [reuters.raw(file_id) for file_id in documents]}
train_data = pd.DataFrame(data).head(10)

# Display the first few rows of the dataset
train_data.head()

# Define a function to remove special characters
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-Z0-9\s]' if remove_digits else r'[a-zA-Z\s]'
    text = re.sub(pattern, "", text)
    return text

# Apply the function to the 'excerpt' column and convert to lowercase
train_data['excerpt_lower'] = train_data['excerpt'].apply(lambda x: remove_special_characters(x.lower()))

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Apply stemming to the 'excerpt_lower' column
train_data['excerpt_stemmed'] = train_data['excerpt_lower'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))

# Tokenization
train_data['tokens'] = train_data['excerpt_lower'].apply(nltk.word_tokenize)

# Stopword Removal
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(text):
    """Custom function to remove the stopwords"""
    return " ".join([word for word in text.split() if word.lower() not in STOPWORDS])

# Apply stopword removal to the 'excerpt_lower' column
train_data['excerpt_no_stopwords'] = train_data['excerpt_lower'].apply(remove_stopwords)

# Display the processed data
train_data.head()










##############################################################################################################################################
# Morphological analysis
##############################################################################################################################################
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import reuters

nltk.download('punkt')  # Download the necessary data for NLTK
nltk.download('reuters')  # Download the Reuters corpus

def separate_root_and_morphemes(word):
    stemmer = PorterStemmer()
    root = stemmer.stem(word)
    morphemes = word[len(root):]  # Remove the rest from the original words to get the morphemes
    return root, morphemes

word = "played"
root, morphemes = separate_root_and_morphemes(word)
print("Original word:", word)
print("Root:", root)
print("Morphemes:", morphemes)
print("\n")

import spacy

# Load the language model with the Reuters dataset
nlp = spacy.load("en_core_web_sm", exclude=["ner"])
reuters_texts = reuters.raw()
nlp.max_length = len(reuters_texts)

# Text to be analyzed
text = "Running is my favorite activity."

# Process the text with Spacy
doc = nlp(text)

# Iterate through tokens and access morphological attributes
for token in doc:
    print(f"Token: {token.text}")
    print(f"Lemma: {token.lemma_}")
    print(f"Part of Speech (POS) tag: {token.pos_}")
    print(f"Morphological Features: {token.morph}")
    print("\n")










##############################################################################################################################################
# N-Gram
##############################################################################################################################################
import nltk
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import reuters
from nltk.corpus import stopwords

# Download the Reuters dataset
nltk.download('reuters')
nltk.download('punkt')
nltk.download('stopwords')

# Load documents from the Reuters dataset
documents = reuters.fileids()
corpus = [reuters.raw(doc_id) for doc_id in documents]

# Concatenate all documents into a single string
text = " ".join(corpus)

# Tokenize the text and remove stopwords
words = nltk.word_tokenize(text)
stop_words = set(stopwords.words("english"))
filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

# Define N for N-grams
n_values = [1, 2, 3]  # Unigram, Bigram, and Trigram

# Create and analyze N-grams
for n in n_values:
    n_grams = list(ngrams(filtered_words, n))
    n_gram_counts = Counter(n_grams)

    # Display the top N most common N-grams
    print(f"{n}-grams analysis:")
    for gram, count in n_gram_counts.most_common(10):
        print(f"{gram}: {count}")
    print("\n")







##############################################################################################################################################
# Pos tagging and chunking
##############################################################################################################################################
import nltk
from nltk import pos_tag
from nltk.chunk import RegexpParser
from nltk import word_tokenize
from nltk import Tree

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Sample text
text = "The quick brown fox jumps over the lazy dog"

# Tokenize the text into words
words = word_tokenize(text)

# Perform POS tagging
pos_tags = pos_tag(words)

# Define a simple grammar for NP (noun phrase) chunking
grammar = r"""
    NP: {<DT>?<JJ>*<NN>}  # Chunk NP (optional determiner + adjectives + noun)
"""

# Create a chunk parser with the defined grammar
chunk_parser = RegexpParser(grammar)

# Apply the chunk parser to the POS-tagged words
chunked_result = chunk_parser.parse(pos_tags)

# Print the POS tags and the chunked result
print("POS Tags:", pos_tags)
print("Chunked Result:", chunked_result)

# Convert the chunked result to a Tree
tree = Tree.fromstring(str(chunked_result))

# Pretty print the tree in the terminal
tree.pretty_print()












##############################################################################################################################################
# Name entity Recognition 
##############################################################################################################################################
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
import spacy
from tabulate import tabulate

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Example text for Named Entity Recognition
text = "On December 12, 2022, OpenAI released GPT-4. The headquarters of OpenAI is located in San Francisco. " \
       "John Doe, born on January 25, 1980, is the CEO of the company."


# Process the text with spaCy NLP pipeline
doc = nlp(text)

# Extract named entities and their labels
entities = [(ent.text, ent.label_) for ent in doc.ents]

# Create a table with tabulate
table_headers = ["Entity", "Label"]
table_data = tabulate(entities, headers=table_headers, tablefmt="grid")

# Print the table
print("Named Entities:")
print(table_data)









##############################################################################################################################################
# DVP
##############################################################################################################################################
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import squarify  # for treemap

# Load Tips dataset
tips = sns.load_dataset('tips')

# Visualize 1: Pair Plot
sns.pairplot(tips, hue='sex', markers=["o", "s"], height=3)
plt.title('Pair Plot of Tips Dataset')
plt.show()

# Visualize 2: Violin Plot with Split
plt.figure(figsize=(10, 6))
sns.violinplot(x='day', y='total_bill', hue='sex', split=True, data=tips)
plt.title('Violin Plot of Total Bill for Each Day')
plt.show()

# Visualize 3: Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='day', y='total_bill', hue='sex', data=tips)
plt.title('Box Plot of Total Bill for Each Day')
plt.show()

# Visualize 4: Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_bill', y='tip', hue='sex', data=tips)
plt.title('Scatter Plot of Total Bill vs. Tip')
plt.show()

# Visualize 5: Count Plot
plt.figure(figsize=(8, 6))
sns.countplot(x='day', hue='sex', data=tips)
plt.title('Count of Each Gender for Each Day')
plt.show()

# Visualize 6: Histogram with KDE
plt.figure(figsize=(10, 6))
sns.histplot(data=tips, x='total_bill', hue='sex', kde=True, bins=20)
plt.title('Total Bill Histogram with KDE')
plt.show()

# Visualize 7: Swarm Plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='day', y='total_bill', hue='sex', data=tips)
plt.title('Swarm Plot of Total Bill for Each Day')
plt.show()

# Visualize 8: Pie Chart
plt.figure(figsize=(8, 8))
tips_by_sex = tips['sex'].value_counts()
plt.pie(tips_by_sex, labels=tips_by_sex.index, autopct='%1.1f%%', startangle=90)
plt.title('Pie Chart of Gender Distribution')
plt.show()

# Visualize 9: Sunburst Plot
fig = px.sunburst(tips, path=['sex', 'day', 'time'], title='Sunburst Plot of Tips Dataset')
fig.show()


