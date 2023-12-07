##############################################################################################################################################
# Preprocessing of text 

# Text preprocessing is a crucial step in natural language processing (NLP) that involves cleaning and transforming raw text data into a format that can be easily understood and analyzed by machine learning algorithms. The goal of text preprocessing is to enhance the quality of the data and improve the performance of NLP models. Here is a comprehensive theory for the preprocessing of text in NLP:


# Tokenization:
# Definition: Tokenization is the process of breaking down a text into smaller units, such as words or subwords, known as tokens.
# Purpose: Tokenization is essential for converting a continuous stream of text into discrete units, enabling further analysis and processing. It forms the foundational step for various NLP tasks.

# Lowercasing:
# Definition: Converting all characters in the text to lowercase.
# Purpose: Lowercasing ensures uniformity and helps in reducing the dimensionality of the data. It prevents the model from treating words in different cases as distinct entities.

# Removing Stop Words:
# Definition: Stop words are common words (e.g., "the," "and," "is") that do not contribute significantly to the meaning of a text.
# Purpose: Removing stop words reduces noise and focuses on the essential content of the text. This step is crucial for improving the efficiency of NLP models and saving computational resources.

# Removing Special Characters and Punctuation:
# Definition: Eliminating non-alphanumeric characters and punctuation marks from the text.
# Purpose: Removing special characters helps in cleaning the text and avoiding unnecessary complexity. It ensures that the model focuses on the semantic content of the text.

# Stemming and Lemmatization:
# Definition: Stemming involves reducing words to their root or base form, while lemmatization involves converting words to their dictionary form.
# Purpose: Stemming and lemmatization help in standardizing words, reducing inflections, and improving the coherence of the text data. This aids in feature extraction and model generalization.

# Handling Contractions:
# Definition: Expanding contractions (e.g., "don't" to "do not").
# Purpose: Expanding contractions ensures consistency in representation and avoids potential misinterpretations, as models may treat contracted forms differently from their expanded counterparts.

# Removing HTML Tags and URLs:
# Definition: Eliminating HTML tags and hyperlinks from text data.
# Purpose: In the case of web data, removing HTML tags and URLs is essential for obtaining clean and meaningful text. It prevents these elements from being misinterpreted or influencing the model's understanding.

# Spell Checking and Correction:
# Definition: Identifying and correcting spelling errors in the text.
# Purpose: Spell checking enhances the quality of the text data by addressing typos and inaccuracies. It ensures that the model is trained on accurate and consistent information.

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

# Morphological analysis is a linguistic approach that involves the study and analysis of the structure and formation of words, including their smallest units, known as morphemes. Morphology is one of the core branches of linguistics that explores the internal structure of words and how they are formed. Here are some key theoretical aspects of morphological analysis:

# Morpheme:
# Definition: A morpheme is the smallest grammatical unit in a language that carries meaning. It can be a word or a part of a word.
# Types: Morphemes are classified into two main types - free morphemes and bound morphemes. Free morphemes can stand alone as meaningful words (e.g., "dog," "run"), while bound morphemes are attached to free morphemes to modify their meaning (e.g., "-s" for plural, "-ed" for past tense).

# Word Formation Processes:
# Derivation: This process involves adding affixes (prefixes, suffixes, infixes) to a base word to create a new word with a different meaning or grammatical category (e.g., "happy" to "unhappy," "friend" to "friendship").
# Inflection: This process involves adding inflections to a word to indicate grammatical information such as tense, number, gender, or case (e.g., "run" to "runs," "cat" to "cats").

# Lexical Morphology vs. Inflectional Morphology:
# Lexical Morphology: Deals with the formation and structure of words in a language, focusing on how new words are created and the rules governing their construction.
# Inflectional Morphology: Concerned with the modification of words to convey grammatical information without changing their underlying meaning. It includes processes like tense, case, and number.

# Morphological Parsing:
# Parsing: The process of breaking down a word into its constituent morphemes and understanding the grammatical and semantic information conveyed by each morpheme.
##############################################################################################################################################
!pip install nltk
!pip install spacy
! python -m spacy download en_core_web_smimport nltk
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

# N-grams in Natural Language Processing (NLP):

# Definition of N-grams:
# N-grams are contiguous sequences of n items from a given sample of text or speech. In the context of natural language processing (NLP), these items are often words, but they can also be characters or other linguistic units.

# Types of N-grams:
# Let's consider a sentence and explore unigrams, bigrams, and trigrams for that sentence. We'll use the sentence: "The quick brown fox jumped over the lazy dog."

# Unigrams (1-grams):
# Unigrams are single words considered in isolation.
# Example: "The", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog".

#Bigrams (2-grams):
# Bigrams are pairs of consecutive words.
# Example: "The quick", "quick brown", "brown fox", "fox jumped", "jumped over", "over the", "the lazy", "lazy dog".

#Trigrams (3-grams):
# Trigrams are triplets of consecutive words.
# Example: "The quick brown", "quick brown fox", "brown fox jumped", "fox jumped over", "jumped over the", "over the lazy", "the lazy dog".

# Applications of N-grams in NLP:
# Language Modeling: N-grams are fundamental in language modeling, helping predict the likelihood of a word given its context. For instance, a trigram model could estimate the probability of the next word based on the previous two words.

# Speech Recognition: N-grams are used in speech recognition systems to model and recognize spoken language patterns. This aids in improving the accuracy of transcriptions.

# Machine Translation: In machine translation systems, N-grams can be employed to identify common phrases and improve the translation of these phrases.

# Text Generation: N-grams can be used to generate coherent and contextually relevant text. By predicting the next word based on preceding ones, systems can produce human-like language.

# Spell Checking and Correction: N-grams are utilized in spell-checking algorithms to identify and correct misspelled words by considering the likelihood of a sequence of words.

# Named Entity Recognition (NER): Identifying entities (names, locations, organizations, etc.) in text can benefit from the context provided by N-grams.

# Sentiment Analysis: Analyzing the sentiment of a piece of text can be enhanced by considering the context of word sequences, and N-grams play a role in capturing this context.

##############################################################################################################################################
import nltk
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import stopwords


nltk.download('punkt')
nltk.download('stopwords')

text = "Thank you so much for your hepl, I really appreatiate your help. Excuse me do you know what time its.sorry for not inviting you.i relly your wa realyy live your gardan?"*100

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

# ///// Part-of-speech (POS) tagging and chunking are essential tasks in natural language processing (NLP) that involve analyzing and annotating the linguistic structure of a sentence.


# // Part-of-Speech (POS) Tagging:
# POS tagging is the process of assigning a specific part-of-speech tag (e.g., noun, verb, adjective, etc.) to each word in a sentence. This task is crucial for understanding the grammatical structure of a sentence and extracting meaningful information from it. The theory behind POS tagging involves the use of statistical models, rule-based approaches, or a combination of both.

# / Statistical Models: Many POS tagging systems are based on statistical models, such as Hidden Markov Models (HMMs) or more recently, deep learning models like Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), and Transformer-based models like BERT. These models learn patterns from large annotated corpora and use this knowledge to predict the most likely POS tag for each word in a given context.

# / Rule-based Approaches: Rule-based POS tagging relies on a set of hand-crafted rules that define the POS tags based on syntactic and morphological features of words. These rules are often linguistic in nature and can be highly effective for languages with clear and consistent grammatical rules.

# / Hybrid Approaches: Some POS tagging systems combine statistical models with rule-based approaches to achieve better accuracy and robustness. This hybrid approach leverages the strengths of both methods to handle a wide range of linguistic phenomena.

# // Chunking:
# Chunking, also known as shallow parsing, involves identifying and grouping words into syntactically related sets called "chunks." These chunks typically correspond to phrases such as noun phrases (NP), verb phrases (VP), etc. Chunking is an intermediate step between POS tagging and full syntactic parsing.

# / Chunking Patterns: Chunking often relies on identifying patterns of POS tags in a sequence. For example, a noun phrase might consist of a determiner, followed by zero or more adjectives, and then a noun. These patterns can be learned from annotated training data or defined through rule-based approaches.

# / Chunking Algorithms: Various algorithms are employed for chunking, including regular expressions, rule-based systems, and machine learning approaches. Machine learning models, such as Conditional Random Fields (CRFs) or Maximum Entropy Markov Models (MEMMs), can learn complex patterns and dependencies to improve chunking accuracy.

# / Applications: Chunking is valuable for information extraction, named entity recognition, and syntactic analysis. It helps in identifying and extracting meaningful chunks of information from text, enabling higher-level NLP tasks.

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

# Named Entity Recognition (NER):
# Named Entity Recognition (NER) is a crucial task in natural language processing that involves identifying and classifying named entities in text into predefined categories such as persons, organizations, locations, dates, and more. The goal is to extract structured information from unstructured text.

# // diffrent Named Entities:
# Persons: Individuals, including names of people.
# Organizations: Companies, institutions, or any organized group with a distinct name.
# Locations: Places, such as cities, countries, or geographical regions.
# Dates: Temporal expressions, including specific dates or date ranges.
# Numerical Entities: Any numeric values, including percentages, money, or quantities.
# Miscellaneous: Other specific categories depending on the application, such as product names, medical terms, etc.

# // Approaches to NER:
# Rule-Based Systems: Rule-based approaches use predefined linguistic rules to identify named entities based on patterns, grammatical structures, and context.
# Statistical Models: Machine learning models, including Conditional Random Fields (CRFs), Hidden Markov Models (HMMs), and more recently, deep learning models like Recurrent Neural Networks (RNNs) and Transformer-based models, can learn patterns from annotated data to predict named entities.
# Hybrid Approaches: Combining rule-based systems with statistical models often leads to more robust and accurate NER systems.

# Features and Context:
# NER models rely on various features, including word embeddings, part-of-speech tags, syntactic features, and contextual information, to capture the nuances of named entities in different contexts.
# Contextual embeddings, such as those generated by models like BERT, have proven effective in capturing the contextual dependencies crucial for accurate NER.

# // Challenges:
# Ambiguity: Named entities can be ambiguous, and context is crucial for disambiguation. For example, "Apple" could refer to the company or the fruit.
# New Entities: NER systems should be adaptive to recognize new entities or entities not present in the training data.
# Multilingual NER: The challenges increase in multilingual scenarios where languages have different structures and entity types.

# // Applications:
# NER is fundamental for various downstream applications, including information extraction, question answering, summarization, and more.
# In the biomedical domain, BioNER focuses on recognizing entities related to genes, proteins, diseases, etc.
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

# // Visualization Techniques in Python:
# Data visualization is a critical aspect of data analysis and interpretation. Python provides several powerful libraries for creating a wide range of visualizations. Here's an overview of visualization techniques commonly used in Python:

# // Choosing the Right Visualization:
# Selecting the appropriate visualization technique depends on the nature of the data and the insights you want to convey. Common chart types include line charts for time-series data, bar charts for categorical data, scatter plots for relationships, and histograms for distributions.
# / Color Theory:Understanding color theory is crucial for creating effective visualizations. Consideration of color schemes, contrasts, and the use of color to convey information is essential. Libraries like Seaborn and Plotly often provide pre-defined color palettes.
# / Interactivity:Interactive visualizations enhance the user's ability to explore data. Libraries like Plotly and Bokeh allow the creation of interactive plots with features like tooltips, zooming, and panning.
# / Annotation and Labeling: Adding labels, annotations, and titles to plots is essential for conveying information clearly. Matplotlib and Seaborn provide functionalities to annotate and label various elements of a plot.
# / Customization:Customizing visualizations is often necessary to match specific design requirements. Matplotlib, Seaborn, and Plotly allow users to customize colors, styles, and other visual elements.


# /// Diffrent charts
# 1. Pair Plot:
# Description: A pair plot is a matrix of scatterplots for a set of variables, allowing you to visualize relationships between variables and distributions of individual variables.
# Theory: The pair plot shows scatterplots for each pair of numerical features in the dataset, colored by the 'sex' variable. The diagonal contains histograms for each variable.
# 2. Violin Plot with Split:
# Description: Violin plots display the distribution of a numeric variable for different categories. Splitting allows comparing distributions side by side.
# Theory: This plot shows the distribution of 'total_bill' for each day, split by 'sex.' The width of the violin represents the density of data points at different values.
# 3. Box Plot:
# Description: Box plots display the distribution of a numeric variable, showing quartiles, median, and potential outliers.
# Theory: This box plot visualizes the distribution of 'total_bill' for each day, with separate boxes for each gender.
# 4. Scatter Plot:
# Description: Scatter plots show the relationship between two numeric variables with points on a two-dimensional plane.
# Theory: The scatter plot displays 'total_bill' on the x-axis, 'tip' on the y-axis, and differentiates points by 'sex.'
# 5. Count Plot:
# Description: Count plots display the count of observations in each category of a categorical variable.
# Theory: This count plot shows the distribution of gender for each day, providing a count of occurrences.
# 6. Histogram with KDE:
# Description: Histograms display the distribution of a single variable, and KDE (Kernel Density Estimation) adds a smooth curve.
# Theory: This histogram with KDE displays the distribution of 'total_bill' for each gender, providing insights into the data's density.
# 7. Swarm Plot:
# Description: Swarm plots show individual data points along an axis, avoiding overlap.
# Theory: This swarm plot displays 'total_bill' for each day, with points differentiated by 'sex.'
# 8. Pie Chart:
# Description: Pie charts represent parts of a whole and are suitable for categorical data.
# Theory: The pie chart shows the distribution of gender in the 'tips' dataset.
# 9. Sunburst Plot:
# Description: Sunburst plots represent hierarchical data in a radial layout.
# Theory: This sunburst plot visualizes the hierarchy of 'sex,' 'day,' and 'time' in the 'tips' dataset.
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
