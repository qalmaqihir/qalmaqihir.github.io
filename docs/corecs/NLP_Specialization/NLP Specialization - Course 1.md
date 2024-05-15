
# NLP Specialization - Course 1

# Natural Language Processing With Classification and Vector Spaces
Note 2023-12-04T07.41.32

========================
## Week 1 Sentiment Analysis with  Logistic Regression

Learn to extract features from text into numerical vectors, then build a binary classifier for tweets using a logistic regression!  

**Learning Objectives**

    Sentiment analysis
    Logistic regression
    Data pre-processing
    Calculating word frequencies
    Feature extraction
    Vocabulary creation
    Supervised learning

### Supervised ML & Sentiment Analysis

#### Conceptual Framework

In supervised machine learning, we employ a conceptual framework involving features (X), labels (Y), a prediction function with parameters, and a cost function.

The prediction function takes features (X) and produces an output (Y hat), which is compared to the actual labels (Y) in the cost function. The goal is to minimize this cost function by updating the parameters of the prediction function.

#### Application to Sentiment Analysis

As an example, let's consider sentiment analysis on tweets. The features (X) are extracted from the tweets and converted into numerical representations. The model is then trained on these features, allowing us to predict sentiments in new tweets.

---

### Vocabulary and Feature Extraction

#### Vocabulary

The vocabulary is a crucial component for converting textual data into a numerical format. It consists of unique words derived from the dataset of tweets.

#### Sparse Representation

The sparse representation involves assigning 1 if a word appears in a tweet and 0 otherwise. This results in a sparse vector of length |Vocabulary|, where each element indicates the presence or absence of a specific word.

#### Linear Regression Model Challenges

The linear regression model needs to learn parameters proportional to the vocabulary size (n + 1 parameters). This can lead to challenges, including large training and prediction times.

---

### Negative and Positive Frequencies

#### Word Count for Model Features

To enhance our model, we can generate counts indicating how frequently a word appears in positive and negative classes. This information can be derived from a corpus, leading to the creation of the vocabulary and the frequency counts for each class.

---

### Feature Extraction with Frequencies

#### Feature Extraction

Consider a feature vector $\(X_m\)$ defined as:

$$\[X_m = [1, \sum_{w} \text{Freqs}(w, 1), \sum_{w} \text{Freqs}(w, 0)]\]$$

Here, \(w\) represents words, and the sums are over the vocabulary. The feature vector captures the presence of words in both positive (1) and negative (0) classes.

---

### Preprocessing - Stemming and Stop Words

#### Elimination of Irrelevant Words

Preprocessing involves eliminating stop words, punctuation, handles, and URLs from tweets. Stemming reduces words to their base stems, such as transforming "tuning," "tuned," and "tune" to "Tun." Normalization, including converting all words to lowercase, ensures uniformity.

#### Tweet Vectorization

Each preprocessed tweet is converted into a vector, and these vectors are organized into a matrix for further analysis.

---

### Overview of Logistic Regression

#### Sigmoid Function

In logistic regression, the sigmoid function replaces the prediction function. It outputs values between 0 and 1, making it suitable for binary classification tasks. The sigmoid function is defined as:

$$\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]$$

---

### Logistic Regression: Training

#### Iteration for Convergence

Training the logistic regression model involves iterating to minimize the cost function. The iterative process converges to the minimum cost, refining the model's parameters.

---

### Logistic Regression: Testing

#### Validation Dataset and Accuracy Calculation

Testing the model involves using a validation dataset to assess its performance. The accuracy is calculated as the ratio of correctly predicted instances to the total number of instances:

$$\[ \text{Accuracy} = \frac{\sum_{i} (\text{predict}(i) == y(i)_{va})}{m} \]$$

Here, $\(m\)$ represents the total number of instances.

---

### Cost Function

#### Equation and Intuition

The cost function measures the disparity between predicted and actual values. The logistic regression cost function is defined as:

 $$ \[ J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))] \]$$
Where:
- $\(m\)$ is the number of instances.
- $\(y^{(i)}\)$ is the actual label.
- $\(h_\theta(x^{(i)})\)$ is the predicted output.

Minimizing the cost function optimizes the parameters (\(\theta\)) for accurate predictions.

---

## Week 2 Sentiment Analysis with Naive Bayes 

Learn the theory behind Bayes' rule for conditional probabilities, then apply it toward building a Naive Bayes tweet classifier of your own!

**Learning Objectives**

    Error analysis
    Naive Bayes inference
    Log likelihood
    Laplacian smoothing
    conditional probabilities
    Bayes rule
    Sentiment analysis
    Vocabulary creation
    Supervised learning

### Probability and Bayes' Rule

#### Probability of an Event

The probability of an event, denoted as \( P(A) \), represents the likelihood of occurrence of that event. It ranges from 0 (impossible event) to 1 (certain event).

#### Conditional Probabilities

Conditional probabilities \( P(B|A) \) represent the probability of event B occurring given that event A has occurred. It is defined as the ratio of the probability of the intersection of events A and B to the probability of event A.

#### Bayes' Rule

Bayes' Rule relates conditional probabilities:

\[ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} \]

In the context of Twitter analysis, for a positive tweet:
\[ P(\text{positive|happy'}) = \frac{P(\text{positive} \cap \text{happy})}{P(\text{happy'})} \]

\[ P(\text{happy'|positive}) = \frac{P(\text{happy'} \cap \text{positive})}{P(\text{positive})} \]

\[ P(\text{positive|"happy"}) = P(\text{positive | happy}) \cdot \frac{P(\text{Positive})}{P(\text{happy})} \]


#### Naive Bayes Inference for Binary Classification

For binary classification, Naive Bayes inference conditions are expressed as:

\[ P(\text{pos})/P(\text{neg}) \cdot \prod_{i=1}^{m} P(w_i | \text{pos})/P(w_i|\text{neg}) \]

#### Table of Probabilities

Probabilities are often organized in a table, providing conditional probabilities for each word given the sentiment class.

---

### Laplacian Smoothing

Laplacian Smoothing is employed to avoid zero probabilities, especially when a word has not appeared in a particular class. The formula for smoothed probability is:

\[ P(w_i|\text{class}) = \frac{\text{freq}(w_i, \text{class}) + 1}{\text{N}_\text{class} + \text{V}_\text{class}} \]

Where:
- \(\text{N}_\text{class}\) is the frequency of all words in the class.
- \(\text{V}_\text{class}\) is the number of unique words in the class.

---

### Log Likelihood

#### Ratio of Probabilities

The ratio of probabilities is expressed as:

\[ \text{ratio}(w_i) = \frac{P(w_i | \text{pos})}{P(w_i|\text{neg})} \]

- Neutral words have a ratio of 1.
- Positive words have a ratio approaching positive infinity.
- Negative words have a ratio approaching 0.

#### Naive Bayes Inference and Log Likelihood

The Naive Bayes Inference is facilitated by comparing the product of prior ratio and probabilities. Log likelihood is used to mitigate the risk of underflow by transforming the product into a sum:

\[ \text{Log Likelihood} = \log\left(\frac{P(\text{pos})}{P(\text{neg})} \cdot \prod_{i=1}^{m} \frac{P(w_i | \text{pos})}{P(w_i|\text{neg})}\right) \]

This is computed as the sum of the log prior and log likelihood.

---

These concepts form the basis of Bayesian probability, Naive Bayes classification for sentiment analysis, Laplacian Smoothing, and Log Likelihood. Understanding these principles is essential for implementing effective sentiment analysis models.


### Probability and Bayes' Rule

#### Probability of an Event

The probability of an event \( P(A) \) is the likelihood of its occurrence and ranges between 0 (impossible) and 1 (certain).

#### Conditional Probabilities

Conditional probabilities \( P(B|A) \) signify the probability of event B happening given that event A has occurred.

#### Probability of B Given A

\[ P(B|A) = \frac{\text{Probability of } A \cap B}{\text{Probability of } A} \]

#### Bayes' Rule

For Twitter analysis of a happy, positive tweet:

\[ P(\text{positive|happy'}) = \frac{P(\text{positive} \cap \text{happy})}{P(\text{happy'})} \]

\[ P(\text{happy'|positive}) = \frac{P('happy' \cap \text{Positive})}{P(\text{Positive})} \]

\[ P(\text{positive|"happy"}) = P(\text{positive | happy}) \cdot \frac{P(\text{Positive})}{P(\text{happy})} \]

---

### Naive Bayes for Sentiment Analysis

#### Naive Bayes Inference for Binary Classification

The Naive Bayes inference rule for binary classification:

\[ \text{Naive Bayes Inference} = \prod_{i=1}^{m} \frac{P(w_i | \text{pos})}{P(w_i|\text{neg})} \]

#### Table of Probabilities

Probabilities for each word are organized in a table, helping calculate the Naive Bayes inference for sentiment analysis.

---

### Laplacian Smoothing

Laplacian Smoothing is a technique to prevent zero probabilities, especially for words with zero conditional probability.

\[ P(w_i|\text{class}) = \frac{\text{freq}(w_i, \text{class}) + 1}{\text{N}_\text{class} + \text{V}_\text{class}} \]

Where:
- \(\text{N}_\text{class}\) is the frequency of all words in the class.
- \(\text{V}_\text{class}\) is the number of unique words in the class.

---

### Log Likelihood

#### Ratio of Probabilities

\[ \text{ratio}(w_i) = \frac{P(w_i | \text{pos})}{P(w_i|\text{neg})} \]

- Neutral words have a ratio of 1.
- Positive words have a ratio approaching positive infinity.
- Negative words have a ratio approaching 0.

#### Naive Bayes Inference and Log Likelihood

Naive Bayes Inference is determined by comparing the product of prior ratio and probabilities:

\[ \text{Naive Bayes Inference} = \frac{P(\text{pos})}{P(\text{neg})} \cdot \prod_{i=1}^{m} \frac{P(w_i | \text{pos})}{P(w_i|\text{neg})} \]

However, using logs prevents numerical underflow:

\[ \text{Log Likelihood} = \log\left(\frac{P(\text{pos})}{P(\text{neg})} \cdot \prod_{i=1}^{m} \frac{P(w_i | \text{pos})}{P(w_i|\text{neg})}\right) \]

This is computed as the sum of the log prior and log likelihood.

---

### Calculate the Lambdas

\[ \Lambda(w) = \log\left(\frac{P(w|\text{pos})}{P(w|\text{neg})}\right) \]

Log likelihood is calculated using the formula:

\[ \text{Log Likelihood} = \log\left(\frac{P(w|\text{pos})}{P(w|\text{neg})}\right) \]

These concepts provide the foundation for applying Bayes' Rule, Naive Bayes inference, Laplacian Smoothing, and Log Likelihood in sentiment analysis. Understanding these principles is crucial for developing effective models.



## Week 3 

### Training Naive Bayes

#### Step 0: Corpus Collection and Annotation

- Collect a corpus and annotate it, dividing it into positive and negative groups.

#### Step 1: Preprocessing

- Convert all text to lowercase.
- Remove punctuation, URLs, and names.
- Eliminate stop words.
- Apply stemming.
- Tokenize the text.

#### Step 2: Word Count

- Count the frequency of each word in the corpus.

#### Step 3: Conditional Probability Estimation

- Calculate \(P(w|c)\) using Laplacian Smoothing: \(\frac{\text{freq}(w, \text{class}) + 1}{\text{N}_\text{class} + \text{V}_\text{class}}\).
  - \(\text{N}_\text{class}\): Frequency of all words in the class.
  - \(\text{V}_\text{class}\): Number of unique words in the class.

#### Step 4: Lambda Calculation

- Compute \(\Lambda(w) = \log\left(\frac{P(w|\text{pos})}{P(w|\text{neg})}\right)\).

#### Step 5: Log Prior Estimation

- Estimate the log prior: \(\text{log\_prior} = \log\left(\frac{D_\text{pos}}{D_\text{neg}}\right)\), where \(D_\text{pos}\) is the number of positive tweets, and \(D_\text{neg}\) is the number of negative tweets. If the dataset is balanced, \(\text{log\_prior} = 0\).

---

### Testing Naive Bayes

- Utilize a validation dataset to assess the performance of the trained Naive Bayes model.

---

### Applications of Naive Bayes

- **Author Identification:** Identifying the authorship of texts.
- **Spam Filtering:** Filtering out spam emails.
- **Information Retrieval:** Retrieving relevant information from a dataset.
- **Word Disambiguation:** Resolving ambiguity in the meaning of words.

---

### Naive Bayes Assumptions

1. **Independence of Words:** Assumes that words in a sentence are independent of each other.
2. **Dependence on Relative Frequencies:** Relies on the relative frequencies of words in the corpus.

---

### Error Analysis

- **Impact of Punctuation and Stop Words:** The removal of punctuation or stop words can have varying impacts.
- **Word Order:** The order of words, especially negations, can significantly affect results.
- **Adversarial Attacks:** Sarcasm, irony, and euphemisms may be challenging for Naive Bayes.

These errors can lead to misclassifications in examples or tweets, emphasizing the need for careful consideration during model development.

--- 


## Week 3  Vector Space Models

Vector space models capture semantic meaning and relationships between words. You'll learn how to create word vectors that capture dependencies between words, then visualize their relationships in two dimensions using PCA.

**Learning Objectives**

    Covariance matrices
    Dimensionality reduction
    Principal component analysis
    Cosine similarity
    Euclidean distance
    Co-occurrence matrices
    Vector representations
    Vector space models
    

#### Why Vector Space Models?

- **Purpose:** Determine the similarity between sentences.
  - e.g., "Where are you heading?" vs. "Where are you from?" vs. "What is your age?" vs. "How old are you?"
  - Helps capture semantic similarity even with different word arrangements.

#### Applications of Vector Space Models

- **Capture Dependencies:**
  - e.g., "You eat cereal from a bowl." (cereal--bowl)
- **Information Extraction:**
  - Extract information about who, what, how, etc.
- **Used in:**
  - Machine translation.
  - Chatbots.

#### Fundamental Concept

- "You shall know a word by the company it keeps." - Firth (1957)

---

### Word by Word and Word by Doc

#### Co-occurrence Word by Word Design

- **Co-occurrence:**
  - Count the number of times words occur together within a certain distance \(k\).
  
- **Word by Word Design:**
  - **Explanation:** Measures how often words co-occur.
  - **Building Co-occurrence Matrix:**
    - Count occurrences of each word within \(k\) distance.
  
#### Co-occurrence Matrix for Word by Document Design

- Count occurrences of words within specific categories (e.g., entertainment, economy, machine learning).
- Provides a vector space representation.
- Use/plot vectors to find similarity or clusters.
- Numeric similarity using angle distance or cosine similarity.

---

### Euclidean Distance

- **Explanation:**
  - Measures the straight-line distance between two points in space.
- **Examples:**
  - Two vectors: \([2, 3]\) and \([5, 7]\).
  - n-vector example: \([a_1, a_2, ..., a_n]\).

---

### Cosine Similarity

#### Summary of Euclidean Distance Issue:

- Sensitive to the length of vectors.
- Ignores the direction of vectors.

#### How Cosine Similarity Helps:

- Considers the cosine of the angle between vectors.
- Robust to vector length variations.
- Reflects the similarity in direction.

#### Cosine Similarity Formula:


$\[ \text{Cosine Similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} \]$$

- \(\mathbf{A} \cdot \mathbf{B}\): Dot product of vectors \(\mathbf{A}\) and \(\mathbf{B}\).
- \(\|\mathbf{A}\|\) and \(\|\mathbf{B}\|\): Euclidean norms of vectors \(\mathbf{A}\) and \(\mathbf{B}\).

This formula quantifies the cosine of the angle between vectors, providing a normalized measure of similarity. It ranges from -1 (completely dissimilar) to 1 (identical), with 0 indicating orthogonality.



### Manipulating Words in Vector Spaces

#### Using Simple Vector Operations

- **Purpose:**
  - Find relationships between words using vector addition and subtraction.
  - e.g., Determine the capital city of a country.

- **Example:**
  - If \( \text{USA-Washington} \) is a known vector, then \( \text{Russia} + \text{City} \) would yield a similar vector.

- **Key Points:**
  - Leverage known relationships to make predictions.
  - Identify patterns in text using vector manipulations.

---

### Visualization and PCA

#### Dealing with High-Dimensional Vectors

- **Challenge:**
  - Word or sentence vectors can be high-dimensional.

- **PCA (Principal Component Analysis):**
  - **Purpose:**
    - Reduce high-dimensional vectors for visualization.
  - **Process:**
    - Identify principal components (dimensions) that capture the most variance.
    - Retain essential information while reducing dimensionality.

- **Significance:**
  - Allows effective visualization of word embeddings or document vectors.

These concepts in manipulating vectors and using PCA contribute to understanding relationships between words and sentences in vector spaces, facilitating meaningful predictions and visualization.


### PCA Algorithms

#### Eigenvectors

**Definition:**
Eigenvectors represent uncorrelated features for your data. In the context of PCA, they become crucial in understanding the inherent structure of high-dimensional datasets.

**Significance:**
Each eigenvector corresponds to a principal component, capturing distinct information in the data. They are essential for dimensionality reduction and feature transformation.

#### Step 1: Getting Uncorrelated Features

**Mean Normalization:**
Mean normalization ensures that the data is centered around zero, removing any bias. It involves subtracting the mean (\(\mu_{x_i}\)) and dividing by the standard deviation (\(\sigma_{x_i}\)) for each feature (\(x_i\)).

**Co-Variance Matrix:**
The co-variance matrix provides insights into how features vary together. It's calculated by summing the outer products of the mean-normalized data. This step is crucial for understanding the relationships between different features.

**Singular Value Decomposition (SVD):**
SVD decomposes the co-variance matrix into three matrices: \(U\), \(S\), and \(V^T\). \(U\) contains the eigenvectors, and \(S\) has the corresponding singular values. This decomposition facilitates the transformation of the data into a new basis defined by the eigenvectors.

#### Step 2: Projecting Data to New Features

**Utilize Eigenvectors and Eigenvalues (\(U S\)):**
The eigenvectors and eigenvalues from the SVD are used to project the data onto a new set of features. This step retains the most significant information while reducing the dimensionality of the dataset.

---

### The Rotation Matrix

#### Counterclockwise Rotation

**Process:**
1. **Translate Data to the Origin:**
   - Shifting the data to the origin simplifies the rotation process.

2. **Perform Rotation:**
   - Apply the rotation matrix to the translated data. The rotation matrix is constructed from the angle of rotation.

3. **Translate Data Back:**
   - Return the data to its original position. This step completes the counterclockwise rotation.

#### Clockwise Rotation

**Process:**
1. **Translate Data to the Origin:**
   - Similar to counterclockwise rotation, shift the data to the origin.

2. **Perform Clockwise Rotation:**
   - For a clockwise rotation, the negative angle rotation matrix is applied to the translated data.

3. **Translate Data Back:**
   - Finally, translate the data back to its original position, completing the clockwise rotation.

---

Understanding these PCA algorithms is fundamental for data scientists and analysts to effectively analyze and visualize high-dimensional datasets. Eigenvectors provide insights into feature relationships, and the rotation matrix operations enable manipulation of data orientation in vector spaces.


## Week 4: Machine Traanslation and Document Search

Learn to transform word vectors and assign them to subsets using locality sensitive hashing, in order to perform machine translation and document search.

**Learning Objectives**

    Gradient descent
    Approximate nearest neighbors
    Locality sensitive hashing
    Hash functions
    Hash tables
    K nearest neighbors
    Document search
    Machine translation
    Frobenius norm

###  Nearest Neighbor Search and Locality Sensitive Hashing

#### Introduction
- Focus: Quick nearest neighbor search using locality-sensitive hashing.
- Application: Translate words between languages by manipulating word vectors.
- Goal: Learn to implement machine translation and document search efficiently.

#### Overview
- **Skills to Practice:**
  1. Transforming word vectors.
  2. Implementing k-nearest neighbors.
  3. Understanding hash tables for word vector assignment.
  4. Dividing vector space into regions.
  5. Implementing locality-sensitive hashing for approximated k-nearest neighbors.

- **Tasks:**
  - Machine translation: Translate English "hello" to French "bonjour."
  - Document search: Find similar documents based on a given sentence.
  
#### Transforming Word Vectors
- **Machine Translation Process:**
  1. Generate word embeddings for English and French.
  2. Learn a transformation matrix (R) to align English vectors with French vectors.
  3. Search for similar word vectors in the French vector space.

- **Matrix Transformation (R):**
  - Define R randomly.
  - Compare X times R with actual French word embeddings (Y).
  - Iteratively improve R using gradient descent.

- **Training Subset:**
  - Collect a subset of English-French word pairs for training.
  - Align word vectors by stacking them in matrices X and Y.

- **Objective Function:**
  - Measure distance between attempted translation (X times R) and actual French vectors (Y).
  - Optimize R by minimizing the square of the Frobenius norm.

#### Frobenius Norm and Gradient Calculation
- **Frobenius Norm:**
  - Measure of the magnitude or norm of a matrix.
  - Easier to work with the square of the Frobenius norm.
  - Formula: Square each element, sum them, and take the square root.

- **Gradient Descent:**
  - Derive the loss function with respect to the matrix R.
  - Use the square of the Frobenius norm for easier calculations.
  - Iterate to update R until the loss falls below a threshold.

#### Conclusion
- **Outcome:**
  - Ability to align word vectors for machine translation.
  - Understanding of efficient nearest neighbor search using locality-sensitive hashing.

#### Key Takeaways
- **Core Concept:**
  - Transforming word vectors using matrices for cross-language alignment.

- **Practical Application:**
  - Efficiently find nearest neighbors using locality-sensitive hashing.

- **Optimization Technique:**
  - Gradient descent for iteratively improving transformation matrices.

- **Flexibility:**
  - Train on a subset, apply to a broader vocabulary for scalability.

- **Continuous Learning:**
  - Emphasis on practice and understanding through assignments.

#### Expert Insights
- **Matrix Transformation Impact:**
  - Discuss the trade-offs and nuances in selecting and refining the transformation matrix.

- **Real-world Challenges:**
  - Explore complexities in scaling to larger datasets and diverse language pairs.

- **Further Reading:**
  - Suggest additional resources and research papers for in-depth exploration.

#### Advanced Thoughts
- **Beyond Nearest Neighbors:**
  - Discuss potential advancements or alternative approaches in NLP tasks beyond nearest neighbor search.

- **Ethical Considerations:**
  - Explore ethical implications of efficient document search and translation.

- **Emerging Trends:**
  - Discuss recent developments in NLP and their relevance to the covered topics.


### K-Nearest Neighbors and Hashing

#### K-Nearest Neighbors Operation
- **Key Operation:**
  - Finding the k nearest neighbors of a vector.
  - Essential for various NLP techniques.

- **Transformation Output:**
  - Transformed vector (after applying R matrix) resides in the French word vector space.
  - Not necessarily identical to existing French word vectors.

- **Similar Word Search:**
  - Search through actual French word vectors to find a similar word.
  - Example: Translating "hello" may yield "salut" or "bonjour."

#### Finding Neighbors Efficiently
- **Analogy: Finding Nearby Friends:**
  - Similar to finding friends nearby in a geographical space.
  - Linear search through all friends is time-intensive.

- **Optimization Idea:**
  - Filter friends by region (e.g., North America) to narrow down the search.
  - Efficiently organize subsets of data using hash tables.

- **Hash Tables:**
  - Efficient data structure for organizing and searching data.
  - Essential for various data-related tasks.

- **Introduction to Hash Tables:**
  - Address the challenge of finding similar word vectors efficiently.
  - Provides a foundation for the upcoming exploration of locality-sensitive hashing.

#### Hash Tables and Hash Functions
- **Concept of Hash Tables:**
  - Organizing data into buckets based on similarity.
  - Each item consistently assigned to the same bucket.

- **Hash Function:**
  - Assigns a hash value (key) to each item, determining its bucket.
  - Example: Modulo operator to assign hash values.

- **Basic Hash Table Implementation:**
  - Create a hash function.
  - Assign items to buckets based on hash values.
  - Efficient organization of data subsets for faster search.

#### Locality-Sensitive Hashing (LSH)
- **Introduction to LSH:**
  - Objective: Place similar word vectors into the same bucket.
  - Utilizes planes to divide vector space into regions.

- **Defining Planes:**
  - Planes used to slice vector space based on the location of vectors.
  - Each plane defines a region where vectors are grouped together.

- **Normal Vectors and Dot Product:**
  - Normal vectors perpendicular to planes.
  - Dot product determines the position of vectors relative to the plane.

- **Understanding Dot Product Sign:**
  - Positive dot product: Vector on one side of the plane.
  - Negative dot product: Vector on the opposite side of the plane.
  - Zero dot product: Vector on the plane.

- **Code Implementation:**
  - Function `side_of_plane` determines which side of the plane a vector lies.
  - Utilizes numpy's `np.dot` and `np.sign` functions.

#### Conclusion
- **Key Concepts Covered:**
  - K-nearest neighbors operation.
  - Efficient search using hash tables.
  - Introduction to locality-sensitive hashing (LSH).

- **Upcoming Topic:**
  - Further exploration of LSH and its application in reducing computational costs.

#### Advanced Thoughts
- **Optimization Challenges:**
  - Discuss challenges in optimizing hash functions for NLP tasks.
  - Considerations for handling high-dimensional word vectors.

- **Comparison with Traditional Methods:**
  - Compare efficiency gains with LSH against traditional search methods.
  - Evaluate trade-offs and practical considerations.

- **Research Directions:**
  - Explore recent research on enhancing nearest neighbor search in high-dimensional spaces.
  - Consider implications for the field of natural language processing.

These notes provide an in-depth understanding of the concepts introduced in Week 4, combining theoretical knowledge with practical insights into the implementation of hash tables and locality-sensitive hashing.


### Multiple Planes and Approximate Nearest Neighbors

#### Combining Multiple Planes
- **Objective:**
  - Combine signals from multiple planes to obtain a single hash value.
  - Each plane provides information about a vector's position.

- **Example:**
  - For each plane, determine if the dot product is positive or negative.
  - Combine intermediate hash values using a specific formula.

- **Combining Signals Formula:**
  - For planes 1, 2, ..., n: `2^0 * h_1 + 2^1 * h_2 + ... + 2^(n-1) * h_n`

- **Rules for Intermediate Hash Values:**
  - If dot product sign >= 0, assign intermediate hash value of 1.
  - If dot product sign < 0, assign intermediate hash value of 0.

- **Implementation in Code:**
  - Calculate dot products for each plane.
  - Determine intermediate hash values based on dot product signs.
  - Combine intermediate hash values using the specified formula.

#### Locality-Sensitive Hashing (LSH) Implementation
- **Purpose of LSH:**
  - Divide vector space into manageable regions using multiple planes.
  - Achieve a single hash value for efficient bucket assignment.

- **Code Implementation Steps:**
  1. Initialize hash value to zero.
  2. For each plane:
     - Calculate dot product and determine sign.
     - Assign intermediate hash value based on sign.
     - Update hash value using the combining formula.
  3. Return the final hash value.

- **Visualization:**
  - Imagine multiple sets of random planes creating hash pockets.
  - Each set contributes to a different way of dividing the vector space.

- **Approximate Nearest Ne...


### Searching Documents Using LSH

#### Document Representation as Vectors
- **Objective:**
  - Represent documents as vectors for efficient search using k-nearest neighbors.

- **Representation Method:**
  - Obtain word vectors for each word in a document.
  - Sum the word vectors to create a document vector.

- **Example:**
  - Document: "I love learning."
  - Words: "I," "love," "learning."
  - Document vector: Word vectors' sum in the same dimension.

- **Mini Dictionary for Word Embeddings:**
  - Create a dictionary for word embeddings.

- **Initialization:**
  - Initialize the document embedding as an array of zeros.

- **Implementation Steps:**
  1. For each word in the document:
     - Get the word vector if the word exists in the dictionary.
     - Add the vectors.
  2. Return the document embedding.

#### Document Search Using K-Nearest Neighbors
- **Objective:**
  - Apply k-nearest neighbors to perform document search.

- **Implementation Steps:**
  1. Create vectors for both the query and the documents.
  2. Use k-nearest neighbors to find the nearest neighbors.

- **General Method for Text Embedding:**
  - Sum word vectors to create document vectors.
  - This method is foundational and widely used in modern NLP.

- **Conclusion:**
  - Basic structure of text embedding reappears in various NLP applications.

###ss# Key Takeaways and Summary

- **Week's Objectives:**
  - Understand and implement Locality-Sensitive Hashing (LSH).
  - Combine signals from multiple planes to achieve approximate nearest neighbors.
  - Apply LSH for efficient k-nearest neighbor search.
  - Represent documents as vectors and perform document search.

- **Locality-Sensitive Hashing:**
  - Division of vector space using multiple planes.
  - Combine signals from planes to obtain a single hash value.
  - Efficiently search for approximate nearest neighbors.

- **Document Search:**
  - Represent documents as vectors by summing word vectors.
  - Apply k-nearest neighbors for document search.
  - Use basic text embedding structure in NLP applications.

- **Implementation Practice:**
  - Code examples provided for combining signals, LSH, and document search.
  - Practical applications of the learned concepts.

- **Future Applications:**
  - LSH and k-nearest neighbors used in various NLP tasks.
  - Document search as a common application of text embedding.
