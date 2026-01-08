### The Bag-of-Words (BoW) model

The BoW model is a fundamental technique in Natural Language Processing (NLP) used to turn text into numbers that a computer can understand. TO explain this, let's use a new set of data involving fruit reviews.

1. **The "Raw" Dataset**

Computers cannot do math on the words "apple" or "tasty", so we start with our raw sentences:
- sentence 1 (x^1): Red apple;
- sentence 2 (x^2): Green apple;
- sentence 3 (x^3): Red apple and green apple.

2. **The vocabulary (dictionary)**

We look at all the sentences and make a list of every unique word. We then assign each word a specific index (a position).

![rnn](https://github.com/user-attachments/assets/a75684ca-869a-47ed-b107-c0fdc984c215)

3. **The Design Matrix (X)**

This is where the bag come in. We ignore the order of the words and simply count how many times each word from our vocabulary appears in each sentence. Each row below represents one of our sentences, and each column represents a word from our vocabulary (indices 0, 1, 2, 3):

```python
X = [ [0, 1, 0, 1],
      [0, 1, 1, 0],
      [1, 2, 1, 1] ]
```
![rnn2](https://github.com/user-attachments/assets/e73741d2-b31e-4500-9198-7394595dde6d)

4. **Class Labels (y)**

In text classification, we usually have a goal, like deciding if a review is positive (1) or negative (0).
- y = [1, 1, 1]. In this case, maybe they are all "Fresh" fruit reviews.

![rnn3](https://github.com/user-attachments/assets/d4d78488-45be-4290-aee0-7855a13cb71e)

5. **The Classifier**

Finally, the matrix X and the labels y are fed into machine learning algorithm (like Logistic Regression). The computer looks at the counts - for example, it might learn that whenever the word apple (index 1) appears with a high count, the label is likely fresh.

f(X) -> y, where
- f is the classifier function that maps the word count vectors to class labels.

**Summary of the concept**
- the model does not care if you say "apple is red" or "red is apple". It only cares about the count; (lose the grammer)
- we are turning a string of text into a vector of numbers; (feature extraction)
- (sparsity) in real life, a vocabulary might have 10 000 words. If a sentence only has 5 words, the row will be mostly zeros.

### How 1D Convolutional Neural Network (1D CNNs) process text or sequence data

Unlike the Bag-of_Words model which treats words as isolated counts, a 1D CNN slides over the text to capture local patterns and context.

1. The input sequence
   - our input is "The coffee is hot.";
   - each word (or character) is converted into numerical vector (called an embedding).
  
![cnn1](https://github.com/user-attachments/assets/05286a72-4d94-4acb-909e-b7f9b0fd21b9)

Embeddings position semantically similar words closer in vector space.
  
2. The convolution operation

1D CNNs apply filters (kernels) that slide across the sequence of word embeddings.

![cnn2](https://github.com/user-attachments/assets/ea2fd788-5f5b-445d-80cc-1104c0b61bec)

**How it works**: the filter (size 2) slides across pairs of word embeddings, computing dot products to detect local patterns (like "coffee is", "is how").

the formula: Feature = σ(∑(embeddingᵢ × filterᵢ) + bias)

3. Feature Maps and Pooling

Each filter produces a feature map capturing specific patterns at different positions.

![cnn3](https://github.com/user-attachments/assets/e749db48-8c32-4d30-8d53-e09ded254a84)

Max Pooling retains the strongest activation for each filter, making the representation position-invariant and reducing dimensionality. 

Pooled Features = [0.42, 0.67, 0.12, ...]

4. Classification

The pooled features are fed into a fully connected layer for classification.

the formula: z = W·pooled_features + b

The network learns weights that combine evidence from different filters to make predictions:
- high activation from intensity filter + sentiment filter → positive;
- High activation from "negation" filter → Could invert sentiment;
- Multiple layers can learn hierarchical patterns.

### Difference between standard machine learning and sequence-based models

1. The Independent and Identically Distributed (IID) Assumption

In most basic machine learning (like classifying images of cats vs. dogs), we assume data is IID.
- knowing one data point tells us nothing about the next. For example, seeing a picture of a cat does not make the next picture more or less likely to be a dog (independent);
- all data points are drawn from the same source or probability distribution (identically distributed);
- the image shows individual points (x^{(1)}, x^{(2)}, x^{(3)}) floating separately because they do not influence each other.

2. Sequential Data is not IID

Sequential data - like text, speech, or stock prices - breaks the independence rule.
- in a sequence, the past predicts the future (strong correlation). For example, if it is 30°C right now (x^{(t-1)}), it is highly likely to be around 30°C in one minute (x^{(t)}). The data points are linked;
- the image shows the points connected by arrows (x^{(1)} → x^{(2)} → x^{(3)}). This represents temporal dependency, where each value depends on what came before it.

**Why this matters**
- **standard models**: algorithms like linear regression or standard neural networks often struggle with sequence because they treat every data point as a fresh start.
- **sequence models**: models like RNNs or LSTMs are specifically designed for the bottom half of the image. They have a memory to track those connections over time.

### The applications of sequential data models

They are designed to handle information where the order and history of data points are critical for prediction. While a standard model might look at a single data point in isolation, these models look at a history of points to understand the present. To explain this, let's look at three new real-world examples:

1. Stock morket forecasting

Financial data is a classic time series where today's price is heavily dependent on yesterday's.
- the sequence (x): a sequence of closing prices for the last 30 days (e.g., $150, $152, $149 ...);
- the target (y) is the predicted price for tomorrow;
- why it works, because the model identifies patterns like a sudden drop, which a single price point wouldn't reveal.

2. Natural language processing

In text, a word's meaning changes based on the words before it.
- the sequence (x): a customer review: "The battery life is really ...";
- the target is predicting the next word (e.g., short or long) or the overall sentiment (positive vs. negative);
- why it works, because the model uses its hidden state (memory) to remember that the subject is "battery life" when it reaches the end of the sentence.

3. Genomics / DNA Sequencing

DNA is a long sequence of four bases (A, T, C, G).
- the sequence (x): A strand like "A-C-G-T-T-G...";
- the target is identifying if this sequence belongs to a specific gen or indicates a potential health risk;
- why it works, because biological functions are determined by the specific order of these bases, similar to how the order of letters creates  words.

**summary**

![sequentialmodel](https://github.com/user-attachments/assets/9d0f1d86-6d59-42e4-b71c-fe9c7037458f)

### How "memory" is physically built into the architecture

1. Feedforward Neural Network

On the left, we see a traditional network where data flows in one direction: from input (x) to output (y).


2. Recurrent Neural Network (the 'memory' model)

On the right, a recurrent edge (the loop) is added to the hidden layer (h).

![difference](https://github.com/user-attachments/assets/da52bd71-ed7b-4db1-8ab0-3a58cb204677)
