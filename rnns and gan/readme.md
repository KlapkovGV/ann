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

