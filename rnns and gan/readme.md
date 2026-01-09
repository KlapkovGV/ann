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
- how it work: imagine we feed the network the word "stir". It might guess the next word is "it";
- the limitation: if we then feed it the word "the", it has completely forgotten that we just said stir;
- the math: each input is treated as independent. It has no internal state to store what happened a second ago.

Best for photos and static data.

2. Recurrent Neural Network (the 'memory' model)

On the right, a recurrent edge (the loop) is added to the hidden layer (h).
- the loop: this represent a feedback connection where the output of the hidden layer at one moment is fed back into itself for the next moment;
- hidden state (h): this acts as the network's memory. It stores a summary of everything it has seen in the sequence so far.

**How it works**
1. We put input "stir". The hidden states remembers "We are currently stirring";
2. We put "the". The network combines this new input with its memory of "stiring";
3. As a result, it can now accurately predict "soup" instead of guessing a random word.

Best for video, voice, and text.

![rnn8](https://github.com/user-attachments/assets/e3a33041-764a-4cf2-8e1c-f160cc246390)

Sebastian Raschka, STAT 453: Intro to Deep Learning and Generative Models, SS 2020

### Unrolling a Recurrent Neural Network

An unrolled RNN is a way to visualize and implement a reccurent neural network by expanding it over discrete time steps, turning the cyclic network into a feed-forward chain.

![unrolling](https://github.com/user-attachments/assets/3a62f64f-6535-497a-a7c6-3a8ede91b0f1)

1. The compressed view (left)

It shows the RNN as single block with loop.
- the input is a single snapshot of sensor data;
- the hidden layer is the "brain" of the network that holds its current memory;
- the loop represents the recurrence. It feeds its own memory back into itself for the next moment.

2. The unrolled view (rigth)

To understand how the math works, we unroll the loop into a chain. Instead of one looping layer, we visualize it as a series of identical layers - one for each Time Step (t).
- time step t-1 represents past. The sensor detects "door opened". This information is stored in the hidden state h^(t-1);
- time step t represent present. The sensor now detects "footsteps". The network combines this new input x^t with its memory of the door opening h^(t-1) to update its current state h^t;
- time step t+1 represent future. The next input arrives. The chain continues, allowing the network to see the story of the data rether than just isolated moments.

**Paramenter sharing**

A critical detail in this image is that weights (the knoledge) inside the block do not change as it unrolls.
- the same mathematical fuction f is used at every step;
- because the weights are shared, the network can process a sequence of any length wheter it is a 3-second clip or a 30-second clip using the same amount of memory.

### RNN Architecture Types

In these diagrams:
- green circles are inputs (x);
- blue circles are hidden states (the network's memory);
- orange circles are outputs (y).

**1. Many-to-One**

The model takes a whole sequence of data and gives us a single answer at the end.

Example: Driver distraction detection
- the flow: the model looks at 30 seconds of steering wheel movements and eye-tracking data and finally outputs a single yes or no for whether the driver is falling asleep.

![rnn4](https://github.com/user-attachments/assets/a8064edc-5dfa-44f6-b562-7701a3a08a32)

**2. One-to-Many**

The model takes a single piece of information and generates an entire sequence.

Example: Generating a trip summary
- the flow: we give the vodel a single image of destination, and it generates a text description like "A sunny beach with palm trees and white sand" (the many outputs, word by word).

![rnn5](https://github.com/user-attachments/assets/3f7050c0-4ef5-4018-9458-4543a15e2d34)

**3. Many-to-Many (Synced)**

For every input that goes in, an output comes out immediatly. The input and output sequences are the same length.

Example: Frame-by-Frame Video Labeling
- the flow: in a dashcam video, for every single frame (x) the camera captures, the model must immediatly output a label (y) like traffic light, or car.

![rnn6](https://github.com/user-attachments/assets/7c0e726f-e5a8-4f26-974c-5bddf7af2e82)

**4. Encoder-Decoder**

The model reads the entire input sequence first, thinks about it, and then starts producing an output sequence that might be a different length.

Example: Language translation (GPS voice)
- the flow: the system hears the English phrase "turn left at the next light". It processes that whole sequence and then generates the Russian translation "Поверните налево на следующем светофоре".

![rnn7](https://github.com/user-attachments/assets/1a82820e-fb5f-4777-b86d-a50ab8c33ef2)


### The mathematical detail for how RNN maintains its "knowledge" across a sequence 

![rnn9](https://github.com/user-attachments/assets/6e665318-ae74-4fd8-961a-8d9d5366c9a2)

*Sebastian Raschka, STAT 453: Intro to Deep Learning and Generative Models, SS 2020*

To explain this, let's use the example of an auto-complete on a smartphone.

**1. The three key weight matrices in texting**

When we type a message, our phone's "brain" (the RNN) uses these specific sets of rules to guess our next word:
- W_hx (input-to-hidden) handles the current word. If we typed "happy", this matrix processes those specific letters so the brain understands the immediate input;
- W_hh (hidden-to-hidden) is the conversation memory. If the word before "happy" was "birthday", this matrix carries that birthday context forward. It tells the brain: "do not forget, we are talking about an anniversary";
- W_yh (hidden-to-output) is the final guess. It takes the combined content (birthday + happy) and translates it into the predicted word we see on the screen: "birthday".

**2. How the sentence works**

The unrolled view shows how these matrices work together step-by-step:
- at t-1 we type "have". The brain (h^(t-1)) stores the idea that a sentence is starting;
- at t we type "a". The network uses W_hh to remember "have" and W_hx to precess "a";

Result: because of W_yh, the phone suggests "great" or "nice". It knows "a" usually follows "have" in this context.
