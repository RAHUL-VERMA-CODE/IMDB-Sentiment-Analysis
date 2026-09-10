# IMDB Sentiment Analysis using Simple RNN

A Natural Language Processing (NLP) application that uses a **Simple Recurrent Neural Network (RNN)** to classify IMDB movie reviews as **Positive** or **Negative**.

The project includes model training, text preprocessing, evaluation, and an interactive **Streamlit web application** for real-time sentiment prediction.

---

## Overview

Sentiment analysis is an NLP task used to determine the emotional polarity of text.

This project uses the **IMDB Movie Review Dataset** and a neural network architecture consisting of an Embedding layer, Simple RNN, and Sigmoid output layer.

### Workflow

```text
IMDB Dataset
     │
     ▼
Integer Encoding
     │
     ▼
Sequence Padding
     │
     ▼
Embedding Layer
     │
     ▼
Simple RNN
     │
     ▼
Sigmoid Output
     │
     ▼
Positive / Negative
```

---

## Model Architecture

```text
Input Sequence (500 tokens)
          │
          ▼
Embedding
10,000 vocabulary × 128 dimensions
          │
          ▼
SimpleRNN
128 units
Activation: tanh
          │
          ▼
Dense
1 unit
Activation: sigmoid
          │
          ▼
Sentiment Prediction
```

### Architecture Details

| Layer     | Configuration                       |
| --------- | ----------------------------------- |
| Input     | Sequence length: 500                |
| Embedding | Vocabulary: 10,000, Dimensions: 128 |
| SimpleRNN | 128 units, `tanh` activation        |
| Dense     | 1 unit, `sigmoid` activation        |

The final sigmoid output represents the probability of the review belonging to the positive class.

```text
Prediction < 0.5  → Negative
Prediction ≥ 0.5  → Positive
```

---

## Dataset

The project uses the **IMDB Movie Review Dataset** provided through TensorFlow/Keras.

| Property                |  Value |
| ----------------------- | -----: |
| Training reviews        | 25,000 |
| Testing reviews         | 25,000 |
| Vocabulary size         | 10,000 |
| Maximum sequence length |    500 |
| Classes                 |      2 |

### Labels

```text
0 → Negative
1 → Positive
```

---

## Data Preprocessing

The IMDB dataset provides reviews as sequences of integer-encoded words.

The vocabulary is restricted to the top 10,000 words:

```python
max_feature = 10000
```

Since movie reviews have different lengths, sequences are padded/truncated to a fixed length of 500:

```python
max_len = 500

x_train = sequence.pad_sequences(
    x_train,
    maxlen=max_len
)

x_test = sequence.pad_sequences(
    x_test,
    maxlen=max_len
)
```

This produces a consistent input shape for the RNN:

```text
(number_of_samples, 500)
```

---

## Training

The model is trained using the Adam optimizer and Binary Crossentropy loss.

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

### Training Configuration

```text
Optimizer        : Adam
Loss Function    : Binary Crossentropy
Batch Size       : 32
Maximum Epochs   : 10
Validation Split : 20%
Early Stopping   : Enabled
```

Early stopping with best-weight restoration is used to reduce overfitting:

```python
earlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)
```

---

## Performance

The model achieves approximately **80% validation accuracy**.

The training process demonstrates the typical behavior of a neural NLP model where training accuracy can continue increasing while validation performance begins to plateau.

Further improvements can be achieved using more advanced recurrent architectures such as **LSTM** or **GRU**.

---

## Streamlit Application

The trained model is integrated with a Streamlit interface for interactive prediction.

### Example

**Input**

```text
This movie was absolutely amazing. I really enjoyed it.
```

**Output**

```text
Sentiment: Positive
Prediction Score: 0.XX
```

The application also accepts negative reviews and returns the corresponding sentiment score.

### Live Demo

**Streamlit App:** `Add your deployed Streamlit URL here`

---

## Project Structure

```text
IMDB-Sentiment-Analysis-RNN/
│
├── main.py
├── simple_Rnn_model.h5
├── simpleRnn.ipynb
├── embedding.ipynb
├── prediction.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

### File Description

| File                  | Description                    |
| --------------------- | ------------------------------ |
| `main.py`             | Streamlit application          |
| `simple_Rnn_model.h5` | Trained RNN model              |
| `simpleRnn.ipynb`     | Model development and training |
| `embedding.ipynb`     | Embedding experiments          |
| `prediction.ipynb`    | Prediction experiments         |
| `requirements.txt`    | Python dependencies            |
| `.gitignore`          | Ignored files and directories  |

---

## Tech Stack

* **Python**
* **TensorFlow / Keras**
* **NumPy**
* **Streamlit**
* **Natural Language Processing**
* **Simple RNN**
* **IMDB Dataset**

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/RAHUL-VERMA-CODE/IMDB-Sentiment-Analysis-RNN.git
```

### 2. Navigate to the project

```bash
cd IMDB-Sentiment-Analysis-RNN
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run Locally

Start the Streamlit application:

```bash
streamlit run main.py
```

The application will be available through the local Streamlit server.

---

## Future Improvements

* Replace Simple RNN with **LSTM / GRU**
* Improve validation accuracy
* Add text normalization and advanced preprocessing
* Improve handling of negation such as `"not good"`
* Add confidence visualization
* Compare multiple deep learning architectures
* Add model performance metrics and confusion matrix
* Containerize the application using Docker
* Deploy the application with a production-ready inference pipeline

---

## Author

### Rahul Verma

B.Tech Computer Science Engineering

**GitHub:**
https://github.com/RAHUL-VERMA-CODE

**LinkedIn:**
https://www.linkedin.com/in/rahul-verma53/

---

## License

This project is intended for educational and portfolio purposes.
