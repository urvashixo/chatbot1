# AI Chatbot using Neural Networks (PyTorch)

A simple intent-based chatbot built with **Python**, **PyTorch**, **NLTK**, and a JSON-based intent dataset. The chatbot uses a feedforward neural network to classify user input into predefined intents and return an appropriate response.

## Features

- Intent-based conversation
- Feedforward Neural Network (PyTorch)
- Text preprocessing with NLTK
- JSON-based training data (`intents.json`)
- Bag-of-Words representation
- Easily customizable intents and responses

## Technologies Used

- Python 
- PyTorch
- NLTK
- NumPy
- JSON

## Project Structure

```
chatbot/
│── chatbot.py          # Main chatbot script
│── intents.json        # Intent dataset
│── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone repolink
cd chatbot
```


### 2. Install dependencies

```bash
pip install torch numpy nltk
```



## Dataset Format

The chatbot uses an `intents.json` file structured like this:

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi",
        "Hello",
        "Hey"
      ],
      "responses": [
        "Hello!",
        "Hi there!",
        "How can I help you?"
      ]
    }
  ]
}
```

### Fields

- **tag** – Intent label.
- **patterns** – Example user inputs.
- **responses** – Possible chatbot replies.

## Model Architecture

The neural network consists of three fully connected layers:

```
Input Layer
      │
      ▼
Linear (Input → 128)
      │
    ReLU
      │
      ▼
Linear (128 → 64)
      │
    ReLU
      │
      ▼
Linear (64 → Output Classes)
```

The output layer predicts the probability of each intent.

## Text Processing

The chatbot preprocesses user input by:

1. Tokenizing sentences using NLTK.
2. Converting words to lowercase.
3. Creating a Bag-of-Words vector.
4. Feeding the vector into the neural network.

## Training

Train the chatbot using:

```bash
python chatbot.py
```

During training:

- The model learns from the patterns in `intents.json`.
- Loss decreases over multiple epochs.
- The trained model is saved as `model.pth`.

## Running the Chatbot

After training:

```bash
python chatbot.py
```

Example conversation:

```
You: Hello
Bot: Hi there!

You: What can you do?
Bot: I can answer questions based on my training.

You: Bye
Bot: Goodbye! Have a great day!
```

## Hyperparameters

| Parameter | Value |
|----------|------:|
| Hidden Layer 1 | 128 neurons |
| Hidden Layer 2 | 64 neurons |
| Activation | ReLU |
| Framework | PyTorch |

## Dependencies

```
torch
numpy
nltk




