from flask import Flask, render_template, request, jsonify
import nltk
import os
import json
import torch
import numpy as np
from data.preprocess import tokenize, bag_of_words
from models.model import NeuralNet
from helper import chat_response



os.environ["NLTK_DATA"] = os.path.join(os.path.dirname(__file__), "nltk")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

FILE = 'optimus.pth'
data = torch.load(FILE)

with open("intents.json", 'r') as f:
    intents =json.load(f)

input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "optimus"


# while True:
#     userMessage = input("You: ")
#     if userMessage.lower() == "exit":
#         print("Goodbye!")
#         break

#     print("Processing...")
#     try:
#         response = chat_response(userMessage, model, intents, all_words, tags)
#         if not response:
#             response = "I'm sorry, I didn't understand that."
#         print("Optimus Prime:", response)
#     except Exception as e:
#         print("Error processing your message:", e)


    

app = Flask(__name__)



@app.route('/')
def home():

    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    msg = data.get('message', "")
    response = chat_response(msg, model, intents, all_words, tags)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)