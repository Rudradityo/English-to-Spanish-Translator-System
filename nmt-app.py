'''
Technical English to Spanish Translation in Natural Language Processing Domain
'''

# Import Libraries
from flask import Flask, request, render_template
import nltk
from transformers import MarianMTModel, MarianTokenizer
import re
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


app = Flask(__name__)


def preprocess_english_sentence(sentence, technical_jargon_list):
    
    # Lowercase sentence and remove leading and trailing whitespaces
    sentence = sentence.lower().strip()
    
    # Remove punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    
    # Remove special characters
    sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
    
    # Remove numbers
    sentence = re.sub(r'\d+', '', sentence)
    
    # Tokenize the sentence into words
    tokens = word_tokenize(sentence)
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Remove technical jargons
    tokens = [word for word in tokens if word not in technical_jargon_list]
    
    # Remove leading and trailing whitespaces and join back into a single string
    preprocessed_sentence = re.sub(r'\s+', ' ', ' '.join(tokens))
    
    return preprocessed_sentence

def preprocess_english_text(english_text, technical_jargon_list):
    
    # Split text into sentences
    english_sentences = sent_tokenize(english_text)
    
    # Process each sentence individually
    preprocessed_english_sentences = [preprocess_english_sentence(english_sentence, technical_jargon_list) for english_sentence in english_sentences]
    # Join processed sentences back into a single string
    preprocessed_english_text = ' '.join(preprocessed_english_sentences)
    
    return preprocessed_english_text


# Function to translate text from English to Spanish
def translate_english_to_spanish(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


def load_tokenizer_model(model_name):
    return MarianTokenizer.from_pretrained(model_name), MarianMTModel.from_pretrained(model_name)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate():
    text_input = request.form.get('text')
    file_input = request.files.get('file')

    # Load the tokenizer and model
    model_name = 'Helsinki-NLP/opus-mt-en-es'  # English to Spanish Transformer Model
    tokenizer, model = load_tokenizer_model(model_name)

    # List of Technical Jargons to remove from Lemmatized English for Simpler translation to Spanish
    technical_jargon_list = ["natural language process", "natural language understand", "natural language generate", "tokenize", "lemmatize", "normalize"]

    if text_input:
        preprocessed_english_text = preprocess_english_text(text_input, technical_jargon_list)
        translated_spanish_text = translate_english_to_spanish(preprocessed_english_text, tokenizer, model)
    elif file_input:
        file_content = file_input.read().decode('utf-8')
        preprocessed_english_text = preprocess_english_text(file_content, technical_jargon_list)
        translated_spanish_text = translate_english_to_spanish(preprocessed_english_text, tokenizer, model)
    else:
        return "No input provided", 400

    return translated_spanish_text


if __name__ == "__main__":
    app.run(debug=True)