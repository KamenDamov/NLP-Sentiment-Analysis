import gradio as gr
from textblob import TextBlob
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.utils import pad_sequences
import string
import pickle
import re
import langid
import numpy as np


def model(name):
    print("Model running")
    #Define function for new input
    #Add textBlob to have all languages included
    #Prepare parameters
    n_words = 20000 # cut texts after this 
    maxlen = 80
    batch_size = 128
    tokenizer = Tokenizer(num_words=n_words, lower=True)


    with open('Emoji_Dict.p', 'rb') as fp:
        Emoji_Dict = pickle.load(fp)
    Emoji_Dict = {v: k for k, v in Emoji_Dict.items()}

    def convert_emojis_to_word(mess):
        for emot in Emoji_Dict:
            mess = re.sub(r'('+emot+')', "_".join(Emoji_Dict[emot].replace(",","").replace(":","").split()), mess)
        return mess

    str_punc = string.punctuation.replace(',', '').replace("'",'')

    def clean(text):
        print("Cleaning")
        global str_punc
        text = convert_emojis_to_word(text)
        text = re.sub(r'[^a-zA-Z ]', '', text)
        text = text.lower()
        return text


    def predict_new_data(text):
        print("Prediction time")
        def dico(arg): 
            val_dict = {'joy':0,
                        'anger':1,
                        'love':2,
                        'sadness':3,
                        'fear':4,
                        'surprise':5}
            for key, val in val_dict.items():
                if val == arg: 
                    return key

        #Handle data that is not english
        text = TextBlob(text).correct()
        text = str(text)
        lang_detect = str(langid.classify(text))[2:4]
        #print(lang_detect)
        if lang_detect != "en": 
            text = TextBlob(text).translate(from_lang = lang_detect, to = "en")
            #print(text)
            
        text = str(text)
        
        #Prepare data to feed the model
        text = clean(text)
        sen_list = []
        sen_list.append(text) 
        tokenizer.fit_on_texts(sen_list)
        text_seq = tokenizer.texts_to_sequences(sen_list)
        text_pad = pad_sequences(text_seq, maxlen=maxlen)
        model = load_model('LSTM_mod2.h5')
        return print("The emotion predicted by the AI: ", dico(np.argmax(model.predict(text_pad))))

textbox = gr.Textbox(lines=5, placeholder="Human, input text data here.")
output = gr.Textbox()

app = gr.Interface(fn=model,inputs=[textbox],outputs=output)

app.launch()