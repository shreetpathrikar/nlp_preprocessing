from utils import *

app = Flask(__name__)

#logic or operational function

@app.route('/',methods=['GET']) 
def index():
    return render_template('index.html')
 

def remove_num(text):
    result = re.sub(r'\d+', '', text)
    return result

def convert_num(text):
    temp_string = text.split() 
    new_str = [] 
    for word in temp_string: 
        if word.isdigit(): 
            temp = q.number_to_words(word) 
            new_str.append(temp) 
        else: 
            new_str.append(word) 
    temp_str = ' '.join(new_str) 
    return temp_str 

def rem_punct(text): 
    translator = str.maketrans(' ', ' ', string.punctuation) 
    return text.translate(translator) 

def rem_stopwords(text): 
    stop_words = set(stopwords.words("english")) 
    word_tokens = word_tokenize(text) 
    filtered_text = [word for word in word_tokens if word not in stop_words] 
    return filtered_text 

def word_tokenizer(text):
    tokens = word_tokenize(text)
    return tokens

def sentence_tokenizer(text):
    sentences = sent_tokenize(text)
    return sentences

stem1 = PorterStemmer() 
def s_words(text): 
    word_tokens = word_tokenize(text) 
    stems = [stem1.stem(word) for word in word_tokens] 
    return stems 

lemma = wordnet.WordNetLemmatizer()
def lemmatize_word(text): 
    word_tokens = word_tokenize(text) 
    lemmas = [lemma.lemmatize(word, pos ='v') for word in word_tokens] 
    return lemmas 
  
  
def pos_tagg(text): 
    word_tokens = word_tokenize(text) 
    return pos_tag(word_tokens) 


def ner(text): 
    word_tokens = word_tokenize(text) 
    word_pos = pos_tag(word_tokens) 
    return (ne_chunk(word_pos)) 


def word_count(text):
    words =word_tokenize(text)
    count = len(words)
    return count


def calculate_frequency(sentence):
    words = word_tokenize(sentence)
    fdist = FreqDist(words)
    word_count_list = [f"{word}: {count}" for word, count in fdist.items()]

    return word_count_list


#render user input from html for VIEW CODE

@app.route('/remove__num')
def user1_html():
    return render_template('remove__num.html')

@app.route('/convert_num')
def user2_html():
    return render_template('convert__num.html')

@app.route('/remove__punctuation')
def user3_html():
    return render_template('remove__punc.html')

@app.route('/remove__stopwords')
def user4_html():
    return render_template('remove__stop.html')

@app.route('/word_tokenizer')
def user5_html():
    return render_template('word_tokenize.html')

@app.route('/sentence_tokenizer')
def user6_html():
    return render_template('sentence_tokenize.html')

@app.route('/stemming')
def user7_html():
    return render_template('stemming_.html')

@app.route('/lemmitizing')
def user8_html():
    return render_template('lemmitizing.html')
 

@app.route('/POSTAG')
def user9_html():
    return render_template('pos_tagging.html')
 
@app.route('/NER')
def user10_html():
    return render_template('ner_.html')

@app.route('/word_count')
def user11_html():
    return render_template('word_count.html')


@app.route('/frequency_dist')
def user12_html():
    return render_template('feq_dis.html')

@app.route('/spacy_tokens')
def user13_html():
    return render_template('spacy_tok.html')

@app.route('/spacy_postag')
def user14_html():
    return render_template('spacy_tag.html')

@app.route('/spacy_wordvector')
def user15_html():
    return render_template('spacy_w2v.html')

@app.route('/spacy_NER')
def user16_html():
    return render_template('spacy_nameER.html')

@app.route('/textblob_correction')
def user17_html():
    return render_template('txtb_correc.html')

@app.route('/textblob_lang_translation')
def user18_html():
    return render_template('txtb_lang_trans.html')

@app.route('/textblob_sentiment_analyzer')
def user19_html():
    return render_template('txtb_senti_anlzr.html')


#storing user input in ariable and pass it to operational function

@app.route('/process', methods=['POST'])
def process():
    user1_input = request.form['user1_input']
    user2_input = request.form['user2_input']
    user3_input = request.form['user3_input']
    user4_input = request.form['user4_input']
    user5_input = request.form['user5_input']
    user6_input = request.form['user6_input']
    user7_input = request.form['user7_input']
    user8_input = request.form['user8_input']
    user9_input = request.form['user9_input']
    user10_input = request.form['user10_input']
    user11_input = request.form['user11_input']
    user12_input = request.form['user12_input']
    
    processed_input1 = remove_num(user1_input)
    processed_input2 = convert_num(user2_input)
    processed_input3 = rem_punct(user3_input)
    processed_input4 = rem_stopwords(user4_input)
    processed_input5 = word_tokenizer(user5_input)
    processed_input6 = sentence_tokenizer(user6_input)
    processed_input7 = s_words(user7_input)
    processed_input8 = lemmatize_word(user8_input)
    processed_input9 = pos_tagg(user9_input)
    processed_input10 = ner(user10_input)
    processed_input11 = word_count(user11_input)
    processed_input12 = calculate_frequency(user12_input )
    

    if processed_input1:
        return f"Processed input: {processed_input1}"
    elif processed_input2:
        return f"Processed input: {processed_input2}"
    elif processed_input3:
        return f"Processed input: {processed_input3}"
    elif processed_input4:
        return f"Processed input: {processed_input4}"
    elif processed_input5:
        return f"Processed input: {processed_input5}"
    elif processed_input6:
        return f"Processed input: {processed_input6}"
    elif processed_input7:
        return f"Processed input: {processed_input7}"
    elif processed_input8:
        return f"Processed input: {processed_input8}"
    elif processed_input9:
        return f"Processed input: {processed_input9}"
    elif processed_input10:
        return f"Processed input: {processed_input10}"
    elif processed_input11:
        return f"Processed input: {processed_input11}"
    elif processed_input12:
        return f"Processed input: {processed_input12}"

    
    else:
        print("something went wrong try again")














# SPACY
@app.route('/spacy/')
def spacy():
    return render_template('spacy.html')

def token_spc(text):
    # nlp = spacy.load("en_core_web_sm")   loaded in utils.py
    doc = nlp(text)
    tokens = []
    for token in doc:
        tokens.append(token.text)
    return tokens

def pos_tagging(text):
    # nlp = spacy.load("en_core_web_sm")      loaded in utils.py
    doc = nlp(text)
    pos_tags = []
    for token in doc:
        pos_tags.append((token.text, token.pos_))
    return pos_tags


def word_similarity(word1, word2):
    token1 = nlp1(word1)
    token2 = nlp1(word2)
    if token1.has_vector and token2.has_vector:
        similarity = token1.similarity(token2)
        return similarity
    else:
        return None 
    

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities
       
        
        
        
        
        
        
        
@app.route('/spacyy', methods=['POST'])
def spacyy(): 
    
    user13_input = request.form['user13_input']
    user14_input = request.form['user14_input']
    user15a_input = request.form['user15a_input']
    user15b_input = request.form['user15b_input']
    user16_input = request.form['user16_input']
    
    
    
    processed_input13 = token_spc(user13_input)
    processed_input14 = pos_tagging(user14_input)
    processed_input15 = word_similarity(user15a_input,user15b_input)
    processed_input16 = extract_entities(user16_input)
    
    
    
    if processed_input13:
        return f"Processed input: {processed_input13}"
    elif processed_input14:
        return f"Processed input: {processed_input14}"
    elif processed_input15:
        return f"Processed input: {processed_input15}"
    elif processed_input16:
        return f"Processed input: {processed_input16}"
    else:
        print("try again ") 
    
























@app.route('/textblob/')
def textblob():
    return render_template('textblob.html')

def correct_text(input_text):
    text_blob = TextBlob(input_text)
    corrected_text = text_blob.correct()
    return corrected_text

def translate_text(text, target_language):
    try:
        blob = TextBlob(text)
        translated_text = blob.translate(from_lang='en', to=target_language)
        return translated_text
    except Exception as e:
        print(f"Error during translation: {e}")
        return None

def get_sentiment(text):
    blob1 = TextBlob(text, analyzer=NaiveBayesAnalyzer())
    sentiment = blob1.sentiment
    return sentiment


    


        
@app.route('/textblobs', methods=['POST'])
def textblobs(): 
    
    user17_input = request.form['user17_input']
    user18a_input = request.form['user18a_input']
    user18b_input = request.form['user18b_input']
    user19_input = request.form['user19_input']
   
    

    
    processed_input17 = correct_text(user17_input)
    processed_input18 = translate_text(user18a_input,user18b_input)
    processed_input19 = get_sentiment(user19_input)
    
    
    
    
    
    
    
    
    
    
    
    if processed_input17:
        return f"Processed input: {processed_input17}"
    elif processed_input18:
        return f"Processed input: {processed_input18}"
    elif processed_input19:
        return f"Processed input: {processed_input19}"
    
    else:
        print("try again ")
    
    












if __name__ == '__main__':
    app.run(debug=True)