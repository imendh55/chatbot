from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import uuid
from datetime import datetime
from googletrans import Translator
import nltk
import json
import os
import string
import requests #pour faire des requêtes HTTP externes (API météo, recherche...).
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.qparser import MultifieldParser
from whoosh.query import FuzzyTerm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec

DATASET_PATH = 'votre_dataset.json'
INDEX_DIR = "iset_index"
FEEDBACK_FILE = "feedback.json"
LOCAL_RESPONSE_THRESHOLD = 0.7
WORD2VEC_MODEL_PATH = 'word2vec.model'
WORD_EMBEDDING_DIM = 100
DUCKDUCKGO_API_URL = "https://api.duckduckgo.com/"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
OPENWEATHERMAP_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"  # N'oubliez pas de remplacer par votre clé réelle
OPENWEATHERMAP_API_URL = "https://api.openweathermap.org/data/2.5/weather"
NAVIGATION_THRESHOLD = 0.85

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_history.db'
db = SQLAlchemy(app)
translator = Translator()
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words_fr = set(stopwords.words('french'))
stop_words_en = set(stopwords.words('english'))
stop_words_ar = set(stopwords.words('arabic'))
lemmatizer = WordNetLemmatizer()

schema = Schema(   #Définit le schéma de l'index Whoosh
    id=ID(unique=True, stored=True),
    question=TEXT(stored=True),
    answer=TEXT(stored=True)
)


def get_or_create_index():
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
        ix = create_in(INDEX_DIR, schema)
        writer = ix.writer()
        try:
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                for i, item in enumerate(data):
                    if "question" in item and "réponse" in item:
                        writer.add_document(
                            id=str(i),
                            question=item["question"],
                            answer=item["réponse"]
                        )
                    else:
                        print(f"Skipping item {i} due to missing 'question' or 'réponse'.")
        except FileNotFoundError:
            print(f"Error: The file {DATASET_PATH} was not found. Creating an empty index.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {DATASET_PATH}. Creating an empty index.")
        writer.commit()
    else:
        ix = open_dir(INDEX_DIR)
    return ix


whoosh_index = get_or_create_index()


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), nullable=False, index=True)
    sender = db.Column(db.String(10), nullable=False)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Message {self.id} from {self.sender}>'


with app.app_context():
    db.create_all()


def preprocess_text_basic(text):
    if isinstance(text, str):
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in string.punctuation and w.isalnum()]
        return " ".join(tokens)
    return ""


def preprocess_text(text, lang='fr'):
    if isinstance(text, str):
        text = text.lower()
        tokens = word_tokenize(text)
        if lang == 'fr':
            stop_words = stop_words_fr
        elif lang == 'en':
            stop_words = stop_words_en
        elif lang == 'ar':
            stop_words = stop_words_ar
        else:
            stop_words = set(stopwords.words(lang)) if lang in stopwords.langs() else set()
        tokens = [w for w in tokens if w not in stop_words and w not in string.punctuation and w.isalnum()]
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        return " ".join(tokens)
    return ""


def train_word2vec(sentences, vector_size=WORD_EMBEDDING_DIM, window=5, min_count=1, workers=4):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    model.save(WORD2VEC_MODEL_PATH)
    return model


def load_word2vec():
    if os.path.exists(WORD2VEC_MODEL_PATH):
        return Word2Vec.load(WORD2VEC_MODEL_PATH)
    return None

#Création des Tables de la Base de Données :
def get_word2vec_embedding(text, model, lang='fr', embedding_dim=WORD_EMBEDDING_DIM):
    if model is None:
        return [0] * embedding_dim
    processed_text = preprocess_text(text, lang)
    tokens = word_tokenize(processed_text)
    embeddings = [model.wv[word] for word in tokens if word in model.wv]
    if embeddings:
        return sum(embeddings) / len(embeddings)
    else:
        return [0] * embedding_dim


questions = []
responses = []
question_embeddings_bert = []
question_embeddings_tfidf = []
question_embeddings_word2vec = []
categories = []
category_responses = {}
tfidf_vectorizer = None
knn_model = None
svm_model = None
nb_model = None
label_encoder = None
word2vec_model = load_word2vec()  # Charger le modèle Word2Vec au démarrage
question_embeddings_word2vec = []


def safe_detect_lang(text):
    try:
        if isinstance(text, str) and text.strip():
            detection = translator.detect(text)
            lang = getattr(detection, 'lang', None)
            if lang:
                return lang
    except Exception as e:
        print(f"Erreur de détection de langue : {e}")
    return 'fr'  # Langue par défaut

# Fonction robuste de traduction
def safe_translate(text, dest='fr', src='auto'):
    try:
        if isinstance(text, str) and text.strip():
            translation = translator.translate(text, dest=dest, src=src)
            return getattr(translation, 'text', text)
    except Exception as e:
        print(f"Erreur lors de la traduction de '{text}' vers '{dest}' : {e}")
    return text  # Retourne le texte original en cas d'échec


def load_dataset_and_encode():
    global questions, responses, question_embeddings_bert, question_embeddings_tfidf
    global categories, category_responses
    global tfidf_vectorizer, knn_model, svm_model, nb_model, label_encoder
    global question_embeddings_word2vec

    all_data = []
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                print(f"Erreur: Le fichier {DATASET_PATH} ne contient pas une liste JSON.")
                return
    except FileNotFoundError:
        print(f"Dataset privé '{DATASET_PATH}' non trouvé. Le programme va continuer sans charger de données.")
        return
    except json.JSONDecodeError:
        print(f"Erreur de décodage JSON dans le fichier {DATASET_PATH}. Le programme va continuer sans charger de données.")
        return

    questions.clear()
    responses.clear()
    categories.clear()
    category_responses.clear()
    question_embeddings_bert.clear()
    question_embeddings_tfidf.clear()
    question_embeddings_word2vec.clear()

    for item in all_data:
        if 'question' in item and 'réponse' in item:
            questions.append(item['question'])
            responses.append(item['réponse'])
            if 'category' in item and 'category_response' in item:
                categories.append(item['category'])
                if item['category'] not in category_responses:
                    category_responses[item['category']] = item['category_response']

    if not questions:
        print("Aucune donnée de question-réponse chargée. Le programme va continuer sans charger de données.")
        return

    try:
        question_embeddings_bert = embedding_model.encode(questions)
    except Exception as e:
        print(f"Erreur lors de l'encodage BERT: {e}. Le programme va continuer sans les embeddings BERT.")
        question_embeddings_bert = []

    try:
        tfidf_vectorizer = TfidfVectorizer(preprocessor=lambda x: preprocess_text(x, safe_detect_lang(x)))
        question_embeddings_tfidf = tfidf_vectorizer.fit_transform(questions)
    except Exception as e:
        print(f"Erreur lors de l'vectorisation TF-IDF: {e}. Le programme va continuer sans les embeddings TF-IDF.")
        tfidf_vectorizer = None
        question_embeddings_tfidf = []

    if word2vec_model:
        try:
            question_embeddings_word2vec = [get_word2vec_embedding(q, word2vec_model, safe_detect_lang(q)) for q in questions]
        except Exception as e:
            print(f"Erreur lors de la création des embeddings Word2Vec: {e}. Le programme va continuer sans ces embeddings.")
            question_embeddings_word2vec = []

    if categories and question_embeddings_tfidf is not None and question_embeddings_tfidf.shape[0] > 0:
        try:
            label_encoder = LabelEncoder()
            encoded_categories = label_encoder.fit_transform(categories)

            knn_model = KNeighborsClassifier(n_neighbors=5)
            knn_model.fit(question_embeddings_tfidf, encoded_categories)

            svm_model = SVC(kernel='linear', C=1)
            svm_model.fit(question_embeddings_tfidf, encoded_categories)

            nb_model = MultinomialNB()
            nb_model.fit(question_embeddings_tfidf, encoded_categories)
        except Exception as e:
            print(f"Erreur lors de l'entraînement des classifieurs: {e}. Le programme va continuer sans les classifieurs.")
            knn_model = None
            svm_model = None
            nb_model = None
    else:
        print("Les catégories ou les embeddings TF-IDF ne sont pas disponibles pour l'entraînement des classifieurs.")
        knn_model = None
        svm_model = None
        nb_model = None


load_dataset_and_encode()


def search_duckduckgo(query, lang):
    params = {'q': query, 'format': 'json', 'pretty': 1, 'no_html': 1, 'l': lang}
    try:
        response = requests.get(DUCKDUCKGO_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if data and data.get('AbstractText'):
            return data['AbstractText']
        elif data and data.get('Answer'):
            return data['Answer']
        elif data and data.get('RelatedTopics'):
            for topic in data['RelatedTopics']:
                if isinstance(topic, dict) and topic.get('Text'):
                    return topic['Text']
        return None
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête à l'API DuckDuckGo: {e}")
        return None


def search_wikidata(query, lang):
    params = {'action': 'query', 'format': 'json', 'list': 'search', 'srlimit': 1, 'srprop': 'snippet',
              'srsearch': query, 'uselang': lang, 'formatversion': 2}
    try:
        response = requests.get(WIKIDATA_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get('query') and data['query'].get('search') and data['query']['search']:
            return data['query']['search'][0].get('snippet')
        return None
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête à l'API WikiData: {e}")
        return None


def get_weather(city, lang):
    if not OPENWEATHERMAP_API_KEY:
        print("Clé API OpenWeatherMap non configurée.")
        return None
    params = {'q': city, 'appid': OPENWEATHERMAP_API_KEY, 'lang': lang, 'units': 'metric'}
    try:
        response = requests.get(OPENWEATHERMAP_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if data and data.get('weather'):
            weather_desc = data['weather'][0].get('description')
            temp = data.get('main').get('temp')
            return f"La météo à {city} est : {weather_desc}, température : {temp}°C."
        return None
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête à l'API OpenWeatherMap: {e}")
        return None


def save_feedback(question, response, feedback):
    feedback_data = {
        "question": question,
        "response": response,
        "feedback": feedback,
        "timestamp": datetime.now().isoformat()
    }
    try:
        with open(FEEDBACK_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return False


def get_response(user_question, target_lang='fr'):
    local_response = None
    original_response = None
    detected_language = target_lang
    lower_question = user_question.lower()
    proactive_suggestion = None
    navigation_link = None
    best_similarity_score = 0.0

    detected_language = safe_detect_lang(user_question)
    print(f"Langue détectée pour '{user_question}': {detected_language}")
    if not detected_language:
        detected_language = target_lang
        print(f"Aucune langue détectée, utilisation de la langue cible : {detected_language}")

    greetings = {
        'fr': ["bonjour", "salut", "bonsoir", "bonne journée", "bonne soirée"],
        'en': ["hi", "hello", "good morning", "good afternoon", "good evening", "good day", "hey"],
        'ar': ["مرحبا", "أهلاً", "صباح الخير", "مساء الخير", "نهارك سعيد", "مساء سعيد"]
    }

    if detected_language in greetings and any(g in lower_question for g in greetings[detected_language]):
        greeting_response = safe_translate(f"Bonjour ! Comment puis-je vous aider ?", dest=detected_language, src='fr')
        return greeting_response, user_question, greeting_response, proactive_suggestion, navigation_link, 1.0

    local_response_found = False
    # Recherche avec BERT#  Recherche la question la plus similaire 
    if len(question_embeddings_bert) > 0:
        try:
            question_embedding = embedding_model.encode([user_question])[0]
            similarity_scores = cosine_similarity([question_embedding], question_embeddings_bert)[0]
            best_index = similarity_scores.argmax()
            best_similarity_score = similarity_scores[best_index]
            if best_similarity_score >= LOCAL_RESPONSE_THRESHOLD:
                original_response = responses[best_index]
                print(f"Réponse locale BERT trouvée (avant traduction): {original_response}, langue détectée: {detected_language}")
                if detected_language != 'fr':
                    print(f"Traduction de '{original_response}' vers '{detected_language}'")
                    local_response = safe_translate(original_response, dest=detected_language)
                    print(f"Réponse locale BERT traduite: {local_response}")
                else:
                    local_response = original_response
                local_response_found = True
                print(f"Réponse locale trouvée (BERT, langue: {detected_language}) : {local_response} (Similarité : {best_similarity_score})")
                return local_response, user_question, original_response, proactive_suggestion, navigation_link, best_similarity_score
        except Exception as e:
            print(f"Erreur lors de la recherche locale (BERT) : {e}")

    # Recherche avec KNN#recherche en utilisant un modèle KNN entraîné sur les vecteurs TF-IDF.
    if not local_response_found and knn_model and tfidf_vectorizer:
        try:
            processed_question = preprocess_text(user_question, detected_language)
            question_vector = tfidf_vectorizer.transform([processed_question])
            distances, indices = knn_model.kneighbors(question_vector, n_neighbors=1)
            if 1 - distances[0][0] >= LOCAL_RESPONSE_THRESHOLD:
                closest_index = indices[0][0]
                original_response = responses[closest_index]
                best_similarity_score = 1 - distances[0][0]
                print(f"Réponse locale KNN trouvée (avant traduction): {original_response}, langue détectée: {detected_language}")
                if detected_language != 'fr':
                    print(f"Traduction de '{original_response}' vers '{detected_language}'")
                    local_response = safe_translate(original_response, dest=detected_language)
                    print(f"Réponse locale KNN traduite: {local_response}")
                else:
                    local_response = original_response
                local_response_found = True
                print(f"Réponse locale trouvée (KNN, langue: {detected_language}) : {local_response} (Similarité approx. : {best_similarity_score})")
                return local_response, user_question, original_response, proactive_suggestion, navigation_link, best_similarity_score
        except Exception as e:
            print(f"Erreur lors de la recherche locale (KNN) : {e}")

    # Recherche avec Whoosh
    if not local_response_found and whoosh_index:
        try:
            with whoosh_index.reader() as reader:
                fields = ["question", "answer"]
                parser = MultifieldParser(fields, schema=whoosh_index.schema)
                query = parser.parse(user_question)
                results = reader.search(query, limit=1)
                if results and results[0].score >= LOCAL_RESPONSE_THRESHOLD:
                    local_response = results[0]['answer']
                    original_response = local_response  # Puisqu'il vient directement de l'index
                    best_similarity_score = results[0].score  # Le score Whoosh
                    print(f"Réponse locale Whoosh trouvée (avant traduction): {original_response}, langue détectée: {detected_language}")
                    if detected_language != 'fr':
                        print(f"Traduction de '{original_response}' vers '{detected_language}'")
                        local_response = safe_translate(local_response, dest=detected_language)
                        print(f"Réponse locale Whoosh traduite: {local_response}")
                    else:
                        local_response = original_response
                    local_response_found = True
                    print(f"Réponse locale trouvée (Whoosh, langue: {detected_language}) : {local_response} (Score : {best_similarity_score})")
                    return local_response, user_question, original_response, proactive_suggestion, navigation_link, best_similarity_score
        except Exception as e:
            print(f"Erreur lors de la recherche locale (Whoosh) : {e}")

    # Recherche externe
    if not local_response_found:
        print(f"Aucune réponse locale pertinente trouvée en {detected_language}. Tentative de recherche externe via DuckDuckGo...")
        external_response = search_duckduckgo(user_question, detected_language)
        if external_response:
            print(f"Réponse trouvée via DuckDuckGo ({detected_language}) : {external_response}")
            return external_response, user_question, "Réponse trouvée via DuckDuckGo", proactive_suggestion, navigation_link, 0.0
        else:
            print(f"DuckDuckGo n'a pas trouvé de réponse en {detected_language}. Tentative via WikiData...")
            wikidata_response = search_wikidata(user_question, detected_language)
            if wikidata_response:
                print(f"Réponse trouvée via WikiData ({detected_language}) : {wikidata_response}")
                return wikidata_response, user_question, "Réponse trouvée via WikiData", proactive_suggestion, navigation_link, 0.0
            else:
                print(f"Aucune réponse pertinente trouvée ni via DuckDuckGo ni via WikiData en {detected_language}.")
                not_found_message = safe_translate("Désolé, je n'ai pas trouvé d'informations pertinentes pour votre question.", dest=detected_language, src='fr')
                return not_found_message, user_question, "Aucune réponse trouvée", proactive_suggestion, navigation_link, 0.0


def get_or_create_session_id():
    session_id = request.cookies.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id


@app.route('/chatbot', methods=['POST'])
def chatbot_endpoint():
    data = request.get_json()
    user_question = data.get('question')
    lang = data.get('lang', 'fr')

    if not user_question:
        return jsonify({'error': 'Question manquante'}), 400

    session_id = get_or_create_session_id()
    response, original_question, original_response, proactive_suggestion, navigation_link, similarity_score = get_response(
        user_question, lang)

    if isinstance(similarity_score, (float, int)):
        serializable_similarity_score = float(similarity_score)
    else:
        try:
            serializable_similarity_score = float(similarity_score)
        except TypeError:
            serializable_similarity_score = 0.0

    new_message = Message(session_id=session_id, sender='user', text=user_question)
    db.session.add(new_message)
    new_message_bot = Message(session_id=session_id, sender='bot', text=response)
    db.session.add(new_message_bot)
    db.session.commit()

    resp = make_response(jsonify({
        'response': response,
        'original_question': original_question,
        'original_response': original_response,
        'proactive_suggestion': proactive_suggestion,
        'navigation_link': navigation_link,
        'similarity_score': serializable_similarity_score,
        'session_id': session_id,
        'lang': lang  # Send the language back to the client
    }))
    resp.set_cookie('session_id', session_id)
    return resp


@app.route('/history/<session_id>', methods=['GET'])
def get_chat_history(session_id):
    messages = Message.query.filter_by(session_id=session_id).order_by(Message.timestamp).all()
    history = [{'sender': msg.sender, 'text': msg.text, 'timestamp': msg.timestamp.isoformat()} for msg in messages]
    return jsonify(history)


feedback_data = []


@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    question = data.get('question')
    response = data.get('response')
    feedback_text = data.get('feedback')
    session_id = data.get('session_id')

    success = save_feedback(question, response, feedback_text)

    if success:
        return jsonify({'status': 'success', 'message': 'Feedback received'})
    else:
        return jsonify({'status': 'error', 'message': 'Erreur lors de l’enregistrement du feedback'}), 500


@app.route('/feedbacks', methods=['GET'])
def get_feedbacks():
    feedbacks = []
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    feedbacks.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    return jsonify(feedbacks)


@app.route('/learn', methods=['POST'])
def learn_endpoint():
    data = request.get_json()
    if not all(k in data for k in ('new_question', 'new_response')):
        return jsonify({'error': 'Champs manquants'}), 400

    new_question = data['new_question']
    new_response = data['new_response']
    new_entry = {'question': new_question, 'réponse': new_response}

    try:
        with open(DATASET_PATH, 'r+', encoding='utf-8') as f:
            existing_data = json.load(f)
            existing_data.append(new_entry)
            f.seek(0)
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
            f.truncate()

        # Mise à jour de l'index Whoosh de manière incrémentale
        ix = open_dir(INDEX_DIR)
        writer = ix.writer()
        try:
            new_id = str(len(existing_data) - 1)
            writer.add_document(
                id=new_id,
                question=new_question,
                answer=new_response
            )
            writer.commit()
        except Exception as e:
            writer.cancel()
            print(f"Erreur lors de la mise à jour de l'index Whoosh: {e}")
            return jsonify({'error': 'Erreur lors de la mise à jour de l\'index'}), 500

        load_dataset_and_encode()
        return jsonify({'message': 'Nouvelle donnée ajoutée et indexée avec succès'})
    except Exception as e:
        print(f"Erreur lors de l'apprentissage: {e}")
        return jsonify({'error': 'Erreur d’enregistrement'}), 500


if __name__ == '__main__':
    app.run(debug=True)