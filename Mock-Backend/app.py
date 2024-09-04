import random
from flask import Flask, jsonify, request, session
from flask_cors import CORS
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pyttsx3
import sounddevice as sd
import speech_recognition as sr
import threading
import google.generativeai as genai

app = Flask(__name__)
app.config.from_pyfile('config.py', silent=True)
app.secret_key = "amile_interview"  # Necessary for session management
CORS(app)

# MongoDB connection
client = MongoClient('mongodb+srv://vignaramtejtelagarapu:vzNsqoKpAzHRdN9B@amile.auexv.mongodb.net/?retryWrites=true&w=majority&appName=Amile')
db = client['test']
mentors_collection = db['mentors']
students_collection = db['students']

# Initialize Google Generative AI
genai.configure(api_key="AIzaSyC4dc-N80zxk2UZCCOA0oMN94YVT12SVW8")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

def fetch_profiles(collection):
    """Fetches the profiles from MongoDB collection."""
    profiles = list(collection.find({}, {'_id': 0, 'skills': 1, 'username': 1}))
    for profile in profiles:
        if isinstance(profile['skills'], list):
            profile['skills'] = ', '.join(profile['skills'])
    return profiles

def compute_similarity(mentors, students):
    """Computes cosine similarity between mentor and student profiles."""
    all_profiles = [mentor['skills'] for mentor in mentors] + [student['skills'] for student in students]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_profiles)
    cosine_sim = cosine_similarity(tfidf_matrix[:len(mentors)], tfidf_matrix[len(mentors):])
    return cosine_sim

def get_best_match_for_mentors(mentors, students, similarity_matrix, threshold=0.2):
    """Get the best matching students for each mentor, only if the similarity score is above the threshold."""
    results = {mentor['username']: [] for mentor in mentors}
    for student_index, student in enumerate(students):
        best_mentor_index = np.argmax(similarity_matrix[:, student_index])
        best_mentor = mentors[best_mentor_index]
        best_score = similarity_matrix[best_mentor_index, student_index]
        
        if best_score > threshold:
            results[best_mentor['username']].append({"student": student['username'], "score": best_score})
    
    return results

@app.route('/match', methods=['GET'])
def match_mentors_students():
    mentors = fetch_profiles(mentors_collection)
    students = fetch_profiles(students_collection)

    if not mentors or not students:
        return jsonify({"error": "Mentors or students data is missing"}), 404

    similarity_matrix = compute_similarity(mentors, students)
    results = get_best_match_for_mentors(mentors, students, similarity_matrix)

    return jsonify(results), 200

def get_best_match_for_students(mentors, students, similarity_matrix, threshold=0.2):
    """Get the best matching mentors for each student, only if the similarity score is above the threshold."""
    results = {student['username']: [] for student in students}
    for mentor_index, mentor in enumerate(mentors):
        best_student_index = np.argmax(similarity_matrix[mentor_index])
        best_student = students[best_student_index]
        best_score = similarity_matrix[mentor_index, best_student_index]
        
        if best_score > threshold:
            results[best_student['username']].append({"mentor": mentor['username'], "score": best_score})
    
    return results

@app.route('/match-students', methods=['GET'])
def match_students_to_mentors():
    mentors = fetch_profiles(mentors_collection)
    students = fetch_profiles(students_collection)

    if not mentors or not students:
        return jsonify({"error": "Mentors or students data is missing"}), 404

    similarity_matrix = compute_similarity(mentors, students)
    results = get_best_match_for_students(mentors, students, similarity_matrix)
    
    student_username = request.args.get('username')
    mentor_index = int(request.args.get('index', 0))

    if student_username in results:
        mentor_matches = results[student_username]
        if mentor_matches:
            # Use modulo to wrap around the index if it exceeds the array length
            mentor_index = mentor_index % len(mentor_matches)
            return jsonify(mentor_matches[mentor_index]), 200
        else:
            random_mentor = random.choice(mentors)
            return jsonify({"mentor": random_mentor['username'], "score": "Random assignment"}), 200
    else:
        return jsonify({"error": "Student not found"}), 404



def record_audio(duration, sample_rate):
    frames = int(duration * sample_rate)  # Calculate the number of frames
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    return audio

def normalize_audio(audio):
    return np.int16(audio / np.max(np.abs(audio)) * 32767)

def transcribe_audio(audio, sample_rate):
    recognizer = sr.Recognizer()
    normalized_audio = normalize_audio(audio)
    audio_data = sr.AudioData(normalized_audio.tobytes(), sample_rate, 2)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        return None

def ask_question():
    prompt = "You are an interviewer now. Introduce yourself as a Manager in Amile and your name is Hakunamatata and start the interview."
    response = model.generate_content(prompt)
    return response.text

def generate_followup(interview_question, user_response, conversation_history):
    # Include conversation history in the prompt
    history_prompt = "\n".join([f"Q: {item['question']} A: {item['response']}" for item in conversation_history])
    prompt = f"Here is the conversation so far:\n{history_prompt}\nYou asked the candidate: '{interview_question}'. The candidate responded: '{user_response}'. Now respond as an interviewer with a precise question."
    response = model.generate_content(prompt)
    return response.text

def speak_question(question):
    try:
        engine.say(question)
        engine.runAndWait()
    except RuntimeError:
        engine.endLoop()  # End the loop if it's already running
        engine.say(question)
        engine.runAndWait()

@app.route('/ask-question', methods=['GET'])
def get_question():
    # Fetch the first question
    question = ask_question()

    # Initialize session to store conversation history
    if 'conversation_history' not in session:
        session['conversation_history'] = []

    # Start a new thread for speech synthesis to avoid blocking the response
    threading.Thread(target=speak_question, args=(question,)).start()

    return jsonify({"question": question})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    duration = request.json.get('duration')
    sample_rate = request.json.get('sample_rate')
    interview_question = request.json.get('question')

    audio = record_audio(duration, sample_rate)
    user_response = transcribe_audio(audio, sample_rate)

    if user_response:
        # Get the conversation history from the session
        conversation_history = session.get('conversation_history', [])

        # Add the current question and response to the conversation history
        conversation_history.append({"question": interview_question, "response": user_response})
        session['conversation_history'] = conversation_history

        # Generate the follow-up question based on conversation history
        followup = generate_followup(interview_question, user_response, conversation_history)

        # Start a new thread for speech synthesis
        threading.Thread(target=speak_question, args=(followup,)).start()

        return jsonify({"response": followup, "audio": user_response})
    else:
        return jsonify({"error": "Could not understand the audio"}), 400

@app.route('/feedback', methods=['POST'])
def get_feedback():
    conversation_history = request.json.get('conversation_history')
    history_prompt = "\n".join([f"Q: {item['question']} A: {item['response']}" for item in conversation_history])
    prompt = f"Here is the conversation so far:\n{history_prompt}\n What score would you give for the user in the interview? where the responses are from the user and the questions asked by the interviewer and just give me the total score in number with no other response please. I just need number."
    response = model.generate_content(prompt)
    return response.text

@app.route('/display_feedback', methods=['GET'])
def display_feedback():
    feedback = request.json.get('feedback')
    return feedback

if __name__ == '__main__':
    app.run(debug=True)
