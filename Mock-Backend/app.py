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
    """Fetches the profiles from MongoDB collection and processes skills."""
    profiles = list(collection.find({}, {'_id': 0, 'skills': 1, 'username': 1}))
    for profile in profiles:
        if isinstance(profile['skills'], list):
            # Convert skills to lowercase and join them as a string
            profile['skills'] = ', '.join([skill.lower() for skill in profile['skills']])
    return profiles

def compute_similarity(mentors, students):
    """Computes cosine similarity between mentor and student profiles."""
    # Combine mentor and student skills, ensuring lowercase conversion
    all_profiles = [mentor['skills'] for mentor in mentors] + \
                   [student['skills'] for student in students]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_profiles)
    # Compute cosine similarity matrix between mentors and students
    cosine_sim = cosine_similarity(tfidf_matrix[:len(mentors)], tfidf_matrix[len(mentors):])
    
    return cosine_sim

def get_best_match_for_mentors(mentors, students, cosine_sim):
    """Returns the best student matches for each mentor based on similarity, with a threshold."""
    mentor_matches = {}
    threshold = 0.2  # Similarity score threshold
    for i, mentor in enumerate(mentors):
        best_matches = [
            {"student": students[j]['username'], "score": cosine_sim[i][j]}
            for j in range(len(students)) if cosine_sim[i][j] > threshold
        ]
        # Sort matches by highest score
        best_matches.sort(key=lambda x: x["score"], reverse=True)
        mentor_matches[mentor['username']] = best_matches
    return mentor_matches

def get_best_match_for_students(students, mentors, cosine_sim):
    """Returns the best mentor matches for each student based on similarity, with a threshold."""
    student_matches = {}
    threshold = 0.2  # Similarity score threshold

    # Iterate over the students and check that the cosine_sim dimensions are correct
    for j, student in enumerate(students):
        best_matches = []
        for i in range(len(mentors)):
            # Check if the current index exists in cosine_sim to prevent IndexError
            if i < len(cosine_sim) and j < len(cosine_sim[i]):
                if cosine_sim[i][j] > threshold:
                    best_matches.append({
                        "mentor": mentors[i]['username'], 
                        "score": cosine_sim[i][j]
                    })
        # Sort matches by highest score
        best_matches.sort(key=lambda x: x["score"], reverse=True)
        student_matches[student['username']] = best_matches
    
    return student_matches


@app.route('/match', methods=['GET'])
def match_mentors_students():
    """API endpoint to match mentors and students."""
    mentors = fetch_profiles(mentors_collection)
    students = fetch_profiles(students_collection)

    if not mentors or not students:
        return jsonify({"error": "Mentors or students data is missing"}), 404

    # Compute the similarity matrix
    similarity_matrix = compute_similarity(mentors, students)

    results_mentors = get_best_match_for_mentors(mentors, students, similarity_matrix)

    return jsonify(results_mentors), 200
@app.route('/match-students', methods=['GET'])
def match_students_to_mentors():
    mentors = fetch_profiles(mentors_collection)
    students = fetch_profiles(students_collection)

    if not mentors or not students:
        return jsonify({"error": "Mentors or students data is missing"}), 404

    similarity_matrix = compute_similarity(mentors, students)
    results = get_best_match_for_students(students, mentors, similarity_matrix)
    
    student_username = request.args.get('username', '').lower()
    mentor_index = int(request.args.get('index', 0))

    print("Requested student:", student_username)  # Debug print

    if student_username in results:
        mentor_matches = results[student_username]
        if mentor_matches:
            # Use modulo to wrap around the index if it exceeds the array length
            mentor_index = mentor_index % len(mentor_matches)
            return jsonify(mentor_matches[mentor_index]), 200
        else:
            random_mentor = random.choice(mentors)
            return jsonify({"mentor": random_mentor['username'], "score": "Random Assignment" }), 200
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

def fetch_courses(course_ids):
    """Fetches course details from MongoDB based on course IDs."""
    courses = db['courses'].find({'_id': {'$in': course_ids}})
    course_names = {str(course['_id']): course['courseName'] for course in courses}
    return course_names

def fetch_mentors(mentor_ids):
    """Fetches mentor details from MongoDB based on mentor IDs."""
    mentors = db['mentors'].find({'_id': {'$in': mentor_ids}})
    mentor_names = {str(mentor['_id']): mentor['name'] for mentor in mentors}
    return mentor_names

def resolve_user_profile(user_profile):
    """Resolve ObjectIds in user profile to human-readable names."""
    course_ids = user_profile.get('enrolledCourses', [])
    mentor_id = user_profile.get('mentor')
    
    course_names = fetch_courses(course_ids)
    mentor_names = fetch_mentors([mentor_id]) if mentor_id else {}
    
    # Update user profile with course names and mentor name
    resolved_profile = user_profile.copy()
    resolved_profile['enrolledCourses'] = [course_names.get(str(course_id), str(course_id)) for course_id in course_ids]
    resolved_profile['mentor'] = mentor_names.get(str(mentor_id), str(mentor_id)) if mentor_id else None
    
    return resolved_profile

def ask_question(user_profile):
    # Create a prompt incorporating the user profile data
    user_profile = resolve_user_profile(user_profile)
    user_info = f"User Profile: {user_profile}"
    prompt = f"You are an interviewer now. Introduce yourself as a Manager in Amile and your name is Hakunamatata. Start the interview based on the following user profile data: {user_info}"
    response = model.generate_content(prompt)
    return response.text

def generate_followup(interview_question, user_response, conversation_history, user_profile):
    # Include user profile data in the prompt if necessary
    user_profile = resolve_user_profile(user_profile)
    history_prompt = "\n".join([f"Q: {item['question']} A: {item['response']}" for item in conversation_history])
    user_info = f"User Profile: {user_profile}"
    prompt = f"Here is the conversation so far:\n{history_prompt}\nUser info: {user_info}\nYou asked the candidate: '{interview_question}'. The candidate responded: '{user_response}'. Now respond as an interviewer with a precise question."
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
    # Retrieve username from query parameters or session cookies
    username = request.args.get('username')  # or you can get it from cookies/session

    if not username:
        return jsonify({"error": "Username is required"}), 400

    # Fetch user profile from MongoDB
    user_profile = students_collection.find_one({'username': username}, {'_id': 0})
    
    if not user_profile:
        return jsonify({"error": "User not found"}), 404

    # Use user_profile for generating a question
    question = ask_question(user_profile)

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
    username = request.json.get('username')  # Include username

    if not duration or not sample_rate or not interview_question or not username:
        return jsonify({"error": "Missing required parameters"}), 400

    audio = record_audio(duration, sample_rate)
    user_response = transcribe_audio(audio, sample_rate)

    if user_response:
        # Get the conversation history from the session
        conversation_history = session.get('conversation_history', [])

        # Add the current question and response to the conversation history
        conversation_history.append({"question": interview_question, "response": user_response})
        session['conversation_history'] = conversation_history

        # Fetch user profile for generating follow-up
        user_profile = students_collection.find_one({'username': username}, {'_id': 0})

        if user_profile is None:
            return jsonify({"error": "User profile not found"}), 404

        # Generate the follow-up question based on conversation history and user profile
        followup = generate_followup(interview_question, user_response, conversation_history, user_profile)

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
