from flask import Flask, render_template, request, redirect, url_for, session
from flask_pymongo import PyMongo
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import os
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import re
from datetime import datetime

os.environ['OPENAI_API_KEY']='YOUR PERSONAL OPEN API KEY'
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

app = Flask(__name__, static_url_path='/static')
app.config["SECRET_KEY"] = "1241f366ecf2af7cbf180a0bab94fbdea617358a"
app.config["MONGO_URI"] = "mongodb+srv://rahulstark2:Password@cluster0.eepoml0.mongodb.net/Job_Sentiment_Analysis?retryWrites=true&w=majority"
mongodb_client = PyMongo(app)
db = mongodb_client.db

@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        text = request.form['emaill']
        password = request.form['passwordl']
        existing_email = db.users.find_one({'email': text})
        existing_user = db.users.find_one({'username': text})
        print(existing_email)
        print(existing_user)
        userexists2=False
        passwordnew="cfsdcfrsdcredcsadcsdscdafc"
        if existing_email is not None:
            passwordnew = existing_email.get('password')
            username = existing_email.get('username')
        if existing_user is not None:
            passwordnew = existing_user.get('password')
            username=existing_user.get('username')
        if password==passwordnew:
            userexists2=True
            session['username'] = username
            return redirect(url_for('welcomeloggedin',userexists2=userexists2,username=username))


        
        return redirect(url_for('welcome', userexists2=userexists2))
    
@app.route('/logout')
def logout():
    # Remove the username from the session if it exists
    session.pop('username', None)

    # Redirect the user to the login page or any other desired page
    return redirect(url_for('welcome'))


@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        username = request.form['usernamer']
        email = request.form['emailr']
        password = request.form['passwordr']

        # Check if the user already exists in the database (based on email)
        existing_email = db.users.find_one({'email': email})
        existing_user = db.users.find_one({'username': username})
        userexists=False
        if existing_user or existing_email:
            userexists=True

        # Create a new user document
        else:
            new_user = {
                'username': username,
                'email': email,
                'password': password  # Note: In production, hash the password for security
            }

            # Insert the new user into the MongoDB database
            db.users.insert_one(new_user)

        return redirect(url_for('welcome', userexists=userexists))  # Redirect to the welcome page after registration



@app.route('/')
def welcome():
    userexists = request.args.get('userexists')
    userexists2 = request.args.get('userexists2')
    return render_template('welcome.html',userexists=userexists,userexists2=userexists2)

@app.route('/loggedin')
def welcomeloggedin():
    userexists2 = request.args.get('userexists2')
    username = request.args.get('username')
    if username is None:
        username=session['username']
    return render_template('welcome2.html',userexists2=userexists2,username=username)
    

@app.route('/analyze')
def job_description():
    if 'username' in session and session['username'] is not None:
        flag=1
    else:
        flag=0
    print(flag)
    return render_template('job_description_analyze.html',flag=flag)

@app.route('/result', methods=['POST'])
def analyze():
    job_description = request.form['job_description']
    job_description2=job_description
    max_length = 2600  # Maximum sequence length supported by the model
    if len(job_description) > max_length:
        # Create a parser and tokenizer
        parser = PlaintextParser.from_string(job_description, Tokenizer("english"))

        # Use LSA (Latent Semantic Analysis) for summarization
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, 2)  # Number of sentences in the summary

        job_description = ' '.join(str(sentence) for sentence in summary)
        

    #AutoTokenizer
    
    print(len(job_description))
    encoded_text = tokenizer(
        job_description,
        padding='max_length',  # Pad to the model's max sequence length
        truncation=True,       # Truncate if the text exceeds the max length
        return_tensors='pt',
        max_length=512  # Replace with the actual max length
    )
    output = model(**encoded_text)
    scores = output.logits[0].detach().numpy()  # Access logits directly
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    print(scores_dict['roberta_pos'])

    if(scores_dict['roberta_neu']>scores_dict['roberta_pos']):
        autoscore=scores_dict['roberta_neu'];
    else:
        autoscore=scores_dict['roberta_pos'];


    #pipeline
   
    sent_pipeline = pipeline("sentiment-analysis")
    if len(job_description2) > 2600:
            result = sent_pipeline(job_description)
    else:
            result = sent_pipeline(job_description2)
    

    print(result[0]['score'])

    result[0]['score']=(autoscore+result[0]['score'])/2
    result[0]['score']="{:.0%}".format(result[0]['score'])

    print(result[0]['score'])


    convo = ConversationChain(llm=OpenAI(temperature=0.7))
    name=convo.run(job_description2+"Just tell me why this job description is "+result[0]['label']+". Keep the answer short and precise and Start the answer like this - The job description you provided. ")
    print(name)
    job_details=convo.run("From the job description just tell me what is the job title,key responsibilities,top 4 qualifications,skills required,job location and salary ?Just give me the answer in keywords?I want the answer exactly in this format - Job Title: , Key Responsibilities: , Qualifications: , Skills Required: , Job location:  , Salary:  ")
    text=job_details

    print(text)
    
    keywords = [
    'Job Title',
    'Key Responsibilities',
    'Qualifications',
    'Skills Required',
    'Job Location',
    'Salary'
    ]

    # Create a regular expression pattern to match the keywords
    pattern = "|".join(re.escape(keyword) for keyword in keywords)
    pattern = re.compile(pattern, re.IGNORECASE)

    # Find the starting positions of the matched keywords
    matches = [match.start() for match in re.finditer(pattern, text)]
    job_title=text[matches[0]+10:matches[1]-2]
    key_responsibilities=text[matches[1]+21:matches[2]-2]
    qualifications=text[matches[2]+15:matches[3]-2]
    skills_required=text[matches[3]+16:matches[4]-2]
    job_location=text[matches[4]+13:matches[5]-2]
    salary=text[matches[5]+7:-1]

    responsibilities_list = key_responsibilities.split(', ')
    if len(responsibilities_list) > 35:
        key_responsibilities = ', '.join(responsibilities_list[:-2])

    username = session.get('username')

    job_info = {
        'username': username,
        'job_description': job_description,
        'sentiment_label': result[0]['label'],
        'sentiment_score': result[0]['score'],  # Sentiment score
        'job_title': job_title,
        'key_responsibilities': key_responsibilities,
        'qualifications': qualifications,
        'skills_required': skills_required,
        'job_location': job_location,
        'salary': salary,
        'timestamp': datetime.now()  # Timestamp for when this information is stored
    }

    db.job_info.insert_one(job_info)

    if 'username' in session and session['username'] is not None:
        flag=1
    else:
        flag=0 

    return render_template('result.html', result=result,name=name,job_title=job_title,key_responsibilities=key_responsibilities,qualifications=qualifications,skills_required=skills_required,job_location=job_location,salary=salary,flag=flag)

if __name__ == '__main__':
    app.run(debug=True,port=5001)
