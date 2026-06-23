# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, session, Response
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = "emotion_ai"

camera = cv2.VideoCapture(0)
camera_running = False

emotion_counts = {
    "Happy": 0,
    "Sad": 0,
    "Angry": 0,
    "Neutral": 0,
    "Surprise": 0
}

current_emotion = "Neutral"
current_emoji = "😐"
current_suggestion = "Stay balanced"

# FILES
open("users.txt","a",encoding="utf-8").close()
open("history.txt","a",encoding="utf-8").close()

# ---------------- USERS ----------------
def check_user(u,p):
    with open("users.txt","r",encoding="utf-8") as f:
        for i in f:
            x=i.strip().split(",")
            if len(x)==2 and x[0]==u and x[1]==p:
                return True
    return False

def save_user(u,p):
    with open("users.txt","a",encoding="utf-8") as f:
        f.write(f"{u},{p}\n")

# ---------------- EMOTION ----------------
def detect_emotion(w,h):
    s=w+h
    if s>450:return "Happy"
    elif s>400:return "Neutral"
    elif s>350:return "Sad"
    elif s>300:return "Angry"
    else:return "Surprise"

def ui(e):
    data={
        "Happy":("😊","Keep smiling"),
        "Sad":("😢","Stay strong"),
        "Angry":("😡","Relax"),
        "Neutral":("😐","Balanced"),
        "Surprise":("😲","Observe")
    }
    return data.get(e,("😐","Stay"))

# ---------------- LOGIN ----------------
@app.route('/')
def login():
    return render_template("login.html")

@app.route('/login',methods=['POST'])
def login_user():
    u=request.form['username']
    p=request.form['password']

    if check_user(u,p):
        session['user']=u
        return redirect('/dashboard')

    return render_template("login.html",error="Wrong login")

# ---------------- REGISTER ----------------
@app.route('/register',methods=['GET','POST'])
def register():
    if request.method=='POST':
        save_user(request.form['username'],request.form['password'])
        return redirect('/')
    return render_template("register.html")

# ---------------- DASHBOARD ----------------
@app.route('/dashboard')
def dashboard():
    if not session.get('user'):
        return redirect('/')

    return render_template("index.html",
        user=session['user'],
        emotion=current_emotion,
        emoji=current_emoji,
        suggestion=current_suggestion,
        camera_running=camera_running
    )

# ---------------- CAMERA ----------------
def gen():
    global current_emotion,current_emoji,current_suggestion

    face=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

    while True:
        if not camera_running:
            break

        ret,frame=camera.read()
        if not ret:
            break

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            emo=detect_emotion(w,h)
            current_emotion=emo
            current_emoji,current_suggestion=ui(emo)

            emotion_counts[emo]+=1

            with open("history.txt","a",encoding="utf-8") as f:
                f.write(f"{datetime.now()} | {emo}\n")

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,emo,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

        _,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()

        yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

@app.route('/video')
def video():
    if not camera_running:
        return Response("", mimetype='text/plain')

    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- START / STOP ----------------
@app.route('/start_camera')
def start():
    global camera_running
    camera_running=True
    return redirect('/dashboard')

@app.route('/stop_camera')
def stop():
    global camera_running
    camera_running=False
    return redirect('/dashboard')

# ---------------- GRAPH ----------------
@app.route('/graph')
def graph():
    os.makedirs("static",exist_ok=True)

    plt.bar(emotion_counts.keys(),emotion_counts.values())
    plt.title("Emotion Graph")
    plt.savefig("static/graph.png")
    plt.close()

    return render_template("index.html",
        user=session.get('user'),
        emotion=current_emotion,
        emoji=current_emoji,
        suggestion=current_suggestion,
        camera_running=camera_running,
        graph=True
    )

# ---------------- HISTORY ----------------
@app.route('/history')
def history():
    if not session.get('user'):
        return redirect('/')

    with open("history.txt","r",encoding="utf-8") as f:
        logs=f.readlines()

    return render_template("history.html",logs=logs)

# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__=="__main__":
    app.run(debug=True)