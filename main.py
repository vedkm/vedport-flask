from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/software')
def get_software_portfolio():
  return render_template('software.html')

@app.route('/software/<project>')
def get_project(project):
  return render_template('software/'+project+'.html', project=project)

@app.route('/audio')
def get_audio_portfolio():
  return render_template('audio.html')

@app.route('/contact')
def get_contact():
  return render_template('contact.html')


app.run(host='0.0.0.0', port=81)
