from flask import (
    Flask,
    jsonify,
    render_template,
    request
)

from model import MisinformationClassifier

# Create classifier
classifier = MisinformationClassifier()

# Create web app
app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']

    prediction = classifier(text)[0]

    return(render_template('index.html', prediction=str(prediction)))

if __name__ == "__main__":
    app.run(port='8088',threaded=False)
