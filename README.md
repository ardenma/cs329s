# cs329s

## Setup
1. First, let's set up a virtual environment. Run the command `python3 -m venv env` to create a virtual environment named `env`.
2. Next, let's activate our virtual environment. You will need to re-activate your virtual environment every time you run this code. To activate your virtual environment, run the command `source env/bin/activate`.
3. Our project requires some dependencies in order to run. To install the dependencies necessary for this project, first run the command `pip install wheel`. Then run the command `pip install -r requirements.txt`.
    - If your `pip` is not up to date, you may need to run the command `pip install -U pip` to upgrade your `pip` version.
4. Next, we need to train our model. To train our model, run the command `python src/train.py`. This will create our misinformation classifier, train it on the LIAR dataset, and save the model's learned weights to the `saved_models` directory.

## Running the Application
1. To run the application, run the command `python src/main.py`. This will typically take a few seconds to create the application. Once the application has been created, you will see several logs, beginning with `Serving Flask app "main" (lazy loading)`. Once you see these logs, your app is now up and running.
2. To try out the application, visit http://127.0.0.1:8088/.
