# cs329s

## Setup
1. use `python3 -m venv env` to create a virtual environment
2. activate the virtual environment with `source env/bin/activate` and then `pip install -r requirements.txt` to install all the python dependencies
    - you may need to `pip install -U pip` to upgrade the pip version, as well as `pip install wheel`
    - might need to `sudo apt install libomp5 libomp-dev` for faiss
3. install the project as a package (for imports to work) with `pip install -e .`
4. setup wandb account and get API key (https://wandb.ai/)

## Training Models
1. run `python train.py` which should create two models `embedding_mode.pt` and `prediction_model.pt` in the `saved_models` directory
    - IMPORTANT: you will need these models for the future steps
    - run `python train.py --contrastive` to run with contrastive loss
2. run `python evaluate.py` to evaluate the saved models

## Running the Application
1. use `ray start --head` to start the ray cluster
2. use `python src/serve.py` to start the application (the application will run until `ray stop` is called)
    - Note: if you receive a message about redis failing to start, try this https://github.com/ray-project/ray/issues/6146
3. use `ray stop` to kill the ray cluster

## Testing
1. use `ray start --head` to start the ray cluster
2. try some tests.
    - e.g. `python tests/simple_http_request.py` launches the application and tries to make 5 POST requests to the application and prints the response.
3. use `ray stop` to kill the ray cluster

## TODO
- write more tests
- cleanup train script
- improve models
- pass a config to app to setup the models (instead of hardcoding model paths in `app/model_deployment.py`)