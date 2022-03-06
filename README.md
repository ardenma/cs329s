# cs329s

## Setup
1. use `python3 -m venv env` to create a virtual environment
2. activate the virtual environment with `source env/bin/activate` and then `pip install -r requirements.txt` to install all the python dependencies
    - you may need to `pip install -U pip` to upgrade the pip version, as well as `pip install wheel`
    - might need to `sudo apt install libomp5 libomp-dev` on linux or `brew install libomp` on mac for faiss 
3. install the project as a package (for imports to work) with `pip install -e .`
4. setup wandb account and get API key (https://wandb.ai/), login using the command`wandb login`

## Training Models
1. Run `python src/train.py` which should create two models `embedding_mode.pt` and `prediction_model.pt` in the `saved_models` directory
    - IMPORTANT: you will need these models for the future steps
    - use `config/config_default` to configure your run, these settings will be read by wandb
    - to run a sweep, simply configure `config/sweep.py`, go to the `src` directory, run `wandb sweep config/sweep.py` and then copy and paste the command that is outputted to launch the wandb agent (prepend the agent launch command with `nohup` if the process is dying after a while).
2. run `python evaluate.py` to evaluate the saved models
    - run `python evaluate.py --model_path PATH_TO_MODEL` to evaluate using the embedding model and the FAISS index + majority voter as the prediction model
    - run `python evaluate.py --artifact ARTIFACT_NAME` to run evaluation using a wandb artifact (e.g. `daily-tree-15-3-labels:v4`)

## Running the Application
1. use `ray start --head` to start the ray cluster
    - Note: if you receive a message about redis failing to start, try this https://github.com/ray-project/ray/issues/6146
2. use `python src/serve.py` to start the application (the application will run until `ray stop` is called)
3. use `ray stop` to kill the ray cluster

## Testing
1. use `ray start --head` to start the ray cluster
2. try some tests.
    - e.g. `python src/tests/test.py --test_name throughput -n 100` launches the application and tries to make 100 POST requests to the application and measures the throughput.
3. use `ray stop` to kill the ray cluster

## TODO
- write more tests
- cleanup train script
- improve models
- pass a config to app to setup the models (instead of hardcoding model paths in `app/model_deployment.py`)
- add more command line args for scripts
- integrate embedding model + index + majority voter into deployment
