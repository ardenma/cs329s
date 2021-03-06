# CS329s Misinformation Detection App

## Summary of Approach:
1. Train a transformer model to generate embededings from query strings using data from the LIAR dataset, a collection of politifact statements labeled by their truthfulness.
2. Use this model to generate an embedding space of the LIAR dataset examples
3. Then, given a query we can generate and embedding, match it to the K-closest embeddings in our embedding space, and the utilize a voting based approach to generate inferences (e.g. if 2/3 votes are 'true' we will return the label true!)

## Model Information:
- For our embedding model we finetuned a DistilBERT (https://arxiv.org/pdf/1910.01108.pdf) model on the LIAR dataset
- For our prediction model, we used a KNN model with K=3 (breaking ties randomly) on the embedding space generated by our embedding model over the training examples in the LIAR dataset
## Dataset Information:
- We used the LIAR dataset (https://huggingface.co/datasets/liar) for training and evaluating our model
- Because the LIAR dataset is quite challenging (highest accuracy reported in the [original paper](https://arxiv.org/pdf/1705.00648.pdf) was 27%) we used a **modified** version of the dataset where we combined the labels as follows: (**pants-fire**, **false**) -> **false**, (**barely-true**, **half-true**) -> **unsure**, (**mostly-true**, **true**) -> **true**
- All reported metrics are based off this label-joining scheme

# Instructions to Use This Repo
## Setup
1. use `python3 -m venv env` to create a virtual environment
2. activate the virtual environment with `source env/bin/activate` and then `pip install -r requirements.txt` to install all the python dependencies
    - only tested for python 3.6
    - you may need to `pip install -U pip` to upgrade the pip version, as well as `pip install wheel`
    - might need to `sudo apt install libomp5 libomp-dev` on linux or `brew install libomp` on mac for faiss 
3. install the project as a package (for imports to work) with `pip install -e .`
4. setup wandb account and get API key (https://wandb.ai/), login using the command`wandb login`

## Training Models
1. Run `python backend/train.py` which should create two models `embedding_mode.pt` and `prediction_model.pt` in the `saved_models` directory
    - IMPORTANT: you will need these models for the future steps
    - use `config/config_default` to configure your run, these settings will be read by wandb
    - to run a sweep, simply configure `config/sweep.py`, go to the `backend` directory, run `wandb sweep config/sweep.py` and then copy and paste the command that is outputted to launch the wandb agent (prepend the agent launch command with `nohup` if the process is dying after a while).
2. run `python evaluate.py` to evaluate the saved models
    - run `python evaluate.py --model_path PATH_TO_MODEL` to evaluate using the embedding model and the FAISS index + majority voter as the prediction model
    - run `python evaluate.py --artifact ARTIFACT_NAME` to run evaluation using a wandb artifact (e.g. `daily-tree-15-3-labels:v4`)

## Running the Backend Application
1. use `ray start --head` to start the ray cluster
    - Note: if you receive a message about redis failing to start, try this https://github.com/ray-project/ray/issues/6146
2. use `python backend/serve.py` to start the application (the application will run until `ray stop` is called)
    - Use the option `--detached` to have the serve instance run in the background.
3. use `ray stop` to kill the ray cluster

## Running the Frontend Application
1. Use the command `streamlit run frontend/app.py`.

## Testing
1. use `ray start --head` to start the ray cluster
2. try some tests.
    - e.g. `python backend/tests/test.py --test_name throughput -n 100` launches the application and tries to make 100 POST requests to the application and measures the throughput.
3. use `ray stop` to kill the ray cluster

