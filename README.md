# cs329s

## Setup
1. use `python3 -m venv env` to create a virtual environment
2. activate the virtual environment with `source env/bin/activate`
    - you may need to `pip install -U pip` to upgrade the pip version, as well as `pip install wheel`
3. install the project as a package (for imports to work) with `pip install -e .`

## Running the Application
1. use `ray start --head` to start the ray cluster
2. use `python src/serve.py` to start the application (the application will run until `ray stop` is called)
3. use `ray stop` to kill the ray cluster

## Testing
1. use `ray start --head` to start the ray cluster
2. try some tests.
    - e.g. `python tests/simple_http_request.py` launches the application and tries to make 5 POST requests to the application and prints the response.
3. use `ray stop` to kill the ray cluster