# cs329s

## Setup
1. use `python3 -m venv env` to create a virtual environment
2. activate the virtual environment with `source env/bin/activate`
    - you may need to `pip install -U pip` to upgrade the pip version, as well as `pip install wheel`
3. install the project as a package (for imports to work) with `pip install -e .`
4. use `ray start --head` to start the ray cluster
5. use `python src/serve.py` to start the application
    - set `serve.start(detached=False)` to `serve.start(detached=True)` to have it run even after the script finishes
5. use `ray stop` to kill the ray cluster