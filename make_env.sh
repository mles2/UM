PYTHON_PATH=`which python3`
virtualenv -p $PYTHON_PATH env
env/bin/pip install -r requirements.txt
