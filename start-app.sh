
ECHO "################  Welcome to ML2Viz ################"

ECHO "make sure python version is 3.0+ and virtual env is installed 

visit - https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/ for more info

"

lsof -ti:5000 | xargs kill
lsof -ti:6006 | xargs kill
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
python3 src/app.py &
tensorboard --logdir=logs
lsof -ti:5000 | xargs kill
deactivate
ECHO "Have a nice day!"

