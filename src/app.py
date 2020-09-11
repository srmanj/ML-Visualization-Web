from flask import Flask, request, render_template
from model import CVtrain
import argparse, os, zipfile
from werkzeug.utils import secure_filename


app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--env", default='dev', type=str, help="This is the environment variable")


@app.route('/', methods = ['POST','GET'])
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

	dataset = request.form.get("dataset")
	if dataset == 'other':
		file = request.files['file']
		zipped_file = os.path.join('./data/', secure_filename(file.filename))
		file.save(zipped_file)
		zip_ref = zipfile.ZipFile(zipped_file)
		zip_ref.extractall('./data/')
		zip_ref.close()
		dataset = zipped_file.split('.zip')[0]

	task  = request.form.get("task")
	try:
		architecture = request.form.get("architecture")
	except:
		architecture = None

	try:
		file = request.files['model_path']
		model_path = os.path.join('./data/',secure_filename(file.filename))
		file.save(model_path)

		print(model_path)
	except:
		model_path = None
	try:
		epochs = int(request.form.get("epochs"))
	except:
		epochs = None
	batch_size = int(request.form.get("bs"))
	try:
		optimizer = request.form.get("optimizer")
	except:
		optimizer = None
	try:
		lr = float(request.form.get("lr"))
	except:
		lr = None
	viz = request.form.getlist('visualization')
	num = int(request.form.get("num"))
	# print(dataset,task, architecture, epochs, optimizer, lr, viz)

	json_string = CVtrain(dataset, task, architecture, epochs, batch_size, optimizer, lr, viz, num, model_path)
	# accuracy = 20

	# return render_template('index.html', prediction_text='Accuracy {}'.format(accuracy))
	return render_template('index.html', output=json_string)



if __name__ == '__main__':
	args = parser.parse_args()
	if args.env == 'prod':
		app.run(host = '0.0.0.0', port  = 5000)
	else:
		app.run(port = 5000, debug=True)