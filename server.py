import os
from flask import (render_template, Flask, request, make_response, 
					redirect, url_for, session, jsonify )
import plotly
from flask.ext.bootstrap import Bootstrap

import numpy as np
from flask.ext.wtf import Form
from wtforms import FileField, SubmitField, ValidationError
import json
from celery import Celery
import alm as alm

app = Flask(__name__)
app.config['SECRET_KEY'] = 'top secret!'
bootstrap = Bootstrap(app)

# Celery configuration
direccion_redis = 'redis://localhost:6379/0'
direccion_redis = 'redis://h:p5t6n5r06k8kk01ls0655a1qvdi@ec2-107-22-196-235.compute-1.amazonaws.com:13319'

app.config['CELERY_BROKER_URL'] = direccion_redis
app.config['CELERY_RESULT_BACKEND'] = direccion_redis

# Initialize Celery

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

app.app_context().push()

class UploadForm(Form):
    image_file = FileField('Archivo de datos')
    submit = SubmitField('Cargar')

    def validate_image_file(self, field):
        if ((field.data.filename[-5:].lower() != '.xlsm') 
        	and (field.data.filename[-5:].lower() != '.xlsx')):
            raise ValidationError('Invalid file extension')

@celery.task(bind=True)
def long_task(self,*args):
	for elem in args:
		fullfilename = elem
	# filecookie = request.cookies.get('filename')
	# fullfilename = json.loads(filecookie)['file']
	# fullfilename = session.get('filename',None)
	# fullfilename = 'F://OneDrive//cloud_projects//cir_economic_capital//static//uploads//InCE.xlsm'

	print(fullfilename) 
	fc = alm.FileComputation(fullfilename, self)
	results = fc.compute()
	nlen = results.N()
	table_ce = results.table_ce()
	scen_list = results.scen_list()
	table_list = results.table_list()
	plot_list_mn = results.plot_list_mn()
	plot_list_me = results.plot_list_me()
	string = render_template('Out_Tasas.html', nlen = nlen, table_ce = table_ce, scen_list = scen_list, table_list = table_list, plot_list_mn = plot_list_mn, plot_list_me = plot_list_me)
	# with open("Output.html", "w") as text_file:
		# print(" {}".format(string), file=text_file)
	return string 
	# {'current': 100, 'total': 100, 'status': 'Task completed!',
            # 'result': 1}

@app.route('/', methods=['GET', 'POST'])
def index():
	file_name = None
	form = UploadForm()
	fullfilename = None

	if  'report_n' in request.form and request.form['report_n']=='report_v':
		 return redirect(url_for('reporte'))

	if form.validate_on_submit():
		# print('validacion de formulario')
		file_name = form.image_file.data.filename
		this_file = 'uploads//' + form.image_file.data.filename
		fullfilename = os.path.join(app.static_folder, this_file)

		form.image_file.data.save(fullfilename)
		response = make_response(render_template('index.html', form=form, this_file=file_name,fullfilename=fullfilename))
		response.set_cookie('filename',json.dumps({'file': fullfilename}))
		# session['filename'] = fullfilename
		return response
	else:
		return render_template('index.html', form=form, this_file=file_name,fullfilename=fullfilename)

@app.route('/reporte')
def reporte():
	 # string = session.get('reporte',None)
	# import os as os

	# string = os.startfile('Output.html')
	with open('Output.html','r') as myfile:
		string = myfile.read().replace("\n",'')
	# import pdb; pdb.set_trace()
	return string

@app.route('/longtask',methods=['POST'])
def process():
	# import pdb; pdb.set_trace();
	# print('iniciando apply_async')
	filecookie = request.cookies.get('filename')
	fullfilename = json.loads(filecookie)['file']
	print(fullfilename)
	task = long_task.apply_async((fullfilename,))
	return jsonify({}), 202, {'Location': url_for('taskstatus',
                                                  task_id=task.id)}

@app.route('/status/<task_id>')
def taskstatus(task_id):
    # print('iniciando asyncresult')
    task = long_task.AsyncResult(task_id)
    print(task.state)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)