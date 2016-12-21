import os
from flask import (render_template, Flask, request, make_response, 
					redirect, url_for, session, jsonify,Markup )
#import plotly
from flask.ext.bootstrap import Bootstrap

import numpy as np
from flask.ext.wtf import Form
from wtforms import FileField, SubmitField, ValidationError
import json
from celery import Celery
import alm as alm
# import openpyxl as opxl
# import xl_tools as xl
html_code ='nada'


nlen = None
table_ce = None
scen_list = None
table_list = None
plot_list_mn = None
plot_list_me = None


app = Flask(__name__)
app.config['SECRET_KEY'] = 'top secret!'
bootstrap = Bootstrap(app)

# Celery configuration
direccion_redis = 'redis://localhost:6379/0'
direccion_redis = 'redis://h:p2f4e2dc26bd9bd22de9ac82b66cb3f72c77d5b987331948e2db4d815573b0d88@ec2-184-72-246-90.compute-1.amazonaws.com:18049'

app.config['CELERY_BROKER_URL'] = direccion_redis
app.config['CELERY_RESULT_BACKEND'] = direccion_redis

# Initialize Celery

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'],backend=app.config['CELERY_RESULT_BACKEND'])
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
	global nlen
	global table_ce
	global scen_list
	global table_list
	global plot_list_mn
	global plot_list_me

	for elem in args:
		fc = elem
	# filecookie = request.cookies.get('filename')
	# fullfilename = json.loads(filecookie)['file']
	# fullfilename = session.get('filename',None)
	# fullfilename = 'F://OneDrive//cloud_projects//cir_economic_capital//static//uploads//InCE.xlsm'

	# print(fullfilename) 
	# import ipdb; ipdb.set_trace()
	results = fc.compute(self)
	nlen = results.N()
	table_ce = results.table_ce()
	scen_list = results.scen_list()
	table_list = results.table_list()
	plot_list_mn = results.plot_list_mn()
	plot_list_me = results.plot_list_me()
	string = render_template('Out_Tasas.html', nlen = nlen, table_ce = table_ce, scen_list = scen_list, table_list = table_list, plot_list_mn = plot_list_mn, plot_list_me = plot_list_me)
	# import ipdb;ipdb.set_trace()
	# with open("Output.html", "w") as text_file:
		# print(" {}".format(string), file=text_file)
	return  {'current': 100, 'total': 100, 'status': 'Task completed!',
             'result': string}

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

@app.route('/reporte',methods=['POST','GET'])
def reporte():
	global html_code
	print('empezamos la funcion {}'.format(request.method))
	 # string = session.get('reporte',None)
	# import os as os
	if request.method=='POST':
		print('POST')
		# import ipdb; ipdb.set_trace();
		string = request.json['html_coding']
		html_code = string
		return string
	else:
		# import ipdb; ipdb.set_trace();
		# string = render_template('Out_Tasas.html', nlen = nlen, table_ce = table_ce, scen_list = scen_list, table_list = table_list, plot_list_mn = plot_list_mn, plot_list_me = plot_list_me)

		return render_template('foo.html', html_code=html_code)

	# string = os.startfile('Output.html')
	# with open('Output.html','r') as myfile:
	# 	string = myfile.read().replace("\n",'')
	
	
	#task = long_task.AsyncResult(task_id


@app.route('/longtask',methods=['POST'])
def process():
	
	# print('iniciando apply_async')
	filecookie = request.cookies.get('filename')
	fullfilename = json.loads(filecookie)['file']
	fc = alm.FileComputation(fullfilename)
	# import ipdb; ipdb.set_trace()
	print('longtask')
	print(fullfilename)
	task = long_task.apply_async((fc,))
	# import pdb; pdb.set_trace()
	print('task is {}: '.format(type(task)))
	print('task result is from logn task : {}'.format(type(task.result)) )
	if task.ready()==True:
		#import pdb; pdb.set_trace()
		print('TASK READY FROM LONGTASK POST')
		session['result'] = task.result['result']
	print('task.id= {}'.format(task.id))
	return jsonify({}), 202, {'Location': url_for('taskstatus',
                                                  task_id=task.id)}

@app.route('/status/<task_id>')
def taskstatus(task_id):
    # print('iniciando asyncresult')
    task = long_task.AsyncResult(task_id)
    # print(dir(task))
    
    print('task is (from status :{}'.format(task.state))
    if task.ready() == True:
    	print(' task ready :{}'.format('uff'))
    	session['task'] = task_id
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        #print(dir(task.info))
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
    if task.ready() == True:
        print(' task ready :{}'.format('uff'))
        #import pdb; pdb.set_trace()
    return jsonify(response)

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)
