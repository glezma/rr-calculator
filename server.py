import os
from flask import (render_template, Flask, request, make_response, redirect, url_for, session )
import plotly
from flask.ext.bootstrap import Bootstrap
# plotly.offline.init_notebook_mode() 
import alm as alm
# import importlib as imp
# imp.reload(alm)
# import xlwings as xw
import openpyxl as opxl
import xl_tools as xl
import numpy as np
from flask.ext.wtf import Form
from wtforms import FileField, SubmitField, ValidationError
import json

#+ filename = "F:\OneDrive\python_projects\cir_economic_capital\In.CE.xlsm"
def compute_alm(filename):
	# import pythoncom
	# pythoncom.CoInitialize()
	wb = opxl.load_workbook(filename,data_only=True,use_iterators=False)
	xl_gap_MN = xl.xl_load(wb, 'gap_mn')
	# import pdb; mpdb.set_trace()
	xl_tt_MN = xl.xl_load(wb, 'tt_mn')
	xl_gap_ME = xl.xl_load(wb, 'gap_me')
	xl_tt_ME = xl.xl_load(wb, 'tt_me')
	xl_scendates = xl.xl_load(wb, 'scen')
	buckets_num = xl.xl_load(wb, 'buckets_num')
	buckets_label = xl.xl_load(wb, 'buckets_label')
	TC = xl.xl_load(wb, 'tc')
	PE = xl.xl_load(wb, 'pe')

	kappa1 = 0.0554327
	sigma21 = 0.0709036**2
	lambd_a1 = -0.0801159
	theta1 = 0.059518

	p1 = alm.CIR(theta1,kappa1,lambd_a1,sigma21)

	kappa = 0.435049
	sigma2 = 0.122388**2
	lambd_a = -0.35292
	theta = 0.0222862

	p2 = alm.CIR(theta,kappa,lambd_a,sigma2)
	model_list = [p1,p2]
	correl = 0.0244875
	n_sim = 100000

	gs = alm.GapStack( xl_scendates, xl_gap_MN, xl_gap_ME,
					xl_tt_MN, xl_tt_ME, buckets_num, buckets_label,TC,PE)
	eng = alm.EcapEngine(gs,model_list,correl)
	results = eng.compute(n_sim)

	nlen = results.N()
	table_ce = results.table_ce()
	scen_list = results.scen_list()
	table_list = results.table_list()
	plot_list_mn = results.plot_list_mn()
	plot_list_me = results.plot_list_me()
	return nlen, table_ce, scen_list,table_list, plot_list_mn, plot_list_me

app = Flask(__name__)
app.config['SECRET_KEY'] = 'top secret!'
bootstrap = Bootstrap(app)

class UploadForm(Form):
    image_file = FileField('Archivo de datos')
    submit = SubmitField('Cargar')

    def validate_image_file(self, field):
        if ((field.data.filename[-5:].lower() != '.xlsm') 
        	and (field.data.filename[-5:].lower() != '.xlsx')):
            raise ValidationError('Invalid file extension')

@app.route('/', methods=['GET', 'POST'])
def index():
	file_name = None
	form = UploadForm()
	fullfilename = None

	if  'compute_n' in request.form and request.form['compute_n']=='compute_v':
		 return redirect(url_for('process'))

	if form.validate_on_submit():
		file_name = form.image_file.data.filename
		this_file = 'uploads//' + form.image_file.data.filename
		fullfilename = os.path.join(app.static_folder, this_file)
		form.image_file.data.save(fullfilename)
		response = make_response(render_template('index.html', form=form, this_file=file_name,fullfilename=fullfilename))
		response.set_cookie('filename',json.dumps({'file': fullfilename}))
		return response
	else:
		return render_template('index.html', form=form, this_file=file_name,fullfilename=fullfilename)


@app.route('/process',methods=['GET'])
def process():
	filecookie = request.cookies.get('filename')
	# import pdb;pdb.set_trace()
	fullfilename = json.loads(filecookie)['file'] 
	nlen, table_ce, scen_list,table_list, plot_list_mn, plot_list_me = compute_alm(fullfilename)
	string = render_template('Out_Tasas.html', nlen = nlen, table_ce = table_ce, scen_list = scen_list, table_list = table_list, plot_list_mn = plot_list_mn, plot_list_me = plot_list_me)
	return string

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run()