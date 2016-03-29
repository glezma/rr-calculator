#import openpyxl as xl
import numpy as np

def xl_load(wb,range_name):
#	filename = 'In.CE.xlsm'
#	wb = xl.load_workbook(filename,data_only=True)
	#ws = wb.get_sheet_by_name('Datos')
	rng= wb.get_named_range(range_name).destinations[0]
	ws = rng[0]
	address = rng[1].replace('$','')
	r_f, r_t = address.split(':')
	# import pdb; pdb.set_trace()
	valores = ws[r_f:r_t]
	# print(type(valores))
	n_cols = len(valores[0])
	n_rows = len(valores)
	
	
	if n_rows==1 or n_cols==1:
		mlen = np.max([n_cols,n_rows]) 
		a=[]
		for ii in range(mlen):
			if n_rows==1:
				a.append(valores[0][ii].value)
			else:
				a.append(valores[ii][0].value)
		a=np.asanyarray(a)
	else:
		a = np.zeros((n_rows,n_cols))
		for ii in range(n_rows):
			for jj in range(n_cols):
				a[ii,jj] = valores[ii][jj].value
	# print(a)
	return a

 
#tt_mn=ws.range('tt_mn')

#A = np.zeros((37,3))
#for i in range(2,39):
#   for j in range(1,4):
#      A[i-2,j-1]= tt_mn.cell(row = i, column = j).value

# name = book.name_map['gap_mn'][0]
# name = book.sheet_by_name('Datos')
#import pdb; pdb.set_trace();
