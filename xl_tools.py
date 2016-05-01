import numpy as np

def xl_load(wb,range_name):

	rng= wb.get_named_range(range_name).destinations[0]
	ws = rng[0]
	address = rng[1].replace('$','')
	r_f, r_t = address.split(':')
	valores = ws[r_f:r_t]
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
	return a