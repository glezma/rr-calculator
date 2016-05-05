import numpy as np
import pandas as pd
from flask import Markup
class CIR:
	def __init__(self,theta,kappa,lambd_a,sigma2):
		self.theta = theta
		self.kappa = kappa
		self.lambd_a = lambd_a
		self.sigma2 = sigma2
		self.gamma = ( ( kappa + lambd_a)**2 + 2*sigma2 )**0.5

	def get_B(self,t,T):
		gamma = self.gamma
		kappa = self.kappa
		lambd_a = self.lambd_a
		exp_g = np.exp(gamma*(T-t))
		B = 2*( exp_g - 1) / ( ( gamma+kappa +lambd_a )*( exp_g -1) + 2*gamma  )
		return B


	def get_A(self,t,T):
		gamma =self.gamma
		kappa = self.kappa
		lambd_a = self.lambd_a
		theta =self.theta
		sigma2 = self.sigma2
		A = np.abs(( (2*gamma*np.exp( (gamma+kappa+lambd_a)*(T-t)/2 ) ) / \
		    ((gamma + kappa+lambd_a)*( np.exp(gamma*(T-t)) - 1 ) + 2*gamma) ))**(2*kappa*theta/sigma2)
		return A

	def sim_short_rate(self,r0,dt,mat_eps):
		N, n_sim = mat_eps.shape
		R = np.zeros([N+1,n_sim])
		R[0,:]=r0
		for ii in range(1,N+1):
			R[ii,:] = R[ii-1,:] +(self.theta-R[ii-1,:])*(1-np.exp(-self.kappa*dt))+\
			self.sigma2**0.5*np.sqrt(R[ii-1,:])*np.sqrt(dt)*mat_eps[ii-1,:]
			R[ii,R[ii,:]<0]=0
		return R

	def yield_at_bucket_T(self,t,T,r): 
		# r is a column vector
		# t and T are scalars 
		A = self.get_A(t,T) # scalar
		B = self.get_B(t,T) # scalar

		yab = (1/ ( A*np.exp(-r*B) ) )**(1/(T-t)) - 1;	

		return yab

	def get_yield_curves(self,r_shorts, yield_buckets):
		n_sim = len(r_shorts)
		n_buckets = len(yield_buckets)
		t=0
		yield_curves = np.zeros([n_sim, n_buckets])
		for ii in range(0,n_buckets):
			T=yield_buckets[ii]
			yield_curves[:,ii] = self.yield_at_bucket_T(t,T,r_shorts)
		return yield_curves

class Gap:
	def __init__(self,gap_MN,gap_ME,tt_MN,tt_ME, buckets_num,buckets_lab,scen_dates,TC,PE):
		self.gap_MN = gap_MN
		self.gap_ME = gap_ME
		self.buckets_num = buckets_num
		self.buckets_lab = buckets_lab
		self.tt_MN = tt_MN
		self.tt_ME = tt_ME
		self.scen_dates  = scen_dates
		self.TC = TC
		self.PE = PE
	def __add__(self,other):
		# gap1
		gap_MN = self.gap_MN
		gap_ME = self.gap_ME
		tt_MN = self.tt_MN
		tt_ME = self.tt_ME
		buckets_num = self.buckets_num
		buckets_lab = self.buckets_lab
		scen_dates = self.scen_dates
		TC = self.TC
		PE = self.PE
        # gap2
		gap_MN1 = other.gap_MN
		gap_ME1 = other.gap_ME
		tt_MN1 = other.tt_MN
		tt_ME1 = other.tt_ME
		buckets_num1 = other.buckets_num
		buckets_lab1 = other.buckets_lab
		scen_dates1 = other.scen_dates
		TC1 = other.TC
		PE1 = other.PE
        # Create sum object
		gap_MNr = other.gap_MN
		gap_MEr = other.gap_ME
		tt_MNr = tt_MN
		tt_MEr = tt_ME
		gap_MNr = gap_MN + gap_MN1
		gap_MEr = gap_ME + gap_ME1
		sumation = self.__init__(gap_MNr, gap_MEr, tt_MNr, tt_MEr,
			buckets_num, buckets_lab,scen_dates,TC,PE)
		return sumation
	# 	a = print(self.tt_MN)
	# 	return a

class GapStack:
	def __init__(self, xl_scendates, xl_gap_MN,xl_gap_ME,
				xl_tt_MN, xl_tt_ME, buckets_num, buckets_label,xl_TC,xl_PE)	:
		self.xl_scendates = xl_scendates
		self.buckets_num = buckets_num
		self.buckets_label = buckets_label
		N = len(xl_scendates)
		self.N = N
		l = []
		for ii in range(0,N):
			scen_dates = xl_scendates[ii:ii+1][0]
			TC = xl_TC[ii:ii+1][0]
			PE = xl_PE[ii:ii+1][0]
			gap_MN = xl_gap_MN[ii,:]
			gap_ME = xl_gap_ME[ii,:]
			tt_MN = xl_tt_MN[ii:ii+1,:]
			tt_ME = xl_tt_ME[ii:ii+1,:]
			l.append(Gap(gap_MN, gap_ME, tt_MN, tt_ME,
			 buckets_num, buckets_label,scen_dates,TC,PE))
		self.l = l
	def __getitem__(self,index):
		return self.l[index]

class EcapResult:
	def __init__(self, base_pv, shock_pv, base_pv_mn, shock_pv_mn,
				base_pv_me, shock_pv_me, base_tt_mn, base_tt_me,shock_tt_mn, shock_tt_me,shock_tt_mn_sa,shock_tt_me_sa ,gap):
		self.base_pv = base_pv
		self.shock_pv = shock_pv
		self.base_pv_mn = base_pv_mn
		self.shock_pv_mn = shock_pv_mn
		self.base_pv_me = base_pv_me
		self.base_tt_mn = base_tt_mn
		self.base_tt_me = base_tt_me
		self.shock_pv_me = shock_pv_me
		self.shock_tt_mn = shock_tt_mn
		self.shock_tt_me = shock_tt_me
		self.shock_tt_mn_sa = shock_tt_mn_sa
		self.shock_tt_me_sa = shock_tt_me_sa
		self.gap = gap
		self.CE_global = shock_pv-base_pv
		self.CE_mn = shock_pv_mn-base_pv_mn
		self.CE_me = shock_pv_me-base_pv_me
		self.scen_dates = gap.scen_dates
		self.CE_mn_pe = -self.CE_mn/gap.PE*100*(1-0.234)
		self.CE_me_pe = -self.CE_me/gap.PE*gap.TC*100*(1-0.234)
		self.CE_global_pe = -self.CE_global/gap.PE*100*(1-0.234)

	def get_table_1(self):
		df = pd.DataFrame(self.base_tt_mn.T.round(5)*100,columns=['Tasas base MN'],index=self.gap.buckets_lab)
		df['Tasas base ME'] = self.base_tt_me.T.round(5)*100
		df['Tasas shock MN'] = self.shock_tt_mn.T.round(5)*100
		df['Tasas shock ME'] = self.shock_tt_me.T.round(5)*100
		return df

	def get_graph_me(self):
		import plotly.offline as pl
		import plotly.graph_objs as plo

		data_b = plo.Scatter(x=self.gap.buckets_lab, y=self.base_tt_me*100, 
			mode='lines+markers', yaxis='y2',marker=plo.Marker(size=8), name='base ME') 
		
		data_s = plo.Scatter(x=self.gap.buckets_lab, y=self.shock_tt_me*100, 
			mode='lines+markers', yaxis='y2', marker=plo.Marker(size=8), name='shock ME') 
		
		data_gaps = plo.Bar(x=self.gap.buckets_lab, y=self.gap.gap_ME*100, 
				yaxis='y1', marker=plo.Marker(color='red'), name='Gaps ME') 
			
		layout =  plo.Layout(    title="TT en dolares"   ,
				yaxis=plo.YAxis(title='Gaps'),
                       yaxis2=plo.YAxis(title='Tasas',side='right',overlaying='y')
				)
		pdata = plo.Data([data_gaps,data_b,data_s])
		fig = plo.Figure(data=pdata, layout=layout)
		string =pl.plot(fig ,output_type='div',include_plotlyjs =False,show_link=False)
#		pl.iplot(fig)
		return string
		
	def get_graph_mn(self):
		import plotly.offline as pl
		import plotly.graph_objs as plo

		data_b = plo.Scatter(x=self.gap.buckets_lab, y=self.base_tt_mn*100, 
			mode='lines+markers', yaxis='y2',marker=plo.Marker(size=8),name='base MN') 
		
		data_s = plo.Scatter(x=self.gap.buckets_lab, y=self.shock_tt_mn*100, 
			mode='lines+markers', yaxis='y2', marker=plo.Marker(size=8), name='shock MN') 
		
		data_gaps = plo.Bar(x=self.gap.buckets_lab, y=self.gap.gap_MN*100, 
				yaxis='y1', marker=plo.Marker(color='red'), name='Gaps MN') 
			
		layout =  plo.Layout(    title="TT en soles"   ,
				yaxis=plo.YAxis(title='Gaps'),
                       yaxis2=plo.YAxis(title='Tasas',side='right',overlaying='y')
				)
		pdata = plo.Data([data_gaps,data_b,data_s])
		fig = plo.Figure(data=pdata, layout=layout)
		string =pl.plot(fig ,output_type='div',include_plotlyjs =False,show_link=False)
#		pl.iplot(fig)
		return string
	
		
class EcapResultsStack:
	def __init__(self):
		self.l = []
	def N(self):
		return len(self.l)
	def __getitem__(self,index):
		return self.l[index]
	def append(self,EcapResult):
		self.l.append(EcapResult)
	def __iter__(self):
		return iter(self.l)
	def get_table_0(self):
		df = pd.DataFrame(columns =['VP MN - base ',
		'VP ME - base',
		'VP Total - base',
		'VP MN - shock',
		'VP ME - shock (MM USD)',
		'VP Total - shock',
		'ECAP MN',
		'ECAP ME (MM USD)',
		'ECAP Global',
		'ECAP MN (%PE)',
		'ECAP ME (%PE)',
		'ECAP Global (%PE)',	])
		for sd in self:
			data1 = [round(sd.base_pv_mn,1) , round(sd.base_pv_me,1), round(sd.base_pv,1),
				round(sd.shock_pv_mn,1)  , round(sd.shock_pv_me,1),  round(sd.shock_pv,1),
				round(sd.CE_mn,1) , round(sd.CE_me,1) , round(sd.CE_global,1),
				round(sd.CE_mn_pe,2),round(sd.CE_me_pe,2),round(sd.CE_global_pe,2)]
			df.loc[sd.scen_dates]=data1
		return df
		
	def table_ce(self):
		tabla = self.get_table_0().to_html().replace('<table border="1" class="dataframe">', 
			'<table class="table table-striped table-bordered">')
		tabla = Markup(tabla)
		return tabla
		
	def scen_list(self):
		lista = []
		for r in self:
			elem = r.scen_dates
			elem = Markup(elem)
			lista.append(elem)
		return lista
		
	def table_list(self):
		lista = []
		for r in self:
			elem = r.get_table_1().T.to_html().replace('<table border="1" class="dataframe">', 
			'<table  class="table table-striped table-bordered">')
			elem = Markup(elem)
			lista.append(elem)
		return lista
		
	def plot_list_mn(self):
		lista = []
		for r in self:
			elem = r.get_graph_mn()
			elem = Markup(elem)
			lista.append(elem)
		return lista
		
	def plot_list_me(self):
		lista = []
		for r in self:
			elem = r.get_graph_me()
			elem = Markup(elem)
			lista.append(elem)
		return lista
		
class simulation_setup:
	def __init__(self,NSim,dt,N):
		self.NSim = NSim
		self.dt = dt
		self.N = N
		self.shape = [N,NSim]

class extra_data:
	def __init__(self,TC,PE):
		self.TC = TC
		self.PE = PE

class EcapEngine:
	def __init__(self, gap_stack ,model,correl ):
		self.gap_stack = gap_stack
		self.model = model
		self.correl = correl

	def compute(self,n_sim,task):
		n_var = 2
		n_time = 12
		dt = 1/n_time
		correl = self.correl
		import time as time	
		t0 =time.time()
		sim_opt = simulation_setup(n_sim,dt,n_time)
		np.random.seed(0)
		Omega = np.array([[1, correl],[correl,1]])
		mat_eps_correl = self.normal_correl_shocks(n_sim,n_var,n_time,Omega)
		mat_eps = mat_eps_correl[:,:,0].T
		t1 = time.time()-t0
		t0 = time.time() 		
		sd_s = EcapResultsStack()
		total =self.gap_stack.N
		print('total = {}'.format(total))
		for ii in range(0,total):
			base_pv_mn, base_tt_mn, shock_pv_mn_sa, shock_tt_mn_sa, yield_curves_mn_sa = self.get_ECAP_MN(ii,mat_eps,sim_opt)
			base_pv_me, base_tt_me, shock_pv_me_sa, shock_tt_me_sa, yield_curves_me_sa = self.get_ECAP_ME(ii,mat_eps,sim_opt)
			base_pv, shock_pv, shock_pv_mn, shock_pv_me, shock_tt_mn, shock_tt_me = \
			self.get_ECAP_global(ii,mat_eps_correl,sim_opt)
			sd  = EcapResult(base_pv, shock_pv, base_pv_mn, shock_pv_mn,
				base_pv_me, shock_pv_me, base_tt_mn, base_tt_me,shock_tt_mn, 
				shock_tt_me,shock_tt_mn_sa,shock_tt_me_sa ,self.gap_stack[ii] )
			sd_s.append(sd)
			task.update_state(state='PROGRESS',meta={'current': ii, 'total': total,
                                'status': 'in progress','result':'uff'+str(ii)})
			print(ii)
		t1 = time.time()-t0
		
		return sd_s

	def normal_correl_shocks(self,n_sim,n_var,n_time,omega):
		D,V = np.linalg.eig(omega)
		s = np.random.normal(0,1,[n_sim,n_time,n_var])
		T =V*np.sqrt(D)
		e_sim = np.zeros([n_sim,n_time,n_var])
		for ii in range(0,n_sim):
			e_sim[ii,:,:] = np.dot( T,s[ii,:,:].T).T
		return e_sim


	def get_pv_mn(self,tt,ii):
		n_buckets = len(self.gap_stack.buckets_num)
		buckets_num = self.gap_stack.buckets_num
		gap = self.gap_stack[ii].gap_MN
		NSim = tt.shape[0]

		discount_factors = np.zeros(tt.shape)
		for jj in range(0,n_buckets):
			discount_factors[:,jj] = 1/((1+tt[:,jj])**buckets_num[jj])
		gaps = np.kron(np.ones([NSim,1]), gap)
		present_values = np.sum(gaps*discount_factors,1)
		if len(present_values)==1:
			present_values = float(present_values)
		return present_values

	def get_pv_me(self,tt,ii):
		n_buckets = len(self.gap_stack.buckets_num)
		buckets_num = self.gap_stack.buckets_num
		gap = self.gap_stack[ii].gap_ME
		NSim = tt.shape[0]
		discount_factors = np.zeros(tt.shape)
		for jj in range(0,n_buckets):
			discount_factors[:,jj] = 1/((1+tt[:,jj])**buckets_num[jj])
		gaps = np.kron(np.ones([NSim,1]), gap)

		present_values = np.sum(gaps*discount_factors,1)
		
		if len(present_values)==1:
			present_values = float(present_values)
		return present_values

	def get_percentile_idx(self, sim_pv):
		shock_pv = np.percentile(sim_pv, 100-99.84)
		tmp = np.abs(sim_pv-shock_pv)
		indx = np.where(tmp == tmp.min())
		indx=indx[0][0]
		return indx

	def get_ECAP_MN(self,ii, mat_eps, sim_opt ):
		tt_mn = self.gap_stack[ii].tt_MN
		r0 = self.gap_stack[ii].tt_MN[0,0]
		dt = sim_opt.dt
		RR = self.model[0].sim_short_rate(r0,dt,mat_eps)
		r_shorts = RR[-1,:]
		yield_buckets = self.gap_stack.buckets_num
		yield_curves = self.model[0].get_yield_curves(r_shorts, yield_buckets)
		sim_pv = self.get_pv_mn(yield_curves,ii)
		base_pv = self.get_pv_mn(tt_mn,ii)
		base_tt = tt_mn[0,:]
		indx = self.get_percentile_idx(sim_pv) 
		shock_tt = yield_curves[indx,:]
		shock_pv = sim_pv[indx]
		return base_pv, base_tt, shock_pv, shock_tt, yield_curves

	def get_ECAP_ME(self,ii, mat_eps, sim_opt ):
		tt_me = self.gap_stack[ii].tt_ME
		r0 = self.gap_stack[ii].tt_ME[0,0]
		dt = sim_opt.dt
		RR = self.model[1].sim_short_rate(r0,dt,mat_eps)
		r_shorts = RR[-1,:]
		yield_buckets = self.gap_stack.buckets_num
		yield_curves = self.model[1].get_yield_curves(r_shorts, yield_buckets)
		sim_pv = self.get_pv_me(yield_curves,ii)
		base_pv = self.get_pv_me(tt_me,ii)
		base_tt = tt_me[0,:]
		indx = self.get_percentile_idx(sim_pv) 
		shock_tt = yield_curves[indx,:]
		shock_pv = sim_pv[indx]
		return base_pv, base_tt, shock_pv, shock_tt, yield_curves

	def get_ECAP_global(self,ii,mat_eps_c,sim_opt):
		tt_mn = self.gap_stack[ii].tt_MN
		tt_me = self.gap_stack[ii].tt_ME
		r0_mn = self.gap_stack[ii].tt_MN[0,0]
		r0_me = self.gap_stack[ii].tt_ME[0,0]
		dt = sim_opt.dt
		mat_eps_mn = mat_eps_c[:,:,0].T
		mat_eps_me = mat_eps_c[:,:,1].T
		RR_mn = self.model[0].sim_short_rate(r0_mn,dt,mat_eps_mn)
		RR_me = self.model[1].sim_short_rate(r0_me,dt,mat_eps_me)
		r_shorts_mn = RR_mn[-1,:]
		r_shorts_me = RR_me[-1,:]
		yield_buckets = self.gap_stack.buckets_num
		yield_curves_mn = self.model[0].get_yield_curves(r_shorts_mn, yield_buckets)
		yield_curves_me = self.model[1].get_yield_curves(r_shorts_me, yield_buckets)
		sim_pv_mn = self.get_pv_mn(yield_curves_mn,ii)
		sim_pv_me = self.get_pv_me(yield_curves_me,ii)
		sim_pv = sim_pv_mn + sim_pv_me*self.gap_stack[ii].TC
		base_pv_mn = self.get_pv_mn(tt_mn,ii)
		base_pv_me = self.get_pv_me(tt_me,ii)
		base_pv = base_pv_mn + base_pv_me*self.gap_stack[ii].TC
		indx = self.get_percentile_idx(sim_pv) 
		shock_tt_mn = yield_curves_mn[indx,:]
		shock_tt_me = yield_curves_me[indx,:]
		shock_pv = sim_pv[indx]
		shock_pv_mn = sim_pv_mn[indx]
		shock_pv_me = sim_pv_me[indx]
		return base_pv, shock_pv, shock_pv_mn, shock_pv_me, shock_tt_mn, shock_tt_me

import openpyxl as opxl
import xl_tools as xl

class FileComputation:
	def __init__(self, filename):
		wb = opxl.load_workbook(filename, data_only=True, use_iterators=False)
		xl_gap_MN = xl.xl_load(wb, 'gap_mn')
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

		p1 = CIR(theta1,kappa1,lambd_a1,sigma21)

		kappa = 0.435049
		sigma2 = 0.122388**2
		lambd_a = -0.35292
		theta = 0.0222862

		p2 = CIR(theta,kappa,lambd_a,sigma2)
		model_list = [p1,p2]
		correl = 0.0244875
		n_sim = 100000

		gs = GapStack( xl_scendates, xl_gap_MN, xl_gap_ME,
						xl_tt_MN, xl_tt_ME, buckets_num, buckets_label,TC,PE)
		eng = EcapEngine(gs,model_list,correl)
		
		self.eng = eng
		self.n_sim = n_sim
		# self.task = task

	def compute(self,task):
		results = self.eng.compute(self.n_sim,task)
		return results




