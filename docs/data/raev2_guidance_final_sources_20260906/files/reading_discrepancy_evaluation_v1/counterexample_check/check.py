import mpmath as mp
mp.mp.dps=90
phi=lambda x: mp.exp(-x*x/2)/mp.sqrt(2*mp.pi)
Phi=lambda x: (1+mp.erf(x/mp.sqrt(2)))/2
width=mp.mpf('0.1'); x=mp.mpf(1)
for literal in ['.2','.5','1','2','4']:
 sigma=mp.mpf(literal)
 def conv(mu):
  upper=(x-mu+width)/sigma; lower=(x-mu-width)/sigma
  density=(Phi(upper)-Phi(lower))/(2*width)
  derivative=(phi(upper)-phi(lower))/(2*width*sigma)
  return density,derivative
 qp,dp=conv(mp.mpf(1)); qm,dm=conv(mp.mpf(-1))
 gap=abs(dp/qp-(dp+dm)/(qp+qm))
 upper=mp.mpf('2.2')/(sigma*sigma)
 print('sigma='+literal, 'gap='+mp.nstr(gap,32), 'bound='+mp.nstr(upper,12), 'valid='+str(gap<=upper))
# Signed projection: g=(1,0), correction=(-2,3).
g=(mp.mpf(1),mp.mpf(0)); v=(mp.mpf(-2),mp.mpf(3))
a=sum(x*y for x,y in zip(g,v))/sum(x*x for x in g)
perp=tuple(vv-a*gg for vv,gg in zip(v,g))
print('signed_scale',a,'norm_ratio',abs(a),'perp',perp,'Eq8_using_norm_ratio',tuple(abs(a)*gg+pp for gg,pp in zip(g,perp)))
print('two_Euler_schedule_0_2_endpoint',1*(1+mp.mpf('.5')*0)*(1+mp.mpf('.5')*2))
print('two_Euler_constant_mean1_endpoint',1*(1+mp.mpf('.5'))**2)
