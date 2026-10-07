import os, sys, time
os.environ['OMP_NUM_THREADS']='1'
import _paths, numpy as np, mfc, mpsenh as M
import jax.numpy as jnp
N=20; chi=int(sys.argv[1]); chiw=2*chi
rng=np.random.default_rng(0)
def rand_mps(c):
    b=[1]+[min(c,2**min(i,N-i)) for i in range(1,N)]+[1]
    return [rng.normal(size=(b[i],2,b[i+1]))+1j*rng.normal(size=(b[i],2,b[i+1])) for i in range(N)]
Tw=mfc.normalise(mfc.right_canonicalize(rand_mps(chiw))); T=mfc.normalise(mfc.right_canonicalize(rand_mps(chi)))
Twj=[jnp.asarray(t) for t in Tw]
taus=mfc.marginals(Twj,(1,2,3))
vg,unpack=mfc.make_objective([t.shape for t in T],(1,2,3),0.01,{1:1.0,2:1.0,3:1.0})
x=jnp.asarray(mfc.pack(T))
t0=time.time(); v,g=vg(x,taus,Twj); g.block_until_ready(); print('compile+first eval %.1fs'%(time.time()-t0))
t0=time.time()
for _ in range(10): v,g=vg(x,taus,Twj); g.block_until_ready()
print(f'chi={chi}: {(time.time()-t0)/10*1000:.0f} ms per value+grad')
