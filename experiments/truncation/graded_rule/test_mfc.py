import os
os.environ['OMP_NUM_THREADS']='1'
import _paths, numpy as np, mfc, mpsenh as M
import jax.numpy as jnp
# 1. MPS-native marginals against dense marginals, on a random MPS
rng=np.random.default_rng(0); N=8
bonds=[1,3,5,7,8,7,5,3,1]
T=[rng.normal(size=(bonds[i],2,bonds[i+1]))+1j*rng.normal(size=(bonds[i],2,bonds[i+1])) for i in range(N)]
v=M.mps_to_dense(T); v=v/np.linalg.norm(v)
marg=mfc.marginals([jnp.asarray(t) for t in T],(1,2,3))
from hp_bench import rdmk
for k in (1,2,3):
    d=max(np.abs(np.asarray(marg[k][i])-rdmk(v,N,i,k)).max() for i in range(N-k+1))
    print(f'marginals k={k}: max |MPS-native - dense| = {d:.2e}')
# 2. overlap
Tb=[t+0.3*(rng.normal(size=t.shape)+1j*rng.normal(size=t.shape)) for t in T]
vb=M.mps_to_dense(Tb)
print('overlap diff', abs(complex(mfc.overlap([jnp.asarray(t) for t in T],[jnp.asarray(t) for t in Tb]))-np.vdot(M.mps_to_dense(T),vb)))
# 3. right_canonicalize / compress_svd preserve the state (chi large) and truncate correctly
Tc=mfc.right_canonicalize(T); print('canonicalise state diff', np.linalg.norm(M.mps_to_dense(Tc)-M.mps_to_dense(T)))
Ts=mfc.compress_svd(Tc,3); vs=M.mps_to_dense(Ts); print('compress chi=3 max bond',mfc.max_bond(Ts),' fidelity',abs(np.vdot(v,vs/np.linalg.norm(vs)))**2)
