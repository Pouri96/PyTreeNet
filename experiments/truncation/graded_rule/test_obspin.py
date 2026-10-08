"""Checks for rule/obspin.py: gradients vs finite differences, residual reduction, tangent fidelity, valid MPS."""
import sys
import numpy as np
import _paths  # noqa
import mpsenh as M
import obspin

N, chi, dt, T = 10, 6, 0.1, 2.0
model = 'ising'
cap = {}


class Probe(obspin.ObsPin):
    def _eval(self, Mn, ctx, grad):
        if grad and 'ctx' not in cap and ctx['l'] >= 4 and ctx['r'] >= 4:
            cap['ctx'], cap['M'], cap['self'] = ctx, Mn.copy(), self
        return super()._eval(Mn, ctx, grad)


for energy in (0.0, 1.0):
    print('energy weight', energy)
    cap.clear()
    pin = Probe(model, N, a=1, k=2, taus=(1.0,), mech='tan', passes=4, w_energy=energy)
    Tm, _ = M.run_tebd(model, N, chi, int(T / dt), dt, pin, gates=M.make_gates(model, N, dt))
    if energy == 0:
        ctx, M0 = cap['ctx'], cap['M']
        rng = np.random.default_rng(0)
        d = rng.normal(size=M0.shape) + 1j * rng.normal(size=M0.shape)
        d /= np.linalg.norm(d)
        o0, G = pin._eval(M0, ctx, True)
        eps = 1e-6

        def val(X):
            X = X / np.linalg.norm(X)
            return pin._eval(X, ctx, False)[0]
        fd = (val(M0 + eps * d) - val(M0 - eps * d)) / (2 * eps)
        an = 2 * np.real(np.einsum('jab,ab->j', G.conj(), d))
        print(f'rows {len(o0)}  gradient FD error {np.max(np.abs(fd - an)):.2e} (scale {np.max(np.abs(an)):.2e})')
    lg = np.array(pin.log)
    print(f'   cuts {pin.calls}  fired {pin.fired}  skipped {pin.skipped}  residual after/before median {np.median(lg[:, 1] / lg[:, 0]):.3f}'
          f'  cut infidelity pin/svd median {np.median(lg[:, 3] / lg[:, 2]):.3f}  max bond {max(M.ranks(Tm))}')
    v = M.mps_to_dense(Tm)
    print(f'   state norm {np.linalg.norm(v):.6f}')

# every mechanism, hard check of the pin on a single set of cuts
ex = M.dense_reference(model, N, int(T / dt), dt, gates=M.make_gates(model, N, dt))
Ts, _ = M.run_tebd(model, N, chi, int(T / dt), dt, M.svd_cut, gates=M.make_gates(model, N, dt))
es = M.errors(ex, M.mps_to_dense(Ts), N, model)
print('svd   ', {k: f'{v:.2e}' for k, v in es.items()})
for mech in ('tan', 'fix', 'coef'):
    pin = obspin.ObsPin(model, N, a=0, k=2, mech=mech, passes=4)
    Tm, _ = M.run_tebd(model, N, chi, int(T / dt), dt, pin, gates=M.make_gates(model, N, dt))
    e = M.errors(ex, M.mps_to_dense(Tm), N, model)
    lg = np.array(pin.log)
    print(f'{mech:6s}', {k: f'{v:.2e}' for k, v in e.items()}, f'fired {pin.fired}/{pin.calls} res {np.median(lg[:,1]/lg[:,0]):.3f} cutinf {np.median(lg[:,3]/lg[:,2]):.2f}')

for we in (0.0, 1.0, 10.0):
    pin = obspin.ObsPin(model, N, a=0, k=2, mech='tan', passes=4, w_energy=we)
    Tm, _ = M.run_tebd(model, N, chi, int(T / dt), dt, pin, gates=M.make_gates(model, N, dt))
    e = M.errors(ex, M.mps_to_dense(Tm), N, model)
    print(f'a=0 tan  w_energy={we:4.1f}', {k: f'{v:.2e}' for k, v in e.items()}, f'fired {pin.fired}')
