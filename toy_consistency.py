"""Toy consistency study (Phase -1 sanity simulation, NOT evidence about real data): corridor degeneracy, radar Doppler + LiDAR + IMU-like prior.
Sequential estimation of body velocity v (3D) over windows (mimics marginalized prior).
Estimators:
  FIX   : nominal weights, LiDAR correlated sliding error unmodeled
  TRAIL : TRaIL-style alpha/gamma on radar sqrt-weights (info scales with (gamma*alpha)^2)
  HARD  : LiDAR information removed along weak subspace (DR-LRIO / X-ICP-style projection)
  NUIS  : proposed nuisance-shift: LiDAR residuals get shared shift delta along U_d ~ N(0, s_amb^2), marginalized
"""
import numpy as np, sys
rng = np.random.default_rng(0)
T = 0.1                     # window length [s]
sig_L, sig_D = 0.02, 0.05   # LiDAR point-to-plane [m], Doppler [m/s] (true)
q = 0.05                    # velocity random-walk std per window [m/s]
s_amb_true = 0.03           # correlated along-corridor association error per window [m]
eta, tau = 2.2, 0.2

def make_lidar(n=300, frac_x=0.01):
    V = np.zeros((n,3)); V[:,1:] = rng.normal(size=(n,2)); V += 0.02*rng.normal(size=(n,3))
    m = rng.random(n) < frac_x
    W = rng.normal(size=(m.sum(),3)); W[:,0] = abs(W[:,0])+0.3; V[m] = W
    return V/np.linalg.norm(V,axis=1,keepdims=True)

def make_radar(m=30):
    az = rng.uniform(-np.pi/3, np.pi/3, m); el = rng.uniform(-np.pi/6, np.pi/12, m) - np.deg2rad(20)
    return np.stack([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)], 1)

def weak_subspace(Nm):
    H = Nm.T @ Nm; lam, V = np.linalg.eigh(H)  # ascending
    Ud = V[:, lam/lam[-1] < tau]; S = np.cbrt(max(np.prod(lam),0))/(lam.sum()/3)
    return Ud, S, H

def run(method, n_win=400, s_amb_model=s_amb_true, radar_noise_mult=1.0, outlier_frac=0.0, rho=0.8, sigL_assumed=None, sigD_assumed=None):
    v_true = np.array([1.0, 0, 0]); v_hat = v_true.copy(); P = np.eye(3)*0.01
    pos_err = 0.0; e_list=[]; nees=[]; dnees=[]
    for k in range(n_win):
        v_true = v_true + q*rng.normal(size=3)*np.array([1,0.3,0.1])
        # predict (IMU-like random walk prior)
        P = P + np.diag((q*np.array([1,0.3,0.1]))**2)
        Nm = make_lidar(); L = make_radar()
        Ud, S, H = weak_subspace(Nm)
        # LiDAR: measures n^T (v T + shift) ; correlated shift along x
        v_pred = v_hat.copy()
        shift = (rho*(v_pred-v_true)[0]*T + s_amb_true*rng.normal())*np.array([1,0,0])  # ICP sticks to prediction along corridor
        zL = Nm @ (v_true*T + shift) + sig_L*rng.normal(size=len(Nm))
        # Radar: Doppler on LOS, true noise may be larger than assumed
        zD = L @ v_true + sig_D*radar_noise_mult*rng.normal(size=len(L))
        nout = int(outlier_frac*len(L)); zD[:nout] += rng.uniform(-1,1,nout)  # dynamic/multipath
        # build information form for measurement update on v
        JL = Nm*T
        if method == 'HARD' and Ud.shape[1] > 0:
            Pk = np.eye(3) - Ud@Ud.T            # remove LiDAR info along U_d
            JL = JL @ Pk
        # R_L = sig_L^2 I + s^2 B B^T  (B = N U_d): Woodbury for R_L^{-1}
        def RLi_mul(X):
            sL = sigL_assumed if sigL_assumed else sig_L
            Y = X/sL**2
            if method == 'NUIS' and Ud.shape[1] > 0:
                B = Nm@Ud; C = np.eye(B.shape[1])/s_amb_model**2 + B.T@B/sL**2
                Y = Y - (B/sL**2) @ np.linalg.solve(C, B.T@Y)
            return Y
        sD = sigD_assumed if sigD_assumed else sig_D
        wD = np.ones(len(L))/sD**2
        if method == 'TRAIL' and Ud.shape[1] > 0:
            Z = (L @ Ud); a = (Z**2).sum(1); al = np.exp(eta*a); al /= al.mean()
            g = 1/max(S,1e-3)
            wD = wD*(g*al)**2                    # sqrt-weight scaling -> squared in information
        I_meas = JL.T@RLi_mul(JL) + L.T@(wD[:,None]*L)
        b_meas = JL.T@RLi_mul(zL) + L.T@(wD*zD)
        Ipr = np.linalg.inv(P)
        Ipost = Ipr + I_meas
        P = np.linalg.inv(Ipost)
        v_hat = P @ (Ipr@v_hat + b_meas)
        e = v_hat - v_true; e_list.append(e)
        nees.append(e@np.linalg.solve(P,e))
        dnees.append(e[0]**2/P[0,0])
        pos_err += e[0]*T
    E = np.array(e_list)
    return dict(rmse_vx=np.sqrt((E[:,0]**2).mean()), rmse_v=np.sqrt((E**2).sum(1).mean()),
                anees=np.mean(nees[50:]), dnees_x=np.mean(dnees[50:]), drift_x=abs(pos_err))

def mc(method, reps=10, **kw):
    R = [run(method, **kw) for _ in range(reps)]
    return {k: np.mean([r[k] for r in R]) for k in R[0]}

if __name__ == '__main__':
    import sys
    up = dict(sigL_assumed=0.002, sigD_assumed=0.02)   # upstream sqrt-weights 500 / 50
    cal = {}                                           # true noise (calibrated)
    rows = [('FIX  (upstream w)', 'FIX', up), ('TRAIL(upstream w)', 'TRAIL', up),
            ('HARD (upstream w)', 'HARD', up), ('NUIS (upstream w)', 'NUIS', up),
            ('CAL  (calibrated)', 'FIX', cal), ('HARD (calibrated)', 'HARD', cal), ('NUIS (calibrated)', 'NUIS', cal)]
    scen = {'nominal': {}, 'radar noise x2': dict(radar_noise_mult=2.0), 'radar 10% outliers': dict(outlier_frac=0.1)}
    which = sys.argv[1]
    kw = scen[which]
    print("==", which)
    for name, m, w in rows:
        r = mc(m, **kw, **w)
        print(f"{name:18s} rmse_vx={r['rmse_vx']:.4f} rmse_v={r['rmse_v']:.4f} ANEES={r['anees']:9.2f} dNEES_x={r['dnees_x']:9.2f} |drift_x|={r['drift_x']:.3f}")
