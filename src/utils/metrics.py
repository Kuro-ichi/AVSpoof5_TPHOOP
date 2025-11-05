import numpy as np

def compute_eer(scores, labels):
    # labels: 1=bonafide, 0=spoof; we compute EER on spoof-as-positive or vice versa consistently
    # Here define positive=spoof => use 1 - score (bona-fide prob)
    pos = 1.0 - np.asarray(scores)
    y = 1 - np.asarray(labels).astype(int)
    # sweep thresholds
    ths = np.unique(pos)
    frrs, fars = [], []
    for t in ths:
        y_pred = (pos >= t).astype(int)
        fa = ((y_pred==1) & (y==0)).sum() / max((y==0).sum(),1)   # bona->spoof
        fr = ((y_pred==0) & (y==1)).sum() / max((y==1).sum(),1)   # spoof->bona
        fars.append(fa); frrs.append(fr)
    fars, frrs = np.array(fars), np.array(frrs)
    idx = np.argmin(np.abs(fars-frrs))
    eer = 0.5*(fars[idx]+frrs[idx])
    return float(eer)

def compute_min_tdcf(scores, labels, P_tar=0.01, C_miss=1.0, C_fa=1.0):
    # Simplified min-tDCF variant for reference only
    pos = 1.0 - np.asarray(scores)
    y = 1 - np.asarray(labels).astype(int)  # 1=spoof
    ths = np.unique(pos)
    best = 1e9
    for t in ths:
        y_pred = (pos >= t).astype(int)
        P_miss = ((y_pred==0) & (y==1)).sum() / max((y==1).sum(),1)
        P_fa   = ((y_pred==1) & (y==0)).sum() / max((y==0).sum(),1)
        tdcf = C_miss*P_miss*P_tar + C_fa*P_fa*(1-P_tar)
        if tdcf < best:
            best = tdcf
    return float(best)
