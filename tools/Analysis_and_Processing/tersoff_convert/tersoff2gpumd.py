
import numpy as np

in_file = "1988.tersoff"
out_file = "gpumd.txt"
version = '1989'  # 1988、1989
chi = 1.0

tersoff_file = np.array(np.loadtxt(in_file, dtype=str))
if tersoff_file.ndim == 1:
    tersoff_file = tersoff_file.reshape(1, -1)
e1, e2, e3 = tersoff_file.T[:3]
m, G, L3, c, d, ct0, n, b, L2, B, R, D, L1, A = np.round(tersoff_file.T[3:].astype(float), 4)

symbols = []
for i in range(len(tersoff_file)):
    if e1[i] == e2[i] == e3[i]:
        symbols.append(e1[i])

with open(out_file, 'w') as f:
    if version == '1988':
        f.write(f'tersoff_1988 {len(symbols)} {" ".join(symbols)}\n')
        for i in range(len(symbols)**3):
            idx = f"{i:0{3}b}"  # 如 '101'
            combo = [symbols[int(b)] for b in idx]
            for j in range(len(tersoff_file)):
                if combo[0] == e1[j] and combo[1] == e2[j] and combo[2] == e3[j]:
                    f.write(f"{A[j]} {B[j]} {L1[j]} {L2[j]} {b[j]} {n[j]} {c[j]} {d[j]} {ct0[j]} {L2[j]} {R[j]-D[j]} {R[j]-D[j]+0.3} {m[j]} {L3[j]} {G[j]}\n") 
    elif version == '1989':
        f.write(f'tersoff_1989 {len(symbols)} {" ".join(symbols)}\n')
        for i in range(len(tersoff_file)):
            if e1[i] == e2[i] == e3[i]:
                f.write(f"{A[i]} {B[i]} {L1[i]} {L2[i]} {b[i]} {n[i]} {c[i]} {d[i]} {ct0[i]} {L2[i]} {R[i]-D[i]} {R[i]-D[i]+0.3}\n")
        if len(symbols) == 2:
            f.write(f"{chi}\n")         

