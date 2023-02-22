import numpy as np
from ase.io import read
from pandas import DataFrame

f = np.loadtxt('observer0.out')

time_step = 5e-3
data = []
for k, (frame0, frame1, row0, row1) in enumerate(zip(read('observer0.xyz', ':'),
                                                     read('observer1.xyz', ':'),
                                                     np.loadtxt('observer0.out'),
                                                     np.loadtxt('observer1.out'),
                                                     )):
    time = k * time_step
    energy0 = frame0.get_potential_energy()
    energy1 = frame1.get_potential_energy()
    alats = np.linalg.norm(frame0.cell, axis=1)
    data.append(dict(time=time,
                     energy0_exyz=energy0,
                     energy1_exyz=energy1,
                     energy0_thermo=row0[2],
                     energy1_thermo=row1[2],
                     alat=alats[0],
                     blat=alats[1],
                     clat=alats[2],
                     ))
df = DataFrame.from_dict(data)
# df.to_json('data.json')
print(df)
