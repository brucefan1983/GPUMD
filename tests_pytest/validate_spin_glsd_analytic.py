#!/usr/bin/env python3
"""Independent continuous-time harmonic/Zeeman and Boltzmann acceptance gates."""
import json, math, subprocess, sys
rows=[json.loads(s) for s in subprocess.check_output([sys.argv[1]],text=True).splitlines()]
kb=8.617343e-5
for name in ('damped_field','precession'):
    cases=[r for r in rows if r['name']==name]
    ratios=[a['error']/b['error'] for a,b in zip(cases,cases[1:])]
    assert all(3.5 < r < 4.6 for r in ratios), (name,ratios)
    assert cases[-1]['error'] < .0003, cases
for r in rows:
    name=r['name']
    assert all(math.isfinite(x) for x in r['mean']+r['variance']),r
    if name=='radial': assert r['error'] < 1e-8 and r['energy'] < 1e-12,r
    if name=='zero_field': assert r['error'] < 1e-12,r
    if name=='zero_rest': assert r['error']==0,r
    if name=='damped_field': assert r['energy'] < -.0498,r
    if name in ('thermal','thermal_field','thermal_zero_transient'):
        target=kb*r['temperature']/.4
        if name=='thermal_zero_transient': target*=1-math.exp(-2*2*.4*r['time_ps'])
        for d in range(3):
            mu=.5 if name=='thermal_field' and d==2 else 0
            assert abs(r['mean'][d]-mu) < 6*math.sqrt(target/r['n']),r
            assert abs(r['fourth'][d]/r['variance'][d]**2-3)<.08,r
        r['variance_relative_bias']=sum(r['variance'])/(3*target)-1
        if r['dt_ps'] <= .05: assert abs(r['variance_relative_bias']) < .012,r
        if name=='thermal' and r['dt_ps']==.05:
            cdf=math.erf(math.sqrt(1.5))-math.sqrt(6/math.pi)*math.exp(-1.5)
            assert abs(r['radial_cdf']-cdf)<.008,r
for temp in (100,400):
    biases=[abs(r['variance_relative_bias']) for r in rows if r['name']=='thermal' and r['temperature']==temp]
    assert biases[0]>biases[1]>biases[2],biases
print(json.dumps({'status':'passed','cases':rows},indent=2))
