"""Checkpoint trajectory parser shared by Spin integration tests."""
import re
import numpy as np

def frames(path):
    lines=path.read_text().splitlines();n=int(lines[0]);result=[]
    # Keep all post-checkpoint steps, not just the endpoint.
    for start in range(0,len(lines),n+2):
        h=lines[start+1];props=re.search(r'Properties=([^\s]+)',h)[1].split(':')
        raw=[line.split() for line in lines[start+2:start+2+n]];offset=0;v={}
        for i in range(0,len(props),3):
            key,kind,width=props[i],props[i+1],int(props[i+2]);key={'forces':'force','energy_atom':'potential'}.get(key,key)
            if kind!='S':v[key]=np.array([r[offset:offset+width] for r in raw],float)
            offset+=width
        v['cell']=np.array(re.search(r'Lattice="([^"]+)"',h)[1].split(),float)
        v['time']=np.array([float(re.search(r'Time=([^\s]+)',h)[1])]);result.append(v)
    return result
