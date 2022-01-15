import numpy as np

def read_restart(file, num, vx, vy, vz):
   f= open(file)
   data = {}
   line = f.readline()
   data['header'] = line
   N = int(line.split(' ')[0])
   data['N'] = N
   data['box'] = f.readline()
   data['atom']={}
   data['velocity']=np.zeros([N,3], dtype=float)
   for i in range(N):
       line = f.readline()
       atom = line.split(' ')
       data['atom'][i] = atom[0:5]
       data['velocity'][i, 0] = float(atom[5])
       data['velocity'][i, 1] = float(atom[6])
       data['velocity'][i, 2] = float(atom[7])
   kx = data['velocity'][:, 0].sum()
   ky = data['velocity'][:, 1].sum()
   kz = data['velocity'][:, 2].sum()
   print(kx, ky, kz)
   mx = (vx - data['velocity'][num-1, 0])/ (N - 1)
   my = (vy - data['velocity'][num-1, 1])/ (N - 1)
   mz = (vz - data['velocity'][num-1, 2])/ (N - 1)
   data['velocity'][:, 0] = data['velocity'][:, 0] - mx
   data['velocity'][:, 1] = data['velocity'][:, 1] - my
   data['velocity'][:, 2] = data['velocity'][:, 2] - mz
   data['velocity'][num - 1, 0] = vx
   data['velocity'][num - 1, 1] = vy
   data['velocity'][num - 1, 2] = vz
   kx = data['velocity'][:, 0].sum()
   ky = data['velocity'][:, 1].sum()
   kz = data['velocity'][:, 2].sum()
   print(kx, ky, kz)
   return data

def dump(file, data):
    fout = open(file, 'a')
    outstr =str(data['header']) + str(data['box'])
    fout.write(outstr)
    for i in range(data['N']):
        outstr =  '  '.join(data['atom'][i]) + ' ' + ' '.join(str(i) for i in data['velocity'][i]) + '\n'
        fout.write(outstr)
    fout.close()

def main():
    data = read_restart('./restart.out', 1010103, 14500, 0, 0)
    dump('./xyz.in', data)

if __name__ == "__main__":
    main()
