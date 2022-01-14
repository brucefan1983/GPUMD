import numpy as np

def dump (file, type, data, force_error, energy_error, virial_error):
    nframe = int(data['nframe'])
    fout = open(file, 'w')
    typeout = open(type, 'w')
    outstr = ''
    outstr2 = ''
    outstr3 = ''
    frame = 0
    numbs = 0
    frame_force_error = 0
    for i in range(nframe):
        numbs2 = numbs + data['numbs'][i]
        for j in range(numbs,numbs2):
            if (force_error[j]):
                type = data['atom'][i][2 + j - numbs].split(' ')[0]
                outstr3 = outstr3 + '\n' + type
                frame_force_error = 1
        if (frame_force_error or energy_error[i] or virial_error[i]):
            outstr = outstr + str(data['atom_numbs'][i])
            outstr2 = outstr2 + ''.join(data['atom'][i])
            frame = frame + 1
            print(frame)
        numbs = numbs2
        frame_force_error = 0
    outstr = str(frame) + '\n' + outstr + outstr2
    fout.write(outstr)
    typeout.write(outstr3)
    fout.close()
    typeout.close()

def read_nep(file):

    f = open(file)
    data = {}
    data['nframe'] = f.readline()
    data['atom_numbs'] = {}
    data['atom'] = {}
    nframe = int(data['nframe'])
    data['numbs'] = np.arange(nframe)
    counter = 0
    for i in range(nframe):
        line = f.readline()
        data['atom_numbs'][i] = line
        data['numbs'][i] = line.split(' ')[0]
    lines = f.readlines()
    for i in range(nframe):
        data['atom'][i] = lines[counter: counter + data['numbs'][i] + 2]
        counter = counter + data['numbs'][i] + 2

    f.close()
    return data

def read_force_error(file, switch = 1):

    f = open(file)
    force = np.loadtxt(file, dtype=np.float32, delimiter=' ')
    N = len(force)
    force_error = np.zeros(N)
    if(switch):
        for i in range(N):
            loss = ((((force[i][5] - force[i][2]) ** 2) + ((force[i][4] - force[i][1]) ** 2) + ((force[i][3] - force[i][0]) ** 2))/3) ** 0.5
            print(i)
            if loss > 0.5:
                force_error[i] = 1
    f.close()
    return force_error

def read_energy_error(file, switch = 1):
    f = open(file)
    energy = np.loadtxt(file, dtype=np.float32, delimiter=' ')
    N = len(energy)
    energy_error = np.zeros(N)
    if(switch):
        for i in range(N):
            loss = abs(energy[i][1]-energy[i][0])
            if loss > 0.02:
                energy_error[i] = 1
    f.close()
    return energy_error

def read_virial_error(file, switch = 1):
    f = open(file)
    virial = np.loadtxt(file, dtype=np.float32, delimiter=' ')
    N = len(virial)
    virial_error = np.zeros(N)
    if(switch):
        for i in range(N):
            loss = abs(virial[i][1] - virial[i][0])
            if loss > 0.6:
                virial_error[i] = 1
    f.close()
    return virial_error

def main():
    data = read_nep('./test.in')
    force_error = read_force_error('./force_test.out', 0)
    energy_error = read_energy_error('./energy_test.out')
    virial_error = read_virial_error('./virial_test.out', 0)
    dump('./active_train.in', './type.out', data, force_error, energy_error, virial_error)

if __name__ == "__main__":
    main()
