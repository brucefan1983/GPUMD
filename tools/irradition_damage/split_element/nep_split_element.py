import numpy as np

def dump (file, data, Types):
    nframe = int(data['nframe'])
    fout = open(file, 'w')
    outstr = ''
    outstr2 = ''
    frame = 0
    for i in range(nframe):
      if data['atom_type'][i] in Types:
        outstr = outstr + str(data['atom_numbs'][i])
        outstr2 = outstr2 + ''.join(data['atom'][i])
        frame = frame + 1
        print(frame)
        print(data['atom_type'][i])
    outstr = str(frame) + '\n' + outstr + outstr2
    fout.write(outstr)
    fout.close()


def read_nep(file, Types, version = 1):

    f = open(file)
    data = {}
    data['nframe'] = f.readline()
    data['atom_numbs'] = {}
    data['atom'] = {}
    data['atom_type'] = {}
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

    # distinguish type past
    if (version):
        # distinguish type new
        for i in range(nframe):
            type1 = ''
            type2 = ''
            type3 = ''
            for j in range(data['numbs'][i]):
                type = data['atom'][i][2 + j].split(' ')[0]
                if type == Types[0]:
                    type1 = Types[0]
                elif type == Types[1]:
                    type2 = Types[1]
                elif type == Types[2]:
                    type3 = Types[2]
            data['atom_type'][i] = type1 + type2 + type3
    else:
        # distinguish type past
        for i in range(nframe):
            type1 = ''
            type2 = ''
            type3 = ''
            for j in range(data['numbs'][i]):
                type = int(data['atom'][i][2 + j].split(' ')[0])
                if type == 0:
                    type1 = Types[0]
                    temp = list(data['atom'][i][2 + j])
                    temp[0] = Types[0]
                    data['atom'][i][2 + j] = ''.join(temp)
                elif type == 1:
                    type2 = Types[1]
                    temp = list(data['atom'][i][2 + j])
                    temp[0] = Types[1]
                    data['atom'][i][2 + j] = ''.join(temp)
                elif type == 2:
                    type3 = Types[2]
                    temp = list(data['atom'][i][2 + j])
                    temp[0] = Types[2]
                    data['atom'][i][2 + j] = ''.join(temp)
            data['atom_type'][i] = str(type1) + str(type2) + str(type3)
    f.close()
    return data



def main():
    Types =['Mg','Al','Cu']
    Choose= ['MgAl', 'Mg', 'Al']
    data = read_nep('./MgAlCu_train.in',Types)
    dump('./train.in', data, Choose)

if __name__ == "__main__":
    main()
