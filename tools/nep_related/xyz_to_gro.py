from ase.io import read
import math
all_frames = read("dump.xyz",index=":")
wrap = True
dt = 1
with open("system.gro",'w') as writer:
    for index_j,i in enumerate(all_frames):
        i.wrap()
        n_of_atoms =i.get_global_number_of_atoms()
        writer.write("t= %7.4f\n%d\n" %((dt*index_j+dt)/1000.0,n_of_atoms))  #ps
        ind=1
        elements = i.get_chemical_symbols()
        positions = i.positions
        for index in range(n_of_atoms):
            writer.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" %(ind,"MOL",elements[index],ind,positions[index][0]/10.0,positions[index][1]/10.0,positions[index][2]/10.0))
            ind+=1
        a,b,c,alpha,beta,gamma=i.cell.cellpar()   
        alpha = alpha /180*math.pi
        beta = beta /180*math.pi
        gamma = gamma /180*math.pi        
        bc2 = b**2 + c**2 - 2*b*c*math.cos(alpha)
        h1 = a
        h2 = b * math.cos(gamma)
        h3 = b * math.sin(gamma)
        h4 = c * math.cos(beta)
        h5 = ((h2 - h4)**2 + h3**2 + c**2 - h4**2 - bc2)/(2 * h3)
        h6 = math.sqrt(c**2 - h4**2 - h5**2)
        newlattice = [[h1, 0., 0.], [h2, h3, 0.], [h4, h5, h6]]  #rebuild axes
        writer.write("%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f%8.4f\n" %(newlattice[0][0]/10.0,newlattice[1][1]/10.0,
        newlattice[2][2]/10.0,0.0,0.0,newlattice[1][0]/10.0,0.0,newlattice[2][0]/10.0,newlattice[2][1]/10.0))
#v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y), the last 6 values may be omitted (they will be set to zero). GROMACS only supports boxes with v1(y)=v1(z)=v2(z)=0. 