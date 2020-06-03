/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/


/*----------------------------------------------------------------------------80
The Bussi-Donadio-Parrinello thermostat:
[1] G. Bussi et al. J. Chem. Phys. 126, 014101 (2007).
------------------------------------------------------------------------------*/


#include "ensemble_bdp.cuh"
#include "utilities/common.cuh"
#define DIM 3


// These functions are from  Bussi's website
// https://sites.google.com/site/giovannibussi/Research/algorithms
// See the end of this file for the function definitions
static double resamplekin(double kk, double sigma, int ndeg, double taut);
static double resamplekin_sumnoises(int nn);
static double ran1();
static double gasdev();
static double gamdev(const int ia);


Ensemble_BDP::Ensemble_BDP(int t, int fg, double T, double Tc)
{
    type = t;
    fixed_group = fg;
    temperature = T;
    temperature_coupling = Tc;
}


Ensemble_BDP::Ensemble_BDP
(
    int t,
    int fg,
    int source_input,
    int sink_input,
    double T,
    double Tc,
    double dT
)
{
    type = t;
    fixed_group = fg;
    temperature = T;
    temperature_coupling = Tc;
    delta_temperature = dT;
    source = source_input;
    sink = sink_input;
    // initialize the energies transferred from the system to the baths
    energy_transferred[0] = 0.0;
    energy_transferred[1] = 0.0;
}


Ensemble_BDP::~Ensemble_BDP(void)
{
    // nothing now
}


void Ensemble_BDP::integrate_nvt_bdp_2
(
    const double time_step,
    const double volume,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& potential_per_atom,
    const GPU_Vector<double>& force_per_atom,
    const GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    GPU_Vector<double>& thermo
)
{
    const int number_of_atoms = mass.size();

    velocity_verlet
    (
        false,
        time_step,
        group,
        mass,
        force_per_atom,
        position_per_atom,
        velocity_per_atom
     );

    // get thermo
    int N_fixed = (fixed_group == -1) ? 0 : group[0].cpu_size[fixed_group];
    find_thermo
    (
        volume,
        group,
        mass,
        potential_per_atom,
        velocity_per_atom,
        virial_per_atom,
        thermo
    );

    // re-scale the velocities
    double ek[1];
    thermo.copy_to_host(ek, 1);
    int ndeg = 3 * (number_of_atoms - N_fixed);
    ek[0] *= ndeg * K_B * 0.5; // from temperature to kinetic energy
    double sigma = ndeg * K_B * temperature * 0.5;
    double factor = resamplekin(ek[0], sigma, ndeg, temperature_coupling);
    factor = sqrt(factor / ek[0]);
    scale_velocity_global(factor, velocity_per_atom);
}


// integrate by one step, with heating and cooling, using the BDP method
void Ensemble_BDP::integrate_heat_bdp_2
(
    const double time_step,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom
)
{
    int label_1 = source;
    int label_2 = sink;
    int Ng = group[0].number;

    double kT1 = K_B * (temperature + delta_temperature);
    double kT2 = K_B * (temperature - delta_temperature);
    double dN1 = (double) DIM * (group[0].cpu_size[source] - 1);
    double dN2 = (double) DIM * (group[0].cpu_size[sink] - 1);
    double sigma_1 = dN1 * kT1 * 0.5;
    double sigma_2 = dN2 * kT2 * 0.5;

    // allocate some memory
    std::vector<double> ek(Ng);
    GPU_Vector<double> vcx(Ng), vcy(Ng), vcz(Ng), ke(Ng);

    velocity_verlet
    (
        false,
        time_step,
        group,
        mass,
        force_per_atom,
        position_per_atom,
        velocity_per_atom
     );

    // get center of mass velocity and relative kinetic energy
    find_vc_and_ke
    (
        group,
        mass,
        velocity_per_atom,
        vcx.data(),
        vcy.data(),
        vcz.data(),
        ke.data()
    );

    ke.copy_to_host(ek.data());
    ek[label_1] *= 0.5;
    ek[label_2] *= 0.5;

    // get the re-scaling factors
    double factor_1
        = resamplekin(ek[label_1], sigma_1, dN1, temperature_coupling);
    double factor_2
        = resamplekin(ek[label_2], sigma_2, dN2, temperature_coupling);
    factor_1 = sqrt(factor_1 / ek[label_1]);
    factor_2 = sqrt(factor_2 / ek[label_2]);

    // accumulate the energies transferred from the system to the baths
    energy_transferred[0] += ek[label_1] * (1.0 - factor_1 * factor_1);
    energy_transferred[1] += ek[label_2] * (1.0 - factor_2 * factor_2);

    scale_velocity_local
    (
        factor_1,
        factor_2,
        vcx.data(),
        vcy.data(),
        vcz.data(),
        ke.data(),
        group,
        velocity_per_atom
    );
}


void Ensemble_BDP::compute1
(
    const double time_step,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& potential_per_atom,
    const GPU_Vector<double>& force_per_atom,
    const GPU_Vector<double>& virial_per_atom,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    GPU_Vector<double>& thermo
)
{
    velocity_verlet
    (
        true,
        time_step,
        group,
        mass,
        force_per_atom,
        position_per_atom,
        velocity_per_atom
    );
}


void Ensemble_BDP::compute2
(
    const double time_step,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& potential_per_atom,
    const GPU_Vector<double>& force_per_atom,
    const GPU_Vector<double>& virial_per_atom,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    GPU_Vector<double>& thermo
)
{
    if (type == 4)
    {
        integrate_nvt_bdp_2
        (
            time_step,
            box.get_volume(),
            group,
            mass,
            potential_per_atom,
            force_per_atom,
            virial_per_atom,
            position_per_atom,
            velocity_per_atom,
            thermo
        );
    }
    else
    {
        integrate_heat_bdp_2
        (
            time_step,
            group,
            mass,
            force_per_atom,
            position_per_atom,
            velocity_per_atom
        );
    }
}


// The following functions are from Bussi's website
// https://sites.google.com/site/giovannibussi/Research/algorithms
// I have only added "static" in front of the functions, 
// without any other changes
static double resamplekin(double kk,double sigma, int ndeg, double taut){
/*
  kk:    present value of the kinetic energy of the atoms to be thermalized (in arbitrary units)
  sigma: target average value of the kinetic energy (ndeg k_b T/2)  (in the same units as kk)
  ndeg:  number of degrees of freedom of the atoms to be thermalized
  taut:  relaxation time of the thermostat, in units of 'how often this routine is called'
*/
  double factor,rr;
  if(taut>0.1){
    factor=exp(-1.0/taut);
  } else{
    factor=0.0;
  }
  rr = gasdev();
  return kk + (1.0-factor)* (sigma*(resamplekin_sumnoises(ndeg-1)+rr*rr)/ndeg-kk)
            + 2.0*rr*sqrt(kk*sigma/ndeg*(1.0-factor)*factor);
}


static double resamplekin_sumnoises(int nn){
/*
  returns the sum of n independent gaussian noises squared
   (i.e. equivalent to summing the square of the return values of nn calls to gasdev)
*/
  double rr;
  if(nn==0) {
    return 0.0;
  } else if(nn==1) {
    rr=gasdev();
    return rr*rr;
  } else if(nn%2==0) {
    return 2.0*gamdev(nn/2);
  } else {
    rr=gasdev();
    return 2.0*gamdev((nn-1)/2) + rr*rr;
  }
}


static double gamdev(const int ia)
{
	int j;
	double am,e,s,v1,v2,x,y;

	if (ia < 1) {}; // FATAL ERROR
	if (ia < 6) {
		x=1.0;
		for (j=1;j<=ia;j++) x *= ran1();
		x = -log(x);
	} else {
		do {
			do {
				do {
					v1=ran1();
					v2=2.0*ran1()-1.0;
				} while (v1*v1+v2*v2 > 1.0);
				y=v2/v1;
				am=ia-1;
				s=sqrt(2.0*am+1.0);
				x=s*y+am;
			} while (x <= 0.0);
			e=(1.0+y*y)*exp(am*log(x/am)-s*y);
		} while (ran1() > e);
	}
	return x;
}


static double gasdev()
{
	static int iset=0;
	static double gset;
	double fac,rsq,v1,v2;

	if (iset == 0) {
		do {
			v1=2.0*ran1()-1.0;
			v2=2.0*ran1()-1.0;
			rsq=v1*v1+v2*v2;
		} while (rsq >= 1.0 || rsq == 0.0);
		fac=sqrt(-2.0*log(rsq)/rsq);
		gset=v1*fac;
		iset=1;
		return v2*fac;
	} else {
		iset=0;
		return gset;
	}
}


static double ran1()
{
	const int IA=16807,IM=2147483647,IQ=127773,IR=2836,NTAB=32;
	const int NDIV=(1+(IM-1)/NTAB);
	const double EPS=3.0e-16,AM=1.0/IM,RNMX=(1.0-EPS);
	static int iy=0;
	static int iv[NTAB];
	int j,k;
	double temp;
        static int idum=0; /* ATTENTION: THE SEED IS HARDCODED */

	if (idum <= 0 || !iy) {
		if (-idum < 1) idum=1;
		else idum = -idum;
		for (j=NTAB+7;j>=0;j--) {
			k=idum/IQ;
			idum=IA*(idum-k*IQ)-IR*k;
			if (idum < 0) idum += IM;
			if (j < NTAB) iv[j] = idum;
		}
		iy=iv[0];
	}
	k=idum/IQ;
	idum=IA*(idum-k*IQ)-IR*k;
	if (idum < 0) idum += IM;
	j=iy/NDIV;
	iy=iv[j];
	iv[j] = idum;
	if ((temp=AM*iy) > RNMX) return RNMX;
	else return temp;
}


