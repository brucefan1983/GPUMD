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
// The following functions are from Bussi's website
// https://sites.google.com/site/giovannibussi/Research/algorithms
// I have only added "static" in front of the functions,
// without any other changes
// Reference:
[1] G. Bussi et al. J. Chem. Phys. 126, 014101 (2007).
------------------------------------------------------------------------------*/

static double ran1()
{
  const int IA = 16807, IM = 2147483647, IQ = 127773, IR = 2836, NTAB = 32;
  const int NDIV = (1 + (IM - 1) / NTAB);
  const double EPS = 3.0e-16, AM = 1.0 / IM, RNMX = (1.0 - EPS);
  static int iy = 0;
  static int iv[NTAB];
  int j, k;
  double temp;
  static int idum = 0; /* ATTENTION: THE SEED IS HARDCODED */

  if (idum <= 0 || !iy) {
    if (-idum < 1)
      idum = 1;
    else
      idum = -idum;
    for (j = NTAB + 7; j >= 0; j--) {
      k = idum / IQ;
      idum = IA * (idum - k * IQ) - IR * k;
      if (idum < 0)
        idum += IM;
      if (j < NTAB)
        iv[j] = idum;
    }
    iy = iv[0];
  }
  k = idum / IQ;
  idum = IA * (idum - k * IQ) - IR * k;
  if (idum < 0)
    idum += IM;
  j = iy / NDIV;
  iy = iv[j];
  iv[j] = idum;
  if ((temp = AM * iy) > RNMX)
    return RNMX;
  else
    return temp;
}

static double gasdev()
{
  static int iset = 0;
  static double gset;
  double fac, rsq, v1, v2;

  if (iset == 0) {
    do {
      v1 = 2.0 * ran1() - 1.0;
      v2 = 2.0 * ran1() - 1.0;
      rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0 * log(rsq) / rsq);
    gset = v1 * fac;
    iset = 1;
    return v2 * fac;
  } else {
    iset = 0;
    return gset;
  }
}

static double gamdev(const int ia)
{
  int j;
  double am, e, s, v1, v2, x, y;

  if (ia < 1) {
  }; // FATAL ERROR
  if (ia < 6) {
    x = 1.0;
    for (j = 1; j <= ia; j++)
      x *= ran1();
    x = -log(x);
  } else {
    do {
      do {
        do {
          v1 = ran1();
          v2 = 2.0 * ran1() - 1.0;
        } while (v1 * v1 + v2 * v2 > 1.0);
        y = v2 / v1;
        am = ia - 1;
        s = sqrt(2.0 * am + 1.0);
        x = s * y + am;
      } while (x <= 0.0);
      e = (1.0 + y * y) * exp(am * log(x / am) - s * y);
    } while (ran1() > e);
  }
  return x;
}

static double resamplekin_sumnoises(int nn)
{
  /*
    returns the sum of n independent gaussian noises squared
     (i.e. equivalent to summing the square of the return values of nn calls to gasdev)
  */
  double rr;
  if (nn == 0) {
    return 0.0;
  } else if (nn == 1) {
    rr = gasdev();
    return rr * rr;
  } else if (nn % 2 == 0) {
    return 2.0 * gamdev(nn / 2);
  } else {
    rr = gasdev();
    return 2.0 * gamdev((nn - 1) / 2) + rr * rr;
  }
}

static double resamplekin(double kk, double sigma, int ndeg, double taut)
{
  /*
    kk:    present value of the kinetic energy of the atoms to be thermalized (in arbitrary units)
    sigma: target average value of the kinetic energy (ndeg k_b T/2)  (in the same units as kk)
    ndeg:  number of degrees of freedom of the atoms to be thermalized
    taut:  relaxation time of the thermostat, in units of 'how often this routine is called'
  */
  double factor, rr;
  if (taut > 0.1) {
    factor = exp(-1.0 / taut);
  } else {
    factor = 0.0;
  }
  rr = gasdev();
  return kk + (1.0 - factor) * (sigma * (resamplekin_sumnoises(ndeg - 1) + rr * rr) / ndeg - kk) +
         2.0 * rr * sqrt(kk * sigma / ndeg * (1.0 - factor) * factor);
}
