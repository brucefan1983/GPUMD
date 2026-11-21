/*
    Copyright 2026 Ke Li and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Major contributors:
  Ke Li: Initial implementation.
------------------------------------------------------------------------------*/

#include "integrate/integrate.cuh"
#include "measure/parse_utilities.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include "xrd.cuh"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <set>

namespace
{
constexpr double pi = 3.14159265358979323846;
constexpr int block_size = 128;
constexpr int batch_size = 65536;
constexpr int max_mesh_nodes = 10000000;
constexpr int custom_gaussians = 4;
constexpr int builtin_gaussians = 5;
constexpr double builtin_max_s = 6.0;

struct Node {
  int h, k, l;
  double kx, ky, kz; // cycles per angstrom
  double two_theta;
  double s_squared;
  double lp;
};

struct FormFactor {
  double a[builtin_gaussians], b[builtin_gaussians], c;
};

// Waasmaier and Kirfel, Acta Cryst. A51, 416–431 (1995). doi:10.1107/S0108767394013292.
// Neutral atoms H–Cf, with the Si row for silicon.
// Data from the XrayDB archive, released under CC0 1.0:
// https://github.com/xraypy/XrayDB/blob/96911505a105549100d183a6268cb3fdfd6445d3/data_sources/waasmaeir_kirfel.dat
const FormFactor* find_builtin_form_factor(const std::string& symbol)
{
  struct Entry {
    const char* symbol;
    FormFactor factor;
  };
  static constexpr Entry elements[] = {
    {"H",
     {{0.413048, 0.294953, 0.187491, 0.080701, 0.023736},
      {15.569946, 32.398468, 5.711404, 61.889874, 1.334118},
      0.000049}},
    {"He",
     {{0.732354, 0.753896, 0.283819, 0.190003, 0.039139},
      {11.553918, 4.595831, 1.546299, 26.463964, 0.377523},
      0.000487}},
    {"Li",
     {{0.974637, 0.158472, 0.811855, 0.262416, 0.790108},
      {4.334946, 0.342451, 97.102966, 201.363831, 1.409234},
      0.002542}},
    {"Be",
     {{1.533712, 0.638283, 0.601052, 0.106139, 1.118414},
      {42.662079, 0.595420, 99.106499, 0.151340, 1.843093},
      0.002511}},
    {"B",
     {{2.085185, 1.064580, 1.062788, 0.140515, 0.641784},
      {23.494068, 1.137894, 61.238976, 0.114886, 0.399036},
      0.003823}},
    {"C",
     {{2.657506, 1.078079, 1.490909, -4.241070, 0.713791},
      {14.780758, 0.776775, 42.086842, -0.000294, 0.239535},
      4.297983}},
    {"N",
     {{11.893780, 3.277479, 1.858092, 0.858927, 0.912985},
      {0.000158, 10.232723, 30.344690, 0.656065, 0.217287},
      -11.804902}},
    {"O",
     {{2.960427, 2.508818, 0.637853, 0.722838, 1.142756},
      {14.182259, 5.936858, 0.112726, 34.958481, 0.390240},
      0.027014}},
    {"F",
     {{3.511943, 2.772244, 0.678385, 0.915159, 1.089261},
      {10.687859, 4.380466, 0.093982, 27.255203, 0.313066},
      0.032557}},
    {"Ne",
     {{4.183749, 2.905726, 0.520513, 1.135641, 1.228065},
      {8.175457, 3.252536, 0.063295, 21.813910, 0.224952},
      0.025576}},
    {"Na",
     {{4.910127, 3.081783, 1.262067, 1.098938, 0.560991},
      {3.281434, 9.119178, 0.102763, 132.013947, 0.405878},
      0.079712}},
    {"Mg",
     {{4.708971, 1.194814, 1.558157, 1.170413, 3.239403},
      {4.875207, 108.506081, 0.111516, 48.292408, 1.928171},
      0.126842}},
    {"Al",
     {{4.730796, 2.313951, 1.541980, 1.117564, 3.154754},
      {3.628931, 43.051167, 0.095960, 108.932388, 1.555918},
      0.139509}},
    {"Si",
     {{5.275329, 3.191038, 1.511514, 1.356849, 2.519114},
      {2.631338, 33.730728, 0.081119, 86.288643, 1.170087},
      0.145073}},
    {"P",
     {{1.950541, 4.146930, 1.494560, 1.522042, 5.729711},
      {0.908139, 27.044952, 0.071280, 67.520187, 1.981173},
      0.155233}},
    {"S",
     {{6.372157, 5.154568, 1.473732, 1.635073, 1.209372},
      {1.514347, 22.092527, 0.061373, 55.445175, 0.646925},
      0.154722}},
    {"Cl",
     {{1.446071, 6.870609, 6.151801, 1.750347, 0.634168},
      {0.052357, 1.193165, 18.343416, 46.398396, 0.401005},
      0.146773}},
    {"Ar",
     {{7.188004, 6.638454, 0.454180, 1.929593, 1.523654},
      {0.956221, 15.339877, 15.339862, 39.043823, 0.062409},
      0.265954}},
    {"K",
     {{8.163991, 7.146945, 1.070140, 0.877316, 1.486434},
      {12.816323, 0.808945, 210.327011, 39.597652, 0.052821},
      0.253614}},
    {"Ca",
     {{8.593655, 1.477324, 1.436254, 1.182839, 7.113258},
      {10.460644, 0.041891, 81.390381, 169.847839, 0.688098},
      0.196255}},
    {"Sc",
     {{1.476566, 1.487278, 1.600187, 9.177463, 7.099750},
      {53.131023, 0.035325, 137.319489, 9.098031, 0.602102},
      0.157765}},
    {"Ti",
     {{9.818524, 1.522646, 1.703101, 1.768774, 7.082555},
      {8.001879, 0.029763, 39.885422, 120.157997, 0.532405},
      0.102473}},
    {"V",
     {{10.473575, 1.547881, 1.986381, 1.865616, 7.056250},
      {7.081940, 0.026040, 31.909672, 108.022842, 0.474882},
      0.067744}},
    {"Cr",
     {{11.007069, 1.555477, 2.985293, 1.347855, 7.034779},
      {6.366281, 0.023987, 23.244839, 105.774498, 0.429369},
      0.065510}},
    {"Mn",
     {{11.709542, 1.733414, 2.673141, 2.023368, 7.003180},
      {5.597120, 0.017800, 21.788420, 89.517914, 0.383054},
      -0.147293}},
    {"Fe",
     {{12.311098, 1.876623, 3.066177, 2.070451, 6.975185},
      {5.009415, 0.014461, 18.743040, 82.767876, 0.346506},
      -0.304931}},
    {"Co",
     {{12.914510, 2.481908, 3.466894, 2.106351, 6.960892},
      {4.507138, 0.009126, 16.438129, 76.987320, 0.314418},
      -0.936572}},
    {"Ni",
     {{13.521865, 6.947285, 3.866028, 2.135900, 4.284731},
      {4.077277, 0.286763, 14.622634, 71.966080, 0.004437},
      -2.762697}},
    {"Cu",
     {{14.014192, 4.784577, 5.056806, 1.457971, 6.932996},
      {3.738280, 0.003744, 13.034982, 72.554794, 0.265666},
      -3.254477}},
    {"Zn",
     {{14.741002, 6.907748, 4.642337, 2.191766, 38.424042},
      {3.388232, 0.243315, 11.903689, 63.312130, 0.000397},
      -36.915829}},
    {"Ga",
     {{15.758946, 6.841123, 4.121016, 2.714681, 2.395246},
      {3.121754, 0.226057, 12.482196, 66.203621, 0.007238},
      -0.847395}},
    {"Ge",
     {{16.540613, 1.567900, 3.727829, 3.345098, 6.785079},
      {2.866618, 0.012198, 13.432163, 58.866047, 0.210974},
      0.018726}},
    {"As",
     {{17.025642, 4.503441, 3.715904, 3.937200, 6.790175},
      {2.597739, 0.003012, 14.272119, 50.437996, 0.193015},
      -2.984117}},
    {"Se",
     {{17.354071, 4.653248, 4.259489, 4.136455, 6.749163},
      {2.349787, 0.002550, 15.579460, 45.181202, 0.177432},
      -3.160982}},
    {"Br",
     {{17.550570, 5.411882, 3.937180, 3.880645, 6.707793},
      {2.119226, 16.557184, 0.002481, 42.164009, 0.162121},
      -2.492088}},
    {"Kr",
     {{17.655279, 6.848105, 4.171004, 3.446760, 6.685200},
      {1.908231, 16.606236, 0.001598, 39.917473, 0.146896},
      -2.810592}},
    {"Rb",
     {{8.123134, 2.138042, 6.761702, 1.156051, 17.679546},
      {15.142385, 33.542667, 0.129372, 224.132507, 1.713368},
      1.139548}},
    {"Sr",
     {{17.730219, 9.795867, 6.099763, 2.620025, 0.600053},
      {1.563060, 14.310868, 0.120574, 135.771317, 0.120574},
      1.140251}},
    {"Y",
     {{17.792040, 10.253252, 5.714949, 3.170516, 0.918251},
      {1.429691, 13.132816, 0.112173, 108.197029, 0.112173},
      1.131787}},
    {"Zr",
     {{17.859772, 10.911038, 5.821115, 3.512513, 0.746965},
      {1.310692, 12.319285, 0.104353, 91.777542, 0.104353},
      1.124859}},
    {"Nb",
     {{17.958399, 12.063054, 5.007015, 3.287667, 1.531019},
      {1.211590, 12.246687, 0.098615, 75.011948, 0.098615},
      1.123452}},
    {"Mo",
     {{6.236218, 17.987711, 12.973127, 3.451426, 0.210899},
      {0.090780, 1.108310, 11.468720, 66.684151, 0.090780},
      1.108770}},
    {"Tc",
     {{17.840963, 3.428236, 1.373012, 12.947364, 6.335469},
      {1.005729, 41.901382, 119.320541, 9.781542, 0.083391},
      1.074784}},
    {"Ru",
     {{6.271624, 17.906738, 14.123269, 3.746008, 0.908235},
      {0.077040, 0.928222, 9.555345, 35.860680, 123.552246},
      1.043992}},
    {"Rh",
     {{6.216648, 17.919739, 3.854252, 0.840326, 15.173498},
      {0.070789, 0.856121, 33.889484, 121.686691, 9.029517},
      0.995452}},
    {"Pd",
     {{6.121511, 4.784063, 16.631683, 4.318258, 13.246773},
      {0.062549, 0.784031, 8.751391, 34.489983, 0.784031},
      0.883099}},
    {"Ag",
     {{6.073874, 17.155437, 4.173344, 0.852238, 17.988686},
      {0.055333, 7.896512, 28.443739, 110.376106, 0.716809},
      0.756603}},
    {"Cd",
     {{6.080986, 18.019468, 4.018197, 1.303510, 17.974669},
      {0.048990, 7.273646, 29.119284, 95.831207, 0.661231},
      0.603504}},
    {"In",
     {{6.196477, 18.816183, 4.050479, 1.638929, 17.962912},
      {0.042072, 6.695665, 31.009790, 103.284348, 0.610714},
      0.333097}},
    {"Sn",
     {{19.325171, 6.281571, 4.498866, 1.856934, 17.917318},
      {6.118104, 0.036915, 32.529045, 95.037186, 0.565651},
      0.119024}},
    {"Sb",
     {{5.394956, 6.549570, 19.650681, 1.827820, 17.867832},
      {33.326523, 0.030974, 5.564929, 87.130966, 0.523992},
      -0.290506}},
    {"Te",
     {{6.660302, 6.940756, 19.847015, 1.557175, 17.802427},
      {33.031654, 0.025750, 5.065547, 84.101616, 0.487660},
      -0.806668}},
    {"I",
     {{19.884502, 6.736593, 8.110516, 1.170953, 17.548716},
      {4.628591, 0.027754, 31.849096, 84.406387, 0.463550},
      -0.448811}},
    {"Xe",
     {{19.978920, 11.774945, 9.332182, 1.244749, 17.737501},
      {4.143356, 0.010142, 28.796200, 75.280685, 0.413616},
      -6.065902}},
    {"Cs",
     {{17.418674, 8.314444, 10.323193, 1.383834, 19.876251},
      {0.399828, 0.016872, 25.605827, 233.339676, 3.826915},
      -2.322802}},
    {"Ba",
     {{19.747343, 17.368477, 10.465718, 2.592602, 11.003653},
      {3.481823, 0.371224, 21.226641, 173.834274, 0.010719},
      -5.183497}},
    {"La",
     {{19.966019, 27.329655, 11.018425, 3.086696, 17.335455},
      {3.197408, 0.003446, 19.955492, 141.381973, 0.341817},
      -21.745489}},
    {"Ce",
     {{17.355122, 43.988499, 20.546650, 3.130670, 11.353665},
      {0.328369, 0.002047, 3.088196, 134.907654, 18.832960},
      -38.386017}},
    {"Pr",
     {{21.551311, 17.161730, 11.903859, 2.679103, 9.564197},
      {2.995675, 0.312491, 17.716705, 152.192825, 0.010468},
      -3.871068}},
    {"Nd",
     {{17.331244, 62.783924, 12.160097, 2.663483, 22.239950},
      {0.300269, 0.001320, 17.026001, 148.748993, 2.910268},
      -57.189842}},
    {"Pm",
     {{17.286388, 51.560162, 12.478557, 2.675515, 22.960947},
      {0.286620, 0.001550, 16.223755, 143.984512, 2.796480},
      -45.973682}},
    {"Sm",
     {{23.700363, 23.072214, 12.777782, 2.684217, 17.204367},
      {2.689539, 0.003491, 15.495437, 139.862473, 0.274536},
      -17.452166}},
    {"Eu",
     {{17.186195, 37.156837, 13.103387, 2.707246, 24.419271},
      {0.261678, 0.001995, 14.787360, 134.816299, 2.581883},
      -31.586687}},
    {"Gd",
     {{24.898117, 17.104952, 13.222581, 3.266152, 48.995213},
      {2.435028, 0.246961, 13.996325, 110.863091, 0.001383},
      -43.505684}},
    {"Tb",
     {{25.910013, 32.344139, 13.765117, 2.751404, 17.064405},
      {2.373912, 0.002034, 13.481969, 125.836510, 0.236916},
      -26.851971}},
    {"Dy",
     {{26.671785, 88.687576, 14.065445, 2.768497, 17.067781},
      {2.282593, 0.000665, 12.920230, 121.937187, 0.225531},
      -83.279831}},
    {"Ho",
     {{27.150190, 16.999819, 14.059334, 3.386979, 46.546471},
      {2.169660, 0.215414, 12.213148, 100.506783, 0.001211},
      -41.165253}},
    {"Er",
     {{28.174887, 82.493271, 14.624002, 2.802756, 17.018515},
      {2.120995, 0.000640, 11.915256, 114.529938, 0.207519},
      -77.135223}},
    {"Tm",
     {{28.925894, 76.173798, 14.904704, 2.814812, 16.998117},
      {2.046203, 0.000656, 11.465375, 111.411980, 0.199376},
      -70.839813}},
    {"Yb",
     {{29.676760, 65.624069, 15.160854, 2.830288, 16.997850},
      {1.977630, 0.000720, 11.044622, 108.139153, 0.192110},
      -60.313812}},
    {"Lu",
     {{30.122866, 15.099346, 56.314899, 3.540980, 16.943729},
      {1.883090, 10.342764, 0.000780, 89.559250, 0.183849},
      -51.049416}},
    {"Hf",
     {{30.617033, 15.145351, 54.933548, 4.096253, 16.896156},
      {1.795613, 9.934469, 0.000739, 76.189705, 0.175914},
      -49.719837}},
    {"Ta",
     {{31.066359, 15.341823, 49.278297, 4.577665, 16.828321},
      {1.708732, 9.618455, 0.000760, 66.346199, 0.168002},
      -44.119026}},
    {"W",
     {{31.507900, 15.682498, 37.960129, 4.885509, 16.792112},
      {1.629485, 9.446448, 0.000898, 59.980675, 0.160798},
      -32.864574}},
    {"Re",
     {{31.888456, 16.117104, 42.390297, 5.211669, 16.767591},
      {1.549238, 9.233474, 0.000689, 54.516373, 0.152815},
      -37.412682}},
    {"Os",
     {{32.210297, 16.678440, 48.559906, 5.455839, 16.735533},
      {1.473531, 9.049695, 0.000519, 50.210201, 0.145771},
      -43.677956}},
    {"Ir",
     {{32.004436, 1.975454, 17.070105, 15.939454, 5.990003},
      {1.353767, 81.014175, 0.128093, 7.661196, 26.659403},
      4.018893}},
    {"Pt",
     {{31.273891, 18.445440, 17.063745, 5.555933, 1.575270},
      {1.316992, 8.797154, 0.124741, 40.177994, 1.316997},
      4.050394}},
    {"Au",
     {{16.777390, 19.317156, 32.979683, 5.595453, 10.576854},
      {0.122737, 8.621570, 1.256902, 38.008820, 0.000601},
      -6.279078}},
    {"Hg",
     {{16.839890, 20.023823, 28.428564, 5.881564, 4.714706},
      {0.115905, 8.256927, 1.195250, 39.247227, 1.195250},
      4.076478}},
    {"Tl",
     {{16.630795, 19.386616, 32.808571, 1.747191, 6.356862},
      {0.110704, 7.181401, 1.119730, 90.660263, 26.014978},
      4.066939}},
    {"Pb",
     {{16.419567, 32.738590, 6.530247, 2.342742, 19.916475},
      {0.105499, 1.055049, 25.025890, 80.906593, 6.664449},
      4.049824}},
    {"Bi",
     {{16.282274, 32.725136, 6.678302, 2.694750, 20.576559},
      {0.101180, 1.002287, 25.714146, 77.057549, 6.291882},
      4.040914}},
    {"Po",
     {{16.289164, 32.807171, 21.095163, 2.505901, 7.254589},
      {0.098121, 0.966265, 6.046622, 76.598068, 28.096128},
      4.046556}},
    {"At",
     {{16.011461, 32.615547, 8.113899, 2.884082, 21.377867},
      {0.092639, 0.904416, 26.543257, 68.372963, 5.499512},
      3.995684}},
    {"Rn",
     {{16.070229, 32.641106, 21.489658, 2.299218, 9.480184},
      {0.090437, 0.876409, 5.239687, 69.188477, 27.632641},
      4.020977}},
    {"Fr",
     {{16.007385, 32.663830, 21.594351, 1.598497, 11.121192},
      {0.087031, 0.840187, 4.954467, 199.805801, 26.905106},
      4.003472}},
    {"Ra",
     {{32.563690, 21.396671, 11.298093, 2.834688, 15.914965},
      {0.801980, 4.590666, 22.758972, 160.404388, 0.083544},
      3.981773}},
    {"Ac",
     {{15.914053, 32.535042, 21.553976, 11.433394, 3.612409},
      {0.080511, 0.770669, 4.352206, 21.381622, 130.500748},
      3.939212}},
    {"Th",
     {{15.784024, 32.454899, 21.849222, 4.239077, 11.736191},
      {0.077067, 0.735137, 4.097976, 109.464111, 20.512138},
      3.922533}},
    {"Pa",
     {{32.740208, 21.973675, 12.957398, 3.683832, 15.744058},
      {0.709545, 4.050881, 19.231543, 117.255005, 0.074040},
      3.886066}},
    {"U",
     {{15.679275, 32.824306, 13.660459, 3.687261, 22.279434},
      {0.071206, 0.681177, 18.236156, 112.500038, 3.930325},
      3.854444}},
    {"Np",
     {{32.999901, 22.638077, 14.219973, 3.672950, 15.683245},
      {0.657086, 3.854918, 17.435474, 109.464485, 0.068033},
      3.769391}},
    {"Pu",
     {{33.281178, 23.148544, 15.153755, 3.031492, 15.704215},
      {0.634999, 3.856168, 16.849735, 121.292038, 0.064857},
      3.664200}},
    {"Am",
     {{33.435162, 23.657259, 15.576339, 3.027023, 15.746100},
      {0.612785, 3.792942, 16.195778, 117.757004, 0.061755},
      3.541160}},
    {"Cm",
     {{15.804837, 33.480801, 24.150198, 3.655563, 15.499866},
      {0.058619, 0.590160, 3.674720, 100.736191, 15.408296},
      3.390840}},
    {"Bk",
     {{15.889072, 33.625286, 24.710381, 3.707139, 15.839268},
      {0.055503, 0.569571, 3.615472, 97.694786, 14.754303},
      3.213169}},
    {"Cf",
     {{33.794075, 25.467693, 16.048487, 3.657525, 16.008982},
      {0.550447, 3.581973, 14.357388, 96.064972, 0.052450},
      3.005326}},
  };
  for (const auto& element : elements) {
    if (symbol == element.symbol) {
      return &element.factor;
    }
  }
  return nullptr;
}

__global__ void xrd_intensities(
  const Node* nodes,
  const int number_atoms,
  const int number_selected,
  const int number_factors,
  const int number_gaussians,
  const int* selected,
  const int* types,
  const FormFactor* factors,
  const double* positions,
  double* intensities)
{
  extern __shared__ double weights[];
  __shared__ double real_parts[block_size];
  __shared__ double imag_parts[block_size];
  const int lane = threadIdx.x;
  const auto node = nodes[blockIdx.x];

  for (int t = lane; t < number_factors; t += block_size) {
    const auto factor = factors[t];
    double weight = factor.c;
    for (int j = 0; j < number_gaussians; ++j) {
      weight += factor.a[j] * exp(-factor.b[j] * node.s_squared);
    }
    weights[t] = weight;
  }
  __syncthreads();

  double real = 0.0;
  double imag = 0.0;
  for (int i = lane; i < number_selected; i += block_size) {
    const int j = selected ? selected[i] : i;
    const double k_dot_r = node.kx * positions[j] +
                           node.ky * positions[j + static_cast<size_t>(number_atoms)] +
                           node.kz * positions[j + 2 * static_cast<size_t>(number_atoms)];
    double sine, cosine;
    sincos(2.0 * pi * k_dot_r, &sine, &cosine);
    const double weight = weights[types[j]];
    real += weight * cosine;
    imag += weight * sine;
  }
  real_parts[lane] = real;
  imag_parts[lane] = imag;
  __syncthreads();
  for (int stride = block_size / 2; stride > 0; stride /= 2) {
    if (lane < stride) {
      real_parts[lane] += real_parts[lane + stride];
      imag_parts[lane] += imag_parts[lane + stride];
    }
    __syncthreads();
  }
  if (lane == 0) {
    intensities[blockIdx.x] =
      node.lp * (real_parts[0] * real_parts[0] + imag_parts[0] * imag_parts[0]) / number_selected;
  }
}

double positive_number(const std::string& value)
{
  double result;
  if (!is_valid_real(value, &result) || !std::isfinite(result) || result <= 0.0) {
    PRINT_INPUT_ERROR("compute_xrd requires a finite, positive wavelength and mesh values.");
  }
  return result;
}

double finite_coefficient(const std::string& value, const std::string& record)
{
  double result;
  if (!is_valid_real(value, &result) || !std::isfinite(result)) {
    const std::string error = record + "Invalid coefficient: " + value + ".";
    PRINT_INPUT_ERROR(error.c_str());
  }
  return result;
}
} // namespace

struct XRD::Storage {
  std::map<std::string, FormFactor> input_factors;
  std::vector<Node> nodes;
  std::vector<double> intensities;
  std::vector<FormFactor> cpu_factors;
  std::vector<bool> factor_initialized;
  GPU_Vector<FormFactor> factors;
  GPU_Vector<Node> gpu_nodes;
  GPU_Vector<double> gpu_intensities;
};

XRD::XRD(const std::vector<std::string>& tokens, const std::vector<Group>& groups)
  : storage_(std::make_unique<Storage>())
{
  action_name = "compute_xrd";
  if (tokens.size() < 3 || !is_valid_int(tokens[1], &interval_) || interval_ <= 0) {
    PRINT_INPUT_ERROR("Expected compute_xrd <positive interval> <wavelength> [options].");
  }
  wavelength_ = positive_number(tokens[2]);

  std::set<std::string> options;
  const int count = tokens.size();
  for (int i = 3; i < count; ++i) {
    const auto& key = tokens[i];
    if (!options.insert(key).second) {
      PRINT_INPUT_ERROR("Repeated compute_xrd option.");
    }
    if (key == "group") {
      parse_group(tokens, false, groups, i, grouping_method_, group_id_);
    } else if ((key == "spacing" || key == "scale") && i + 3 < count) {
      manual_ = key == "spacing";
      for (auto& value : mesh_) {
        value = positive_number(tokens[++i]);
      }
    } else if (key == "two_theta" && i + 2 < count) {
      if (
        !is_valid_real(tokens[++i], &angle_min_) || !is_valid_real(tokens[++i], &angle_max_) ||
        !(angle_min_ >= 0 && angle_min_ < angle_max_ && angle_max_ <= 180)) {
        PRINT_INPUT_ERROR("compute_xrd requires 0 <= min 2theta < max 2theta <= 180 in degrees.");
      }
    } else if (key == "lp" && i + 1 < count) {
      const auto& value = tokens[++i];
      if (value != "0" && value != "1") {
        PRINT_INPUT_ERROR("compute_xrd lp expects 0 or 1.");
      }
      use_lp_ = value == "1";
    } else if (key == "form_factor" && i + 1 < count) {
      form_factor_filename_ = tokens[++i];
    } else if (key == "file" && i + 1 < count) {
      filename_ = tokens[++i];
    } else {
      const std::string error = "Invalid or incomplete compute_xrd option: " + key;
      PRINT_INPUT_ERROR(error.c_str());
    }
  }
  if (options.count("scale") && options.count("spacing")) {
    PRINT_INPUT_ERROR("Specify one of compute_xrd scale or spacing.");
  }
  if (form_factor_filename_ == filename_) {
    PRINT_INPUT_ERROR("compute_xrd requires different form factor and output filenames.");
  }
  if (form_factor_filename_.empty()) {
    const double max_s = std::sin(angle_max_ * pi / 360.0) / wavelength_;
    if (max_s > builtin_max_s) {
      PRINT_INPUT_ERROR(
        "compute_xrd built-in form factors require sin(theta)/wavelength <= 6 A^-1. "
        "Adjust two_theta or specify form_factor <filename>.");
    }
  } else {
    read_form_factors();
  }
  printf(
    "Compute X-ray diffraction every %d steps at wavelength %.8g A.\n", interval_, wavelength_);
}

XRD::~XRD() = default;

void XRD::read_form_factors()
{
  std::ifstream input(form_factor_filename_);
  const std::string source = "compute_xrd form factor file " + form_factor_filename_;
  if (!input) {
    const std::string error = "Cannot open " + source + ".";
    PRINT_INPUT_ERROR(error.c_str());
  }
  std::string line;
  for (size_t line_number = 1; std::getline(input, line); ++line_number) {
    const auto tokens = get_tokens_without_comments(line);
    if (tokens.empty()) {
      continue;
    }
    const std::string record = source + ", line " + std::to_string(line_number) + ": ";
    if (tokens.size() != 2 * custom_gaussians + 2) {
      const std::string error = record + "Expected an element and nine coefficients.";
      PRINT_INPUT_ERROR(error.c_str());
    }
    FormFactor factor{};
    for (int j = 0; j < custom_gaussians; ++j) {
      factor.a[j] = finite_coefficient(tokens[1 + 2 * j], record);
      factor.b[j] = finite_coefficient(tokens[2 + 2 * j], record);
    }
    factor.c = finite_coefficient(tokens.back(), record);
    if (!storage_->input_factors.emplace(tokens[0], factor).second) {
      const std::string error = record + "Duplicate element " + tokens[0] + ".";
      PRINT_INPUT_ERROR(error.c_str());
    }
  }
  if (input.bad() || !input.eof()) {
    const std::string error = "Failed to read " + source + ".";
    PRINT_INPUT_ERROR(error.c_str());
  }
  if (storage_->input_factors.empty()) {
    const std::string error = "No coefficients in " + source + ".";
    PRINT_INPUT_ERROR(error.c_str());
  }
}

void XRD::pre_run(
  int number_of_steps, double, Integrate&, std::vector<Group>& groups, Atom& atom, Box& box, Force&)
{
  if (interval_ > number_of_steps) {
    PRINT_INPUT_ERROR("compute_xrd interval exceeds the number of run steps.");
  }

  number_selected_ = atom.number_of_atoms;
  if (grouping_method_ >= 0) {
    const auto& group = groups[grouping_method_];
    number_selected_ = group.cpu_size[group_id_];
  }
  if (number_selected_ <= 0) {
    PRINT_INPUT_ERROR("compute_xrd requires a nonempty atom selection.");
  }

  storage_->cpu_factors.resize(atom.cpu_type_size.size());
  storage_->factor_initialized.assign(storage_->cpu_factors.size(), false);
  storage_->factors.resize(storage_->cpu_factors.size());
  update_factors(atom, groups);

  update_mesh(box);
  output_.open(filename_, std::ios::app);
  if (!output_) {
    const std::string error = "Cannot open compute_xrd output: " + filename_;
    PRINT_INPUT_ERROR(error.c_str());
  }
  output_ << std::setprecision(17);
}

void XRD::update_factors(const Atom& atom, const std::vector<Group>& groups)
{
  const int* indices = nullptr;
  if (grouping_method_ >= 0) {
    const auto& group = groups[grouping_method_];
    indices = group.cpu_contents.data() + group.cpu_size_sum[group_id_];
  }
  // MC can introduce an atom type not previously present in the sampled group.
  // The kernel uses the current device types, including accepted MC swaps.
  bool changed = false;
  for (int i = 0; i < number_selected_; ++i) {
    const int j = indices ? indices[i] : i;
    const int type = atom.cpu_type[j];
    if (!storage_->factor_initialized[type]) {
      const auto& element = atom.cpu_atom_symbol[j];
      const FormFactor* factor = nullptr;
      if (form_factor_filename_.empty()) {
        factor = find_builtin_form_factor(element);
      } else {
        const auto entry = storage_->input_factors.find(element);
        if (entry != storage_->input_factors.end()) {
          factor = &entry->second;
        }
      }
      if (!factor) {
        const std::string error = form_factor_filename_.empty()
                                    ? "No built-in compute_xrd form factor for " + element +
                                        ". Specify form_factor <filename>."
                                    : "Missing element " + element +
                                        " in compute_xrd form factor file " +
                                        form_factor_filename_ + ".";
        PRINT_INPUT_ERROR(error.c_str());
      }
      storage_->cpu_factors[type] = *factor;
      storage_->factor_initialized[type] = true;
      changed = true;
    }
  }
  if (changed) {
    storage_->factors.copy_from_host(storage_->cpu_factors.data());
  }
}

void XRD::update_mesh(const Box& box)
{
  // The reciprocal mesh and phase convention follow the kinematic diffraction equations
  // in Coleman et al., doi:10.1088/0965-0393/21/5/055020.
  // At pre_run, the cached box fast-path flag need not have been set yet.
  // Read the current matrix here and on each sampled step.
  for (int i : {1, 2, 3, 5, 6, 7}) {
    if (box.cpu_h[i] != 0.0) {
      PRINT_INPUT_ERROR("compute_xrd requires an axis-aligned orthogonal cell.");
    }
  }
  auto spacing = mesh_;
  if (!manual_) {
    const int periodic[] = {box.pbc_x, box.pbc_y, box.pbc_z};
    int periodic_count = 0;
    double inverse_sum = 0.0;
    for (int i = 0; i < 3; ++i) {
      if (periodic[i]) {
        const double length = box.cpu_h[4 * i];
        if (!std::isfinite(length) || length <= 0.0) {
          PRINT_INPUT_ERROR("Invalid periodic cell length in compute_xrd.");
        }
        inverse_sum += 1.0 / length;
        ++periodic_count;
      }
    }
    if (periodic_count == 0) {
      PRINT_INPUT_ERROR("Nonperiodic compute_xrd requires explicit spacing.");
    }
    for (int i = 0; i < 3; ++i) {
      spacing[i] *= periodic[i] ? 1.0 / box.cpu_h[4 * i] : inverse_sum / periodic_count;
    }
  }
  const double radius = 2.0 * std::sin(angle_max_ * pi / 360.0) / wavelength_;
  // Allow for roundoff in the integer bounds at the surface of the sphere.
  const double bound = radius * (1.0 + 16.0 * std::numeric_limits<double>::epsilon());
  const double radius_squared = bound * bound;
  if (!std::isfinite(radius_squared) || radius_squared == 0.0) {
    PRINT_INPUT_ERROR("compute_xrd wavelength gives an unrepresentable reciprocal radius.");
  }
  std::array<int, 3> extent;
  for (int i = 0; i < 3; ++i) {
    const double limit = bound / spacing[i];
    if (
      !std::isfinite(spacing[i]) || !std::isfinite(limit) || spacing[i] <= 0.0 ||
      limit > max_mesh_nodes) {
      PRINT_INPUT_ERROR(
        "compute_xrd mesh is too dense. Increase spacing or narrow the angle range.");
    }
    extent[i] = static_cast<int>(limit);
  }
  if (spacing == spacing_) {
    return;
  }
  spacing_ = spacing;

  storage_->nodes.clear();
  size_t candidate_count = 0;
  for (int h = -extent[0]; h <= extent[0]; ++h) {
    const double x = h * spacing[0];
    const double yz_squared = std::max(0.0, radius_squared - x * x);
    const int kmax = std::min(extent[1], static_cast<int>(std::sqrt(yz_squared) / spacing[1]));
    for (int k = -kmax; k <= kmax; ++k) {
      const double y = k * spacing[1];
      const double z_squared = std::max(0.0, yz_squared - y * y);
      const int lmax = std::min(extent[2], static_cast<int>(std::sqrt(z_squared) / spacing[2]));
      candidate_count += 2 * lmax + 1;
      if (candidate_count > max_mesh_nodes) {
        PRINT_INPUT_ERROR("compute_xrd mesh exceeds 10000000 candidate nodes. Increase spacing.");
      }
      for (int l = -lmax; l <= lmax; ++l) {
        const double z = l * spacing[2];
        const double s_squared = (x * x + y * y + z * z) * 0.25;
        const double sine = wavelength_ * std::sqrt(s_squared);
        if (!(sine > 0.0 && sine < 1.0)) {
          continue; // omit forward scattering and the backscattering singularity
        }
        const double two_theta = 2.0 * std::asin(sine) * 180.0 / pi;
        if (two_theta < angle_min_ - 1.0e-12 || two_theta > angle_max_ + 1.0e-12) {
          continue;
        }
        double lp = 1.0;
        if (use_lp_) {
          const double cos_two_theta = 1.0 - 2.0 * sine * sine;
          lp = (1.0 + cos_two_theta * cos_two_theta) / (std::sqrt(1.0 - sine * sine) * sine * sine);
        }
        storage_->nodes.push_back({h, k, l, x, y, z, two_theta, s_squared, lp});
      }
    }
  }
  if (storage_->nodes.empty()) {
    return;
  }
  const size_t buffer_size = std::min(storage_->nodes.size(), static_cast<size_t>(batch_size));
  if (storage_->intensities.size() < buffer_size) {
    storage_->intensities.resize(buffer_size);
    storage_->gpu_nodes.resize(buffer_size);
    storage_->gpu_intensities.resize(buffer_size);
  }
  if (storage_->nodes.size() <= batch_size) {
    storage_->gpu_nodes.copy_from_host(storage_->nodes.data(), storage_->nodes.size());
  }
}

void XRD::end_of_step(
  int,
  int step,
  int,
  int,
  double,
  double,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& groups,
  GPU_Vector<double>&,
  Atom& atom,
  Force&)
{
  if ((step + 1) % interval_) {
    return;
  }
  update_mesh(box);
  update_factors(atom, groups);
  output_ << "# step " << step + 1 << " nodes " << storage_->nodes.size() << " atoms "
          << number_selected_ << " wavelength_A " << wavelength_ << " lp " << use_lp_
          << " spacing_A^-1 " << spacing_[0] << ' ' << spacing_[1] << ' ' << spacing_[2] << '\n'
          << "# h k l two_theta_deg intensity\n";
  const auto& positions =
    is_pimd(integrate.get_type()) ? atom.position_beads[0] : atom.position_per_atom;
  const int* selected = nullptr;
  if (grouping_method_ >= 0) {
    const auto& group = groups[grouping_method_];
    selected = group.contents.data() + group.cpu_size_sum[group_id_];
  }
  const int number_gaussians = form_factor_filename_.empty() ? builtin_gaussians : custom_gaussians;
  for (size_t start = 0; start < storage_->nodes.size(); start += batch_size) {
    const int count = std::min(storage_->nodes.size() - start, static_cast<size_t>(batch_size));
    if (storage_->nodes.size() > batch_size) {
      storage_->gpu_nodes.copy_from_host(storage_->nodes.data() + start, count);
    }
    xrd_intensities<<<count, block_size, storage_->cpu_factors.size() * sizeof(double)>>>(
      storage_->gpu_nodes.data(),
      atom.number_of_atoms,
      number_selected_,
      storage_->cpu_factors.size(),
      number_gaussians,
      selected,
      atom.type.data(),
      storage_->factors.data(),
      positions.data(),
      storage_->gpu_intensities.data());
    GPU_CHECK_KERNEL
    storage_->gpu_intensities.copy_to_host(storage_->intensities.data(), count);
    for (int i = 0; i < count; ++i) {
      const auto& node = storage_->nodes[start + i];
      if (!std::isfinite(storage_->intensities[i])) {
        PRINT_INPUT_ERROR(
          "compute_xrd produced a nonfinite intensity. Check coordinates and wavelength.");
      }
      output_ << node.h << ' ' << node.k << ' ' << node.l << ' ' << node.two_theta << ' '
              << storage_->intensities[i] << '\n';
    }
  }
  output_.flush();
  if (!output_) {
    PRINT_INPUT_ERROR("Failed to write compute_xrd output.");
  }
}
