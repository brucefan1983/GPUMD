#include <cmath>
#include <numeric>
#include <random>
template <typename T>
T RandomValue(const float& Min, const float& Max)
{
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<T> dis(Min, Max);
  return dis(gen);
};
struct RBEPSAMPLE {
  float _alpha=0.5*0.5;
  float _box[3]={27,27,27};
  int _P=1000;
  bool _RBE_random =true;
  float Compute_S()
  {
    float factor[3] = {0.0f};
    Compute_H(factor);
    float factor_3 = factor[0] * factor[1] * factor[2];
    float S = factor_3 - 1;
    return S;
  }
  void Compute_H(float* H)
  {
    for (int i = 0; i < 3; ++i) {
      const float factor = -(_alpha * _box[i] * _box[i]);
      for (int m = -10; m <= 10; m++) {
        float expx = m * m * factor;
        H[i] += exp(expx);
      }
      H[i] *= sqrt(-(factor) / M_PI);
    }
  }
  float MH_Algorithm(float m, float mu, const float* sigma, int dimension)
  {
    float x_wait = FetchSample_1D(mu, sigma[dimension]);
    float m_wait = float(round(x_wait));
    float Prob = (Distribution_P(m_wait, dimension) / Distribution_P(m, dimension)) *
                 (Distribution_q(m, dimension) / Distribution_q(m_wait, dimension));
    Prob = std::min(Prob, float(1.0));
    if (_RBE_random) {
      float u = RandomValue<float>(0.0, 1.0); // random?
      if (u <= Prob)
        m = m_wait;
      return m;
    } else {
      float u = 0.5;
      if (u <= Prob)
        m = m_wait;
      return m;
    }
  }
  float Distribution_P(const float& x, const int dimension)
  {
    float P_m = exp(-pow(2 * M_PI * x / _box[dimension], 2) / (4 * _alpha));
    float H[3] = {0.0f};
    Compute_H(H);
    P_m = P_m / H[dimension];
    return P_m;
  }
  float Distribution_q(const float& x, const int dimension)
  {
    float q_m;
    if (x == 0) {
      q_m = erf((1.0 / 2) / (sqrt(_alpha * pow(_box[dimension], 2) / pow(M_PI, 2))));
    } else
      q_m = (erf(((1.0 / 2) + abs(x)) / (sqrt(_alpha * pow(_box[dimension], 2) / pow(M_PI, 2)))) -
             erf((abs(x) - (1.0 / 2)) / (sqrt(_alpha * pow(_box[dimension], 2) / pow(M_PI, 2))))) /
            2;
    return q_m;
  }
  float FetchSample_1D(const float& mu, const float& sigma)
  {
    float U1, U2, epsilon;
    epsilon = 1e-6;
    if (_RBE_random) {
      do {
        U1 = RandomValue<float>(0.0, 1.0);
      } while (U1 < epsilon);
      U2 = RandomValue<float>(0.0, 1.0);
      float ChooseSample = sigma * sqrt(-2.0 * log(U1)) * cos(2 * M_PI * U2) + mu;
      return ChooseSample;
    } else {
      U1 = 0.5;
      U2 = 0.5;
      float ChooseSample = sigma * sqrt(-2.0 * log(U1)) * cos(2 * M_PI * U2) + mu;
      return ChooseSample;
    }
  }
  void Fetch_P_Sample()
  {
    float epsilonx = 1e-6; // precision
    const float sigma[3]={
	    sqrt(_alpha*_box[0]*_box[0]/(M_PI*M_PI*2)),
	    sqrt(_alpha*_box[1]*_box[1]/(M_PI*M_PI*2)),
	    sqrt(_alpha*_box[2]*_box[2]/(M_PI*M_PI*2))
	  };
    float X_0[3];
    do {
      X_0[0] = float(round(FetchSample_1D(0, sigma[0])));
      X_0[1] = float(round(FetchSample_1D(0, sigma[1])));
      X_0[2] = float(round(FetchSample_1D(0, sigma[2])));
    } while (abs(X_0[0]) < epsilonx && abs(X_0[1]) < epsilonx && abs(X_0[2]) < epsilonx);
	
	printf("%g, %g, %g;\n", X_0[0], X_0[1], X_0[2]);
    for (int i = 1; i < _P; i++) {
      float X_1[3] = {
        MH_Algorithm(X_0[0], 0, sigma, 0),
        MH_Algorithm(X_0[1], 0, sigma, 1),
        MH_Algorithm(X_0[2], 0, sigma, 2)};
      X_0[0] = X_1[0];
      X_0[1] = X_1[1];
      X_0[2] = X_1[2];
      if (abs(X_1[0]) < epsilonx && abs(X_1[1]) < epsilonx && abs(X_1[2]) < epsilonx) {
        i = i - 1;
        continue;
      }
	  printf("%g, %g, %g;\n", X_0[0], X_0[1], X_0[2]);
    }
  }
};
int main()
{
  RBEPSAMPLE rbe;
  rbe.Fetch_P_Sample();
  printf("S=%g\n",rbe.Compute_S());
  return 0;
}