#include <torch/extension.h>

#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>

/*
 * BEGIN EXTERNAL IMPLEMENTATION
 *
 * Taken from: https://github.com/thomasluu/plog
 *
 * Changes:
 *   * If input is close to zero -> return zero! We use the same break
 *     condition as the loop below.
 *
 * __device__ code needs to be in the same compile unit as __global__ function
 * calling it -> for ease of use, we just copy the code for now.
 * --obreitwi, * 24-01-20 13:52:12
 */
__host__ __device__ double plog(double x)
{
  if (fabs(x) < 1.4901161193847656e-8) {
    return 0.0;
  }

  double w0,  w1;
  if (x > 0.0 ) {
    w0 = log( 1.2 * x / log(2.4 * x / log1p(2.4 * x)));
  } else {
    double v  = 1.4142135623730950488 * sqrt(1 + 2.7182818284590452354 * x);
    double N2  = 10.242640687119285146 + 1.9797586132081854940 * v;
    double N1  = 0.29289321881345247560 * (1.4142135623730950488 + N2);
    w0 = -1.0  + v * (N2 + v) / (N2 + v + N1 * v);
  }

  while (true ) {
    double e  = exp(w0);
    double f  = w0 * e - x;
    w1 = w0 +  ((f+f) * (1.0 + w0)) / (f * (2.0 + w0) - (e+e) * (1.0 + w0) * (1.0 + w0));
    if (fabs( w0 / w1 - 1.0) < 1.4901161193847656e-8) {
      break;
    }
    w0 = w1;
  }

  return w1;
}

__host__ __device__ float plog(float x)
{
  if (fabs(x) < 0.00034526698300124390840f) {
    return 0.0f;
  }

  float w0, w1;
  if (x > 0.0f) {
    w0 = log(1.2f * x / log(2.4f * x / log1p(2.4f * x)));
  } else {
    float v = 1.4142135623730950488f * sqrt(1 + 2.7182818284590452354f * x);
    float N2 = 10.242640687119285146f + 1.9797586132081854940f * v;
    float N1 = 0.29289321881345247560f * (1.4142135623730950488f + N2);
    w0 = -1.0f + v * (N2 + v) / (N2 + v + N1 * v);
  }

  while (true) {
    float e = exp(w0);
    float f = w0 * e - x;
    w1 = w0 + ((f+f) * (1.0f + w0)) / (f * (2.0f + w0) - (e+e) * (1.0f + w0) * (1.0f + w0));
    if (fabs(w0 / w1 - 1.0f) < 0.00034526698300124390840f) {
      break;
    }
    w0 = w1;
  }

  return w1;
}
/*
 * END EXTERNAL IMPLEMENTATION
*/

// full precision but susceptible to numerical instability
// hand-crafted constant: 0xBFD78B56362CEF37
#define MINUS_1_OVER_E -0.3678794411714423f

template<typename scalar_t>
struct minus_1_over_e;

template<>
struct minus_1_over_e<float>
{
    // hand-crafted constant: 0xBEBC5AB1
    // converting the double constant rounds down -> instability
    static constexpr float value = -0.36787942;
};

template<>
struct minus_1_over_e<double>
{
    // hand-crafted constant: 0xBFD78B56362CEF37
    // one bit "larger" than the numpy result -1/np.e
    static constexpr double value = /* -0.367879441171442;*/
                                    -3.6787944117144183E-1;
};

// value above which we simply return infinity
template<typename scalar_t>
struct upper_limit;

template<>
struct upper_limit<float> {
    static constexpr float input = 1.0878e+37;
    static constexpr float result = 81.0591;
};

template<>
struct upper_limit<double> {
    static constexpr double input = 6.4694e+305;
    static constexpr double result = 697.7890;
};


template<typename scalar_t>
__global__
void lambertw0_cuda_kernel(
    const scalar_t* __restrict__ z,
    scalar_t* __restrict__ result,
    size_t z_size)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t idx = index; idx < z_size; idx+=stride)
    {
        // input has to be larger than -(1/e)
        if ((z[idx] > minus_1_over_e<scalar_t>::value) && (z[idx] < upper_limit<scalar_t>::input))
        {
            // This also takes care of NaNs..
            result[idx] = plog(z[idx]);
        }
        else if (z[idx] >= upper_limit<scalar_t>::input)
        {
            result[idx] = upper_limit<scalar_t>::result;
        }
        else
        {
            result[idx] = -std::numeric_limits<scalar_t>::infinity();
        }
    }
}

torch::Tensor lambertw0_cuda(torch::Tensor z)
{
    auto result = torch::zeros_like(z);

    int z_size = 1;

    for (size_t idx=0; idx < z.dim(); ++idx)
    {
        z_size *= z.size(idx);
    }

    const int threads = 1024;
    const int blocks = (z_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(z.type(), "lambertw0_cuda", ([&] {
                lambertw0_cuda_kernel<scalar_t><<<blocks, threads>>>(
                        z.data<scalar_t>(),
                        result.data<scalar_t>(),
                        z_size);
                }));

    return result;
}
