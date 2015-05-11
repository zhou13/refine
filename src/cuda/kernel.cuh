#include "kernels/reduce.cuh"

template<int NT>
__global__ void kReduceSqrtXi2(
    int image_size,
    float *g_fimg_r,
    float *g_fimg_i,
    float *g_sqrtXi2
)
{
	typedef mgpu::CTAReduce<NT, mgpu::plus<float>> R;
    __shared__ typename R::Storage reduce_storage;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = (tid + blockDim.x * blockIdx.x) * image_size;

    float sum = 0;
    for (int pixel = 0; pixel < image_size; pixel += NT) {
        int idx = offset + pixel;
        float rr = g_fimg_r[idx];
        float ii = g_fimg_i[idx];
        float value = rr * rr + ii * ii;
        sum += R::Reduce(tid, value, reduce_storage, mgpu::plus<float>());
    }
    if (!tid) {
        g_sqrtXi2[bid] = sum;
    }
}

template<int NT>
__global__ void kShiftPhase(
    int xdim,
    int ydim,
    int nr_trans,
    float *g_fimg_r,
    float *g_fimg_i,
    float *g_trans_x,
    float *g_trans_y,
    float *g_fshift_r,
    float *g_fshift_i
)
{
    extern __shared__ float s_trans[];
    float *s_trans_x = s_trans;
    float *s_trans_y = s_trans + nr_trans;

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    for (int i = tid; i < nr_trans; i += NT) {
        s_trans_x[i] = g_trans_x[i];
        s_trans_y[i] = g_trans_y[i];
    }
    __syncthreads();
    
    int image_size = xdim * ydim;
    int out_offset = image_size * nr_trans * bid + tid;

    for (int pixel = tid; pixel < image_size; pixel += NT) {
        int x = pixel % xdim;
        int y = pixel / xdim;

        float c = g_fimg_r[image_size * bid + pixel];
        float d = g_fimg_i[image_size * bid + pixel];

        for (int trans_id = 0; trans_id < nr_trans; ++trans_id) {
            float _x = x;
            float _y = y < xdim ? y : y - ydim;

            float xshift = s_trans_x[trans_id];
            float yshift = s_trans_y[trans_id];

            float mult = fmod(_y * xshift + _x * yshift, 1.f);
            if (mult > +0.5f) mult -= 1;
            if (mult < -0.5f) mult += 1;
            float dotp = 2 * M_PI * mult;
            float a = __cosf(dotp);
            float b = __sinf(dotp);
            float ac = a * c;
            float bd = b * d;
            float ab_cd = (a + b) * (c + d);
            g_fshift_r[out_offset + trans_id * image_size] = ac - bd;
            g_fshift_i[out_offset + trans_id * image_size] = ab_cd - ac - bd;
        }
    }
}
