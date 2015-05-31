#include <cmath>
#include <cub/block/block_reduce.cuh>


template<int NT>
__global__ void kReduceSqrtXi2(
    int page_size,
    int image_size,
    float *g_fimg_r,
    float *g_fimg_i,
    float *g_sqrtXi2
)
{
    typedef cub::BlockReduce<float, NT> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = bid * page_size;

    float sum = 0;
    for (int i = 0; i < image_size; i += NT) {
        int pixel = i + tid;
        float value = 0;
        if (pixel < image_size) {
            int idx = offset + pixel;
            float rr = g_fimg_r[idx];
            float ii = g_fimg_i[idx];
            value = rr * rr + ii * ii;
        } else
            value = 0;
        sum += BlockReduceT(temp_storage).Sum(value);
    }
    if (!tid) {
        g_sqrtXi2[bid] = sqrt(sum);
    }
}

template<int NT>
__global__ void kShiftPhase(
    int page_size,
    int image_size,
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
    
    for (int pixel = tid; pixel < image_size; pixel += NT) {
        int x = pixel % xdim;
        int y = pixel / xdim;

        float c = g_fimg_r[page_size * bid + pixel];
        float d = g_fimg_i[page_size * bid + pixel];

        int out_offset = page_size * nr_trans * bid + pixel;
        for (int trans_id = 0; trans_id < nr_trans; ++trans_id) {
            float _x = x;
            float _y = y < xdim ? y : y - ydim;

            float xshift = s_trans_x[trans_id];
            float yshift = s_trans_y[trans_id];

            float mult = fmod(_x * xshift + _y * yshift, 1.f);
            if (mult > +0.5f) mult -= 1;
            if (mult < -0.5f) mult += 1;
            float dotp = 2 * M_PI * mult;
            float a = __cosf(dotp);
            float b = __sinf(dotp);
            float ac = a * c;
            float bd = b * d;
            float ab_cd = (a + b) * (c + d);
            // warning:  this may have precision problems when ac and bd is close!
            g_fshift_r[out_offset + trans_id * page_size] = ac - bd;
            g_fshift_i[out_offset + trans_id * page_size] = ab_cd - ac - bd;
            /*
            if (!bid && !trans_id && !pixel) {
                printf("kShiftPhase()\n");
                printf("    pixel: %d (GPU)\n", pixel);
                printf("    x: %.5f\n", _x);
                printf("    y: %.5f\n", _y);
                printf("    xshift: %.5f\n", xshift);
                printf("    yshift: %.5f\n", yshift);
                printf("    dotp: %.5f\n", dotp);
                printf("    a: %.5f\n", a);
                printf("    b: %.5f\n", b);
                printf("    c: %.5f\n", c);
                printf("    d: %.5f\n", d);
                printf("    ac: %.5f\n", ac);
                printf("    bd: %.5f\n", bd);
                printf("    real: %.5f\n", ac - bd);
                printf("    imag: %.5f\n\n\n", ab_cd - ac - bd);
            }
            */
        }
    }
}

template<int NT>
__global__ void kComputeInvSigma2(
    int page_size,
    int image_size,
    int spectrum_size,
    float sigma2_fudge,
    bool do_zero,
    int *g_resolution,
    int *g_image_to_group,
    float *g_sigma2_noise,
    float *g_invSigma2
)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = bid * page_size;

    int group_id = g_image_to_group[bid];
    for (int pixel = tid; pixel < image_size; pixel += NT) {
        int ires = g_resolution[pixel];
        float invSigma2 = 0.f;
        if (ires > 0 || (pixel == 0 && do_zero)) {
            invSigma2 = 1.f / (
                sigma2_fudge * g_sigma2_noise[group_id * spectrum_size + ires]
            );
            if (pixel == 0)
                printf("invSigma2: %.6f\n", invSigma2);
        }
        g_invSigma2[offset + pixel] = invSigma2;
        // if (!bid && tid == 33) {
        //     printf("tid(GPU)\t:%d\n", tid);
        //     printf("ires\t:%d\n", ires);
        //     printf("group_id\t:%d\n", group_id);
        //     printf("sigma2_fudge\t:%.5f\n", sigma2_fudge);
        //     printf("sigma2_noise\t:%.5f\n", g_sigma2_noise[group_id * spectrum_size + ires]);
        //     printf("invSigma2\t:%.5f\n", invSigma2);
        // }
    }
}


template<int NT>
__global__ void kComputerSquaredDifference(
    int page_size,
    int image_size,
    int nr_trans,
    int nr_orient,
    int order_trans,
    int order_orient,
    int nr_class,
    int iclass,
    bool do_ctf_correction,
    bool do_scale_correction,
    bool do_cc,
    uint8_t *g_significant,
    float *g_fref_r,
    float *g_fref_i,
    float *g_fctf,
    float *g_fshift_r,
    float *g_fshift_i,
    float *g_scale,
    float *g_sqrtXi2,
    float *g_highresXi2,
    float *g_invSigma2,
    float *g_weight,
    uint32_t *g_min_diff2
)
{
    int tid = threadIdx.x;
    int image = blockIdx.x;
    int orient = blockIdx.y;
    
    typedef cub::BlockReduce<float, NT> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    extern __shared__ float buffer[];

    float *s_diff = buffer;
    float *s_suma = buffer + nr_trans;
    uint8_t *s_significant = (uint8_t *)(buffer + 2*nr_trans);

    float scale = 1.f;
    float min_diff = FLT_MAX;

    if (do_scale_correction) {
        scale = g_scale[image];
    }

    int hidden_size = (nr_trans >> order_trans) * (nr_orient >> order_orient);
    for (int i = tid; i < nr_trans; i += NT) {
        s_diff[i] = 0.f;
        s_suma[i] = 0.f;
        int index = (i >> order_trans) +
            (orient >> order_orient) * (nr_trans >> order_trans);
        s_significant[i] = g_significant[image*hidden_size + index];
    }
    __syncthreads();

    for (int _pixel = 0; _pixel < image_size; _pixel += NT) {
        int pixel = _pixel + tid;
        float fref_r = 0.f;
        float fref_i = 0.f;

        if (pixel < image_size) {
            float correction = 1.f;
            if (do_ctf_correction) {
                correction = g_fctf[page_size*image + pixel];
            }
            if (do_scale_correction) {
                correction *= scale;
                correction *= scale;
            }
            fref_r = g_fref_r[page_size*orient + pixel] * correction;
            fref_i = g_fref_i[page_size*orient + pixel] * correction;
        }

        for (int trans = 0; trans < nr_trans; ++trans) {
            if (!s_significant[trans]) {
                continue;
            }

            float d_diff = 0.f, d_suma = 0.f;
            if (pixel < image_size) {
                int ishift = page_size * (trans + nr_trans*image) + pixel;
                if (do_cc) {
                    d_diff -= fref_r * g_fshift_r[ishift];
                    d_diff -= fref_i * g_fshift_i[ishift];
                    d_suma += fref_r * fref_r + fref_i * fref_i;
                } else {
                    float d_real = fref_r - g_fshift_r[ishift];
                    float d_imag = fref_i - g_fshift_i[ishift];
                    float invSigma2 = g_invSigma2[image*page_size + pixel];
                    d_diff = (d_real*d_real + d_imag*d_imag) * 0.5f * invSigma2;
                }
            }
            d_diff = BlockReduceT(temp_storage).Sum(d_diff);
            if (do_cc)
                d_suma = BlockReduceT(temp_storage).Sum(d_suma);
            if (!tid) {
                s_diff[trans] += d_diff;
                if (do_cc)
                    s_suma[trans] += d_suma;
            }
            __syncthreads();
        }
    }


    for (int _i = 0; _i < nr_trans; _i += NT) {
        int i = _i + tid;
        float diff = FLT_MAX;
        if (i < nr_trans && s_significant[i]) {
            diff = s_diff[i];
            if (do_cc)
                diff /= (sqrt(s_suma[i]) * g_sqrtXi2[image]);
            else
                diff += g_highresXi2[image] * 0.5f;
            int index = i + nr_trans*(orient + nr_orient*(iclass + nr_class*image));
            g_weight[index] = diff;
            /*
            if (index == 0) {
                printf("kComputerSquaredDifference\n");
                printf("    tid(GPU)\t:%d\n", tid);
                printf("    gridDim.x\t:%d\n", gridDim.x);
                printf("    gridDim.y\t:%d\n", gridDim.y);
                printf("    nr_trans\t:%d\n", nr_trans);
                printf("    nr_orient\t:%d\n", nr_orient);
                printf("    index\t:%d\n", index);
                printf("    diff2\t: %.9f\n", s_diff[0]);
                printf("    suma2\t: %.9f\n", s_suma[0]);
                printf("    final_diff\t: %.9f\n\n", s_diff[0] / s_suma[0] / g_sqrtXi2[image]);
            }
            */
        }
        float my_min_diff = BlockReduceT(temp_storage).Reduce(diff, cub::Min());
        if (i < nr_trans) {
            if (my_min_diff < min_diff)
                min_diff = my_min_diff;
        }
    }
    if (!tid) {
        atomicMin(g_min_diff2 + image, flipFloat(min_diff));
    }
}

template<int NT>
__global__ void kClearWeight(
    int page_size,
    int image_size,
    float *g_fimg_sum_r,
    float *g_fimg_sum_i,
    float *g_fweight
)
{
    int tid = threadIdx.x;
    int orient = blockIdx.x;
    for (int i = tid; i < image_size; i += NT) {
        g_fimg_sum_r[orient * page_size + i] = 0.f;
        g_fimg_sum_i[orient * page_size + i] = 0.f;
        g_fweight[orient * page_size + i] = 0.f;
    }
}

template<int NT>
__global__ void kSumWeight(
    int page_size,
    int image_size,
    int spectrum_size,
    int nr_trans,
    int nr_orient,
    int nr_class,
    int iclass,
    bool do_ctf_correction,
    bool refs_are_ctf_corrected,
    bool do_scale_correction,
    float *g_weight,
    float *g_weight_sum,
    float *g_weight_sig,
    int *g_image_to_group,
    int *g_resolution,
    float *g_data_vs_prior_class,
    float *g_fref_r,
    float *g_fref_i,
    float *g_fctf,
    float *g_fshift_r,
    float *g_fshift_i,
    float *g_fshift_nomask_r,
    float *g_fshift_nomask_i,
    float *g_scale,
    float *g_invSigma2,

    float *g_wsum_sigma2_noise,
    float *g_wsum_norm_correction,
    float *g_wsum_scale_correction_XA,
    float *g_wsum_scale_correction_AA,
    float *g_fimg_sum_r,
    float *g_fimg_sum_i,
    float *g_fweight
)
{
    int tid = threadIdx.x;
    int image = blockIdx.x;
    int orient = blockIdx.y;
    
    float scale = 1.f;

    if (do_scale_correction) {
        scale = g_scale[image];
    }
    float image_weight_sig = g_weight_sig[image];
    float image_weight_sum = g_weight_sum[image];

    double wsum_norm_correction = 0.f;
    for (int pixel = tid; pixel < image_size; pixel += NT) {
        int resolution = -1;
        float fref_r = 0.f;
        float fref_i = 0.f;
        float invSigma2 = 0.f;
        float data_vs_prior_class = 0;

        float ctf = 1.f;
        float correction = 1.f;
        if (do_ctf_correction) {
            ctf = g_fctf[page_size*image + pixel];
            if (refs_are_ctf_corrected)
                correction = ctf;
        }
        if (do_scale_correction) {
            ctf *= scale;
            correction *= scale;
        }
        fref_r = g_fref_r[page_size*orient + pixel] * correction;
        fref_i = g_fref_i[page_size*orient + pixel] * correction;
        invSigma2 = g_invSigma2[pixel];
        resolution = g_resolution[pixel];
        if (resolution > -1)
            data_vs_prior_class = g_data_vs_prior_class[resolution];

        float fweight = 0.f;
        float fimg_sum_r = 0.f;
        float fimg_sum_i = 0.f;
        float wsum_sigma2_noise = 0.f;
        float wsum_scale_correction_XA = 0.f;
        float wsum_scale_correction_AA = 0.f;
        for (int trans = 0; trans < nr_trans; ++trans) {
            int index = trans + nr_trans*(orient + nr_orient*(iclass + nr_class*image));

            float weight = g_weight[index];
            if (weight < image_weight_sig)
                continue;
            weight /= image_weight_sum;

            int ishift = page_size * (trans + nr_trans*image) + pixel;

            if (resolution > -1) {
                float fshift_r = g_fshift_r[ishift];
                float fshift_i = g_fshift_i[ishift];
                float d_real = fref_r - fshift_r;
                float d_imag = fref_i - fshift_i;
                float wdiff2 = weight * (d_real * d_real + d_imag * d_imag);
                wsum_norm_correction += wdiff2;
                wsum_sigma2_noise += wdiff2;

                if (do_scale_correction && data_vs_prior_class > 3.f) {
                    float sumXA = weight * (fref_r * fshift_r + fref_i * fshift_i);
                    float sumAA = weight * (fref_r * fref_r + fref_i * fref_i);
                    wsum_scale_correction_XA += sumXA;
                    wsum_scale_correction_AA += sumAA;
                }
            }

            float weightxinvsigma2 = weight * ctf * invSigma2;
            fimg_sum_r += g_fshift_nomask_r[ishift] * weightxinvsigma2;
            fimg_sum_i += g_fshift_nomask_i[ishift] * weightxinvsigma2;
            fweight += weightxinvsigma2 * ctf;
            // if (pixel == 0 && image == 0 && orient == 0) {
            //     printf("==================[kSumWeight]================\n");
            //     printf("    pixel: %d\n", pixel);
            //     printf("    image: %d\n", image);
            //     printf("    orient: %d\n", orient);
            //     printf("    trans: %d\n", trans);
            //     printf("    ctf: %.8f\n", ctf);
            //     printf("    weight: %.8f\n", weight);
            //     printf("    invSigma2: %.8f\n", invSigma2);
            //     printf("    weightxinvsigma2: %.8f\n", weightxinvsigma2);
            //     printf("    g_fshift_nomask_r: %.8f\n\n", g_fshift_nomask_r[ishift]);
            // }
        }

        atomicAdd(   g_fweight + pixel + orient*page_size, fweight);
        atomicAdd(g_fimg_sum_r + pixel + orient*page_size, fimg_sum_r);
        atomicAdd(g_fimg_sum_i + pixel + orient*page_size, fimg_sum_i);

        if (resolution > -1) {
            int index_group = g_image_to_group[image] * spectrum_size + resolution;
            atomicAdd(g_wsum_sigma2_noise + index_group, wsum_sigma2_noise);
            int index_image = image * spectrum_size + resolution;
            atomicAdd(g_wsum_scale_correction_XA + index_image, wsum_scale_correction_XA);
            atomicAdd(g_wsum_scale_correction_AA + index_image, wsum_scale_correction_AA);
        }
    }
    atomicAdd(g_wsum_norm_correction + image, wsum_norm_correction);
}
