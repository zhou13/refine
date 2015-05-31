#include "cuda/utils.cuh"
#include "cuda/expectation.cuh"
#include "cuda/kernel.cuh"
#include "ml_optimiser.h"

// number of thread per block
const int THREAD_PER_BLOCK = 192;
// alignment of CUDA
const int CUDA_ALIGN = 32;

struct HostData {
    int nr_image;
    int nr_trans;
    int nr_orient;
    int nr_group;
    int nr_class;
    int nr_spectrum;
    int xdim;
    int ydim;
    int page_size;
    int image_size;
};

struct DeviceData {
    MGPU_MEM(float) fctf;
    MGPU_MEM(float) invSigma2;

    // TODO: batch operation to save memory
    MGPU_MEM(float) fref_r;
    MGPU_MEM(float) fref_i;

    MGPU_MEM(float) fimg_r;
    MGPU_MEM(float) fimg_i;
    MGPU_MEM(float) fimg_nomask_r;
    MGPU_MEM(float) fimg_nomask_i;

    MGPU_MEM(float) fshift_r;
    MGPU_MEM(float) fshift_i;
    MGPU_MEM(float) fshift_nomask_r;
    MGPU_MEM(float) fshift_nomask_i;

    MGPU_MEM(float) sigma2_noise;
    MGPU_MEM(float) scale;
    MGPU_MEM(float) sqrtXi2;
    MGPU_MEM(float) highresXi2;
    MGPU_MEM(float) data_vs_prior_class;

    // should be float, but workaround for the lack of float version of atomicMin
    MGPU_MEM(uint32_t) min_diff2;

    MGPU_MEM(float) weight;
    MGPU_MEM(float) weight_sum;
    MGPU_MEM(float) weight_sig;

    MGPU_MEM(float) trans_x;
    MGPU_MEM(float) trans_y;

    MGPU_MEM(int) resolution;
    MGPU_MEM(int) image_to_group;
    MGPU_MEM(uint8_t) significant;

    MGPU_MEM(float) wsum_sigma2_noise;
    MGPU_MEM(float) wsum_norm_correction;
    MGPU_MEM(float) wsum_scale_correction_XA;
    MGPU_MEM(float) wsum_scale_correction_AA;
    MGPU_MEM(float) fimg_sum_r;
    MGPU_MEM(float) fimg_sum_i;
    MGPU_MEM(float) fweight;

    ContextPtr context;

    void copyArrayToDeviceComplex(
            const vector<MultidimArray<Complex>> &arr,
            MGPU_MEM(float) d_real,
            MGPU_MEM(float) d_imag,
            bool align = true)
    {
        int len = arr.size();
        assert(len > 0);
        vector<float> h_real, h_imag;
        int size = len * arr[0].nzyxdim;
        h_real.reserve(size);
        h_imag.reserve(size);

        for (int i = 0; i < len; ++i) {
            FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(arr[i]) {
#ifdef CHECK_RESULT
                assert(!isnan(DIRECT_MULTIDIM_ELEM(arr[i], n).real));
                assert(!isnan(DIRECT_MULTIDIM_ELEM(arr[i], n).imag));
#endif
                h_real.push_back(DIRECT_MULTIDIM_ELEM(arr[i], n).real);
                h_imag.push_back(DIRECT_MULTIDIM_ELEM(arr[i], n).imag);
            }
            while (align && h_real.size() % CUDA_ALIGN) {
                h_real.push_back(0.f);
                h_imag.push_back(0.f);
            }
        }

        d_real->FromHost(h_real);
        d_imag->FromHost(h_imag);
    }

    void copyArrayToDeviceDouble(
            const vector<MultidimArray<double>> &arr,
            MGPU_MEM(float) d_real,
            bool align = false)
    {
        int len = arr.size();
        assert(len > 0);
        vector<float> h_real, h_imag;
        int size = align_address(len * arr[0].nzyxdim, CUDA_ALIGN);
        h_real.reserve(size);

        for (int i = 0; i < len; ++i) {
            FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(arr[i]) {
#ifdef CHECK_RESULT
                assert(!isnan(DIRECT_MULTIDIM_ELEM(arr[i], n)));
#endif
                h_real.push_back(DIRECT_MULTIDIM_ELEM(arr[i], n));
            }
            while (align && h_real.size() % CUDA_ALIGN) {
                h_real.push_back(0.f);
            }
        }

        d_real->FromHost(h_real);
    }

    void copyArrayFromDeviceDouble(
            MGPU_MEM(float) data,
            vector<MultidimArray<double>> *array,
            int page_size)
    {
        static int count = 0;
        static double sum = 0;

        int nr = array->size();
        vector<float> hdata;
        data->ToHost(hdata, page_size * nr);
        for (int i = 0; i < nr; ++i) {
            FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY((*array)[i]) {
                DIRECT_MULTIDIM_ELEM((*array)[i], n) = hdata[n + i*page_size];
                sum += hdata[n + i*page_size];
            }
        }
        printf("HASH DOUBLE  %3d: %.9f\n", count, sum);
    }

    void copyArrayFromDeviceComplex(
            MGPU_MEM(float) data_r,
            MGPU_MEM(float) data_i,
            vector<MultidimArray<Complex>> *array,
            int page_size)
    {
        static int count = 0;
        static double sum = 0;

        int nr = array->size();
        vector<float> hdata_r, hdata_i;
        data_r->ToHost(hdata_r, page_size * nr);
        data_i->ToHost(hdata_i, page_size * nr);
        for (int i = 0; i < nr; ++i) {
            FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY((*array)[i]) {
                DIRECT_MULTIDIM_ELEM((*array)[i], n).real = hdata_r[n + i*page_size];
                DIRECT_MULTIDIM_ELEM((*array)[i], n).imag = hdata_i[n + i*page_size];
                sum += hdata_r[n + i*page_size];
                sum += hdata_i[n + i*page_size];
            }
        }
        printf("HASH COMPLEX %3d: %.9f\n", count, sum);
    }

};

ExpectationCudaSolver::ExpectationCudaSolver(MlOptimiser *ml) :
    ml(ml)
{
    h = new HostData();
    d = nullptr;
}

ExpectationCudaSolver::~ExpectationCudaSolver()
{
    delete h;
    delete d;
}
// Equal-to because of the series: the nth images in a series will have the same maximum as the first one

void ExpectationCudaSolver::initialize()
{
    static bool initialized = false;
    if (!initialized)
        initialized = true;
    else
        return;

    assert(!ml->do_sim_anneal);
    if (d)
        delete d;
    d = new DeviceData();

    cudaDeviceSynchronize();
    cudaDeviceReset();
    d->context = CreateCudaDevice(0);

    assert(ml->exp_Fimgs.size());
    h->nr_image = ml->exp_Fimgs.size();
    int max_xdim = ml->exp_Fimgs[0].xdim;
    int max_ydim = ml->exp_Fimgs[0].ydim;
    int max_nr_trans_samples = ml->exp_nr_trans *
            ml->sampling.oversamplingFactorTranslations(ml->adaptive_oversampling);
    int max_nr_orient_samples = ml->exp_nr_rot *
            ml->sampling.oversamplingFactorOrientations(ml->adaptive_oversampling);

    int page_size = align_address(max_xdim * max_ydim, CUDA_ALIGN);
    int total_size = page_size * h->nr_image;

    d->resolution = d->context->Malloc<int>(page_size);

    d->fctf = d->context->Malloc<float>(total_size);
    d->fimg_r = d->context->Malloc<float>(total_size);
    d->fimg_i = d->context->Malloc<float>(total_size);
    d->fimg_nomask_r = d->context->Malloc<float>(total_size);
    d->fimg_nomask_i = d->context->Malloc<float>(total_size);

    d->fref_r = d->context->Malloc<float>(max_nr_orient_samples * page_size);
    d->fref_i = d->context->Malloc<float>(max_nr_orient_samples * page_size);

    d->trans_x = d->context->Malloc<float>(max_nr_trans_samples);
    d->trans_y = d->context->Malloc<float>(max_nr_trans_samples);

    d->fshift_r = d->context->Malloc<float>(total_size * max_nr_trans_samples);
    d->fshift_i = d->context->Malloc<float>(total_size * max_nr_trans_samples);
    d->fshift_nomask_r = d->context->Malloc<float>(total_size * max_nr_trans_samples);
    d->fshift_nomask_i = d->context->Malloc<float>(total_size * max_nr_trans_samples);

    d->invSigma2 = d->context->Malloc<float>(total_size);
    d->sqrtXi2 = d->context->Malloc<float>(h->nr_image);
    d->scale = d->context->Malloc<float>(h->nr_image);
    d->min_diff2 = d->context->Malloc<uint32_t>(h->nr_image);
    d->highresXi2 = d->context->Malloc<float>(h->nr_image);
    d->weight = d->context->Malloc<float>(
                ml->mymodel.nr_classes *
                max_nr_orient_samples *
                max_nr_trans_samples *
                h->nr_image);
    d->significant = d->context->Malloc<uint8_t>(h->nr_image * ml->exp_nr_rot * ml->exp_nr_trans);

    h->nr_group = ml->mymodel.nr_groups;
    d->image_to_group = d->context->Malloc<int>(h->nr_image);
    h->nr_spectrum = ml->mymodel.sigma2_noise[0].nzyxdim;
    d->sigma2_noise = d->context->Malloc<float>(h->nr_group * h->nr_spectrum);
    d->data_vs_prior_class = d->context->Malloc<float>(h->nr_spectrum);

    d->weight_sum = d->context->Malloc<float>(h->nr_image);
    d->weight_sig = d->context->Malloc<float>(h->nr_image);

    d->wsum_sigma2_noise = d->context->Malloc<float>(h->nr_group * h->nr_spectrum);
    d->wsum_norm_correction = d->context->Malloc<float>(h->nr_image);
    d->wsum_scale_correction_XA = d->context->Malloc<float>(h->nr_image * h->nr_spectrum);
    d->wsum_scale_correction_AA = d->context->Malloc<float>(h->nr_image * h->nr_spectrum);
    d->fimg_sum_r = d->context->Malloc<float>(page_size * max_nr_orient_samples);
    d->fimg_sum_i = d->context->Malloc<float>(page_size * max_nr_orient_samples);
    d->fweight    = d->context->Malloc<float>(page_size * max_nr_orient_samples);

    std::cerr << "ExpectationCudaSolver::initialize finished" << std::endl
              << "    nr_image: " << h->nr_image << std::endl
              << "    nr_group: " << h->nr_group << std::endl
              << "    nr_spectrum: " << h->nr_spectrum << std::endl
              << "    max_xdim: " << max_xdim << std::endl
              << "    max_ydim: " << max_ydim << std::endl
              << "    max_nr_trans_samples: " << max_nr_trans_samples << std::endl;
}

void ExpectationCudaSolver::copyWindowsedImagesToGPU()
{
    h->nr_image = ml->exp_nr_images;
    h->xdim = ml->exp_current_image_size / 2 + 1;
    h->ydim = ml->exp_current_image_size;
    h->image_size = h->xdim * h->ydim;
    h->page_size = align_address(h->image_size, CUDA_ALIGN);
    std::cerr << "ExpectationCudaSolver::copyWindowsedImagesToGPU" << std::endl
                 ;
              //<< "    xdim:        " << h->xdim << std::endl
              //<< "    ydim:        " << h->ydim << std::endl
              //<< "    image_size:  " << h->image_size << std::endl
              //<< "    page_size:   " << h->page_size << std::endl;

    vector<MultidimArray<Complex>> fimg_win(h->nr_image);
    vector<MultidimArray<Complex>> fimg_nomask_win(h->nr_image);
    vector<MultidimArray<double>>  fctf_win(h->nr_image);
    for (int i = 0; i < h->nr_image; ++i) {
        windowFourierTransform(ml->exp_Fctfs[i], fctf_win[i], ml->exp_current_image_size);
        windowFourierTransform(ml->exp_Fimgs[i], fimg_win[i], ml->exp_current_image_size);
        windowFourierTransform(ml->exp_Fimgs_nomask[i], fimg_nomask_win[i], ml->exp_current_image_size);
    }
    assert(fimg_win.size() == h->nr_image);
    assert(fimg_win[0].xdim == h->xdim);
    assert(fimg_win[0].ydim == h->ydim);
    d->copyArrayToDeviceComplex(fimg_win, d->fimg_r, d->fimg_i, true);
    d->copyArrayToDeviceComplex(fimg_nomask_win, d->fimg_nomask_r, d->fimg_nomask_i, true);

    d->copyArrayToDeviceDouble(ml->mymodel.sigma2_noise, d->sigma2_noise, false);
    d->copyArrayToDeviceDouble(fctf_win, d->fctf, true);

    if (h->ydim == ml->coarse_size) {
        d->resolution->FromHost(ml->Mresol_coarse.data, ml->Mresol_coarse.nzyxdim);
    } else {
        d->resolution->FromHost(ml->Mresol_fine.data, ml->Mresol_fine.nzyxdim);
    }
    d->highresXi2->FromHost(
                std::vector<float>(ml->exp_highres_Xi2_imgs.begin(),
                                   ml->exp_highres_Xi2_imgs.end()));

    vector<float> scale(h->nr_image, 0.f);
    for (int i = 0; i < h->nr_image; ++i) {
        int ipart = ml->exp_iimg_to_ipart[i];
        int iseries = i - ml->exp_starting_image_no[ipart];
        int particle_id = ml->exp_ipart_to_part_id[ipart];
        int group_id = ml->mydata.getGroupId(particle_id, iseries);
        scale[i] = ml->mymodel.scale_correction[group_id];
        assert(iseries == 0);
    }
    d->scale->FromHost(scale);

    vector<int> image_to_group;
    image_to_group.reserve(h->nr_image);
    for (int i = 0; i < h->nr_image; ++i) {
        int ipart = ml->exp_iimg_to_ipart[i];
        int iseries = i - ml->exp_starting_image_no[ipart];
        int particle_id = ml->exp_ipart_to_part_id[ipart];
        int group_id = ml->mydata.getGroupId(particle_id, iseries);
        image_to_group.push_back(group_id);
        assert(iseries == 0);
    }
    d->image_to_group->FromHost(image_to_group);

    kReduceSqrtXi2<THREAD_PER_BLOCK><<<h->nr_image, THREAD_PER_BLOCK>>>(
        h->page_size,
        h->image_size,
        *d->fimg_r,
        *d->fimg_i,
        *d->sqrtXi2
    );

#ifdef CHECK_RESULT
    if ((ml->iter == 1 && ml->do_firstiter_cc) || ml->do_always_cc) {
        vector<float> _sqrtXi2;
        d->sqrtXi2->ToHost(_sqrtXi2, h->nr_image);
        check_diff_array(ml->exp_local_sqrtXi2, _sqrtXi2);
        std::cerr << "    sqrtXi2 seems to be correct" << std::endl;
    }
#endif

    kComputeInvSigma2<THREAD_PER_BLOCK><<<h->nr_image, THREAD_PER_BLOCK>>>(
        h->page_size,
        h->image_size,
        h->nr_spectrum,
        ml->sigma2_fudge,
        false,
        *d->resolution,
        *d->image_to_group,
        *d->sigma2_noise,
        *d->invSigma2
    );
#ifdef CHECK_RESULT
    {
        vector<float> invSigma2;
        d->invSigma2->ToHost(invSigma2, h->nr_image * h->page_size);
        int i = 0;
        int nr_error = 0;
        int nr_total = 0;
        for (auto &img : ml->exp_local_Minvsigma2s) {
            FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img) {
                float error = relative_diff(0.001 + abs(DIRECT_MULTIDIM_ELEM(img, n)),
                                            0.001 + abs(invSigma2[i * h->page_size + n]));
                if (error > 0.1) {
                    fprintf(stderr, "    invSigma2 error %.8f vs %.8f at (%d, %d)\n",
                            DIRECT_MULTIDIM_ELEM(img, n),
                            invSigma2[i * h->page_size + n], i, (int)n);
                    ++nr_error;
                }
                ++nr_total;
            }
            ++i;
        }
        float error_rate = 100.f * float(nr_error) / nr_total;
        printf("    invSigma2's error rate is %.5f%%\n", error_rate);
    }
#endif
}

void ExpectationCudaSolver::getShiftedImages()
{
    std::vector<float> translation_x;
    std::vector<float> translation_y;
    h->nr_trans = 0;
    for (int i = 0; i < ml->exp_nr_trans; ++i) {
        std::vector<Matrix1D<double>> tmp_trans;
        ml->sampling.getTranslations(i, ml->exp_current_oversampling, tmp_trans);
        for (auto &dir : tmp_trans) {
            translation_x.push_back(XX(dir) / -ml->mymodel.ori_size);
            translation_y.push_back(YY(dir) / -ml->mymodel.ori_size);
            ++h->nr_trans;
        }
    }

    d->trans_x->FromHost(translation_x, h->nr_trans);
    d->trans_y->FromHost(translation_y, h->nr_trans);
    std::cerr << "ExpectationCudaSolver::getShiftedImages" << std::endl
                 ;
              // << "    nr_trans:  " << h->nr_trans << std::endl;

    kShiftPhase<THREAD_PER_BLOCK>
            <<<h->nr_image, THREAD_PER_BLOCK, 2 * sizeof(float) * h->nr_trans>>>(
        h->page_size,
        h->image_size,
        h->xdim,
        h->ydim,
        h->nr_trans,
        *d->fimg_r,
        *d->fimg_i,
        *d->trans_x,
        *d->trans_y,
        *d->fshift_r,
        *d->fshift_i
    );
    kShiftPhase<THREAD_PER_BLOCK>
            <<<h->nr_image, THREAD_PER_BLOCK, 2 * sizeof(float) * h->nr_trans>>>(
        h->page_size,
        h->image_size,
        h->xdim,
        h->ydim,
        h->nr_trans,
        *d->fimg_nomask_r,
        *d->fimg_nomask_i,
        *d->trans_x,
        *d->trans_y,
        *d->fshift_nomask_r,
        *d->fshift_nomask_i
    );

#ifdef CHECK_RESULT
    {
        vector<float> shifted_r, shifted_i;
        vector<float> shifted_nomask_r, shifted_nomask_i;
        int samples = h->nr_image * h->nr_trans;
        assert(samples == ml->exp_local_Fimgs_shifted.size());
        {
            d->fshift_r->ToHost(shifted_r, samples * h->page_size);
            d->fshift_i->ToHost(shifted_i, samples * h->page_size);
            int i = 0;
            int nr_total = 0;
            int nr_violate = 0;
            for (auto &img : ml->exp_local_Fimgs_shifted) {
                FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img) {
                    float error_r = relative_diff(0.001 + abs(DIRECT_MULTIDIM_ELEM(img, n).real),
                                                  0.001 + abs(shifted_r[h->page_size * i + n]));
                    float error_i = relative_diff(0.001 + abs(DIRECT_MULTIDIM_ELEM(img, n).imag),
                                                  0.001 + abs(shifted_i[h->page_size * i + n]));
                    nr_total += 2;
                    nr_violate += (error_r > 0.05) + (error_i > 0.05);
                }
                ++i;
            }
            float error_rate = nr_violate / (float)(nr_total) * 100;
            assert(error_rate < 20);
            std::cerr << "    fshift's error rate is " << error_rate << "%" << std::endl;
        }
        {
            d->fshift_nomask_r->ToHost(shifted_nomask_r, samples * h->page_size);
            d->fshift_nomask_i->ToHost(shifted_nomask_i, samples * h->page_size);
            int i = 0;
            int nr_total = 0;
            int nr_violate = 0;
            for (auto &img : ml->exp_local_Fimgs_shifted_nomask) {
                FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img) {
                    float error_r = relative_diff(0.001 + abs(DIRECT_MULTIDIM_ELEM(img, n).real),
                                                  0.001 + abs(shifted_nomask_r[h->page_size * i + n]));
                    float error_i = relative_diff(0.001 + abs(DIRECT_MULTIDIM_ELEM(img, n).imag),
                                                  0.001 + abs(shifted_nomask_i[h->page_size * i + n]));
                    nr_total += 2;
                    nr_violate += (error_r > 0.05) + (error_i > 0.05);
                }
                ++i;
            }
            float error_rate = nr_violate / (float)(nr_total) * 100;
            assert(error_rate < 20);
            std::cerr << "    fshift_nomask's error rate is " << error_rate << "%" << std::endl;
        }
    }
#endif
}

void ExpectationCudaSolver::getSquaredDifference()
{
    h->nr_class = ml->mymodel.nr_classes;
    h->nr_orient = ml->exp_nr_rot * ml->exp_nr_oversampled_rot;

    std::cerr << "ExpectationCudaSolver::getSquareDifference" << std::endl
                 ;
              //<< "    nr_class:  " << h->nr_class << std::endl
              //<< "    nr_orient:  " << h->nr_orient << std::endl
              //<< "    nr_trans:  " << h->nr_trans << std::endl
              //<< "    nr_project:  " << h->nr_class * h->nr_orient << std::endl
              //<< "    iclass_min:  " << ml->iclass_min << std::endl
              //<< "    iclass_max:  " << ml->iclass_max << std::endl;

    vector<Matrix1D<double>> orientations;
    vector<Matrix2D<double>> matrices;
    vector<double> dir, psi;
    for (int irot = 0; irot < ml->exp_nr_rot; ++irot) {
        int _dir = irot / ml->exp_nr_psi;
        int _psi = irot % ml->exp_nr_psi;
        vector<Matrix1D<double>> tmp_orient;
        ml->sampling.getOrientations(_dir, _psi, ml->exp_current_oversampling, tmp_orient);
        orientations.insert(orientations.end(), tmp_orient.begin(), tmp_orient.end());

        for (int i = 0; i < (int)tmp_orient.size(); ++i) {
            dir.push_back(_dir);
            psi.push_back(_psi);
        }
    }

    assert(h->nr_orient == (int)orientations.size());
    for (auto &ori : orientations) {
        Matrix2D<double> A;
        Euler_angles2matrix(XX(ori), YY(ori), ZZ(ori), A);
        // A = (ml->exp_R_mic * A).inv();
        A = A.inv();
        matrices.push_back(A);
    }

#ifdef CHECK_RESULT
    assert(ml->exp_Mweight.nzyxdim == h->nr_image * h->nr_class * ml->sampling.NrSamplingPoints(ml->exp_current_oversampling));
#endif
    d->weight->FromHost(vector<float>(h->nr_image * h->nr_class * ml->sampling.NrSamplingPoints(ml->exp_current_oversampling), -999.f));
    d->min_diff2->FromHost(vector<uint32_t>(h->nr_image, flipFloat(FLT_MAX)));

    // TODO: batchlize h->nr_class
    vector<MultidimArray<Complex>> fref(h->nr_orient);
    for (int iclass = ml->iclass_min; iclass <= ml->iclass_max; ++iclass) {
        vector<uint8_t> significant;
        for (int i = 0; i < h->nr_image; ++i)
            for (int j = 0; j < ml->exp_nr_rot; ++j) {
                int dir = j / ml->exp_nr_psi;
                int psi = j % ml->exp_nr_psi;
                float pdf_orientation;
                if (ml->mymodel.orientational_prior_mode == NOPRIOR)
                    pdf_orientation = DIRECT_MULTIDIM_ELEM(ml->mymodel.pdf_direction[iclass], dir);
                else
                    pdf_orientation = ml->sampling.getPriorProbability(dir, psi);

                for (int k = 0; k < ml->exp_nr_trans; ++k) {
                    if (ml->exp_ipass == 0)
                        significant.push_back(1);
                    else {
                        int hidden = k + ml->exp_nr_trans*(j + ml->exp_nr_rot*iclass);
                        bool sig = DIRECT_A2D_ELEM(ml->exp_Mcoarse_significant, i, hidden);
                        significant.push_back(sig && pdf_orientation > 0);
                    }
                }
            }

        for (int iorient = 0; iorient < (int)orientations.size(); ++iorient) {
            fref[iorient].resize(h->ydim, h->xdim);
            ml->mymodel.PPref[iclass].get2DFourierTransform(fref[iorient], matrices[iorient], IS_INV);
        }
        d->copyArrayToDeviceComplex(fref, d->fref_r, d->fref_i, true);
        d->significant->FromHost(significant);

        bool do_cc = (ml->iter == 1 && ml->do_firstiter_cc) || ml->do_always_cc;
        size_t share_size = (sizeof(float) * 2 + sizeof(uint8_t)) * h->nr_trans;
        kComputerSquaredDifference<THREAD_PER_BLOCK>
                <<<dim3(h->nr_image, h->nr_orient), THREAD_PER_BLOCK, share_size>>>(
            h->page_size,
            h->image_size,
            h->nr_trans,
            h->nr_orient,
            ml->exp_current_oversampling * 2,
            ml->exp_current_oversampling * (ml->mymodel.ref_dim == 2 ? 1 : 3),
            h->nr_class,
            iclass,
            ml->do_ctf_correction && ml->refs_are_ctf_corrected,
            ml->do_scale_correction,
            do_cc,
            *d->significant,
            *d->fref_r,
            *d->fref_i,
            *d->fctf,
            *d->fshift_r,
            *d->fshift_i,
            *d->scale,
            *d->sqrtXi2,
            *d->highresXi2,
            *d->invSigma2,
            *d->weight,
            *d->min_diff2
        );
    }
#ifdef CHECK_RESULT
    {
        int mydim = h->nr_image * h->nr_class * h->nr_trans * h->nr_orient;
        // fprintf(stderr, "mydim: %d\nexp_Mweight dim: %d\n", mydim, (int)ml->exp_Mweight.nzyxdim);
        assert(ml->exp_Mweight.nzyxdim == mydim);
        vector<float> weight;
        d->weight->ToHost(weight, mydim);

        int nr_error = 0;
        int nr_total = 0;
        for (int i = 0; i < mydim; ++i) {
            int index = i;
            int over_trans = index % ml->exp_nr_oversampled_trans;
            index /= ml->exp_nr_oversampled_trans;
            int over_rot = index % ml->exp_nr_oversampled_rot;
            index /= ml->exp_nr_oversampled_rot;
            // int hidden = index;
            int itrans = index % ml->exp_nr_trans;
            index /= ml->exp_nr_trans;
            int irot = index % ml->exp_nr_rot;
            index /= ml->exp_nr_rot;
            int iclass = index % h->nr_class;
            index /= h->nr_class;
            int image = index;


            int my_trans = over_trans + ml->exp_nr_oversampled_trans * itrans;
            int my_rot = over_rot + ml->exp_nr_oversampled_rot * irot;
            index = my_trans + h->nr_trans*(my_rot + h->nr_orient*(iclass + h->nr_class*image));
            float Mweight = DIRECT_MULTIDIM_ELEM(ml->exp_Mweight, i);
            if (relative_diff(0.001+abs(Mweight), 0.001+abs(weight[index])) > 0.05  ) {
               /* printf("    Mweight: %.8f\t myWeight: %.8f\tsig: %d\tindex: %d\n",
                       Mweight, weight[index],
                       ml->exp_ipass == 0 ? 9 : DIRECT_MULTIDIM_ELEM(ml->exp_Mcoarse_significant, hidden),
                       index);*/
                ++nr_error;
            }
            ++nr_total;
        }
        float error_rate = 100.f * float(nr_error) / nr_total;
        std::cerr << "    Mweight's error rate is " << error_rate << "%" << std::endl;

        vector<uint32_t> min_diff2i;
        d->min_diff2->ToHost(min_diff2i, h->nr_image);
        for (int i = 0; i < h->nr_image; ++i) {
            float min_diff2 = reverseFlipFloat(min_diff2i[i]);
            assert(flipFloat(min_diff2) == min_diff2i[i]);
            if (relative_diff(ml->exp_min_diff2[i], min_diff2) > 0.05) {
                fprintf(stderr, "    min_diff2 has precision problem at %d (%.8f vs %.8f)\n",
                        i, ml->exp_min_diff2[i], min_diff2);
                exit(1);
            }
        }
    }
#endif
}

void ExpectationCudaSolver::convertSquaredDifferencesToWeights()
{
    int mydim = h->nr_image * h->nr_class * h->nr_trans * h->nr_orient;
    vector<float> weight;
    d->weight->ToHost(weight, mydim);

    ml->exp_Mweight.resize(h->nr_image, ml->mymodel.nr_classes * ml->sampling.NrSamplingPoints(ml->exp_current_oversampling, false));
    ml->exp_min_diff2.resize(h->nr_image);

    for (int i = 0; i < mydim; ++i) {
        int index = i;
        int over_trans = index % ml->exp_nr_oversampled_trans;
        index /= ml->exp_nr_oversampled_trans;
        int over_rot = index % ml->exp_nr_oversampled_rot;
        index /= ml->exp_nr_oversampled_rot;
        int itrans = index % ml->exp_nr_trans;
        index /= ml->exp_nr_trans;
        int irot = index % ml->exp_nr_rot;
        index /= ml->exp_nr_rot;
        int iclass = index % h->nr_class;
        index /= h->nr_class;
        int image = index;

        int my_trans = over_trans + ml->exp_nr_oversampled_trans * itrans;
        int my_rot = over_rot + ml->exp_nr_oversampled_rot * irot;
        index = my_trans + h->nr_trans*(my_rot + h->nr_orient*(iclass + h->nr_class*image));
        DIRECT_MULTIDIM_ELEM(ml->exp_Mweight, i) = weight[index];
    }

    vector<uint32_t> min_diff2i;
    d->min_diff2->ToHost(min_diff2i, h->nr_image);
    for (int i = 0; i < h->nr_image; ++i) {
        ml->exp_min_diff2[i] = reverseFlipFloat(min_diff2i[i]);
    }

    ml->convertAllSquaredDifferencesToWeights();

    for (int i = 0; i < mydim; ++i) {
        int index = i;
        int over_trans = index % ml->exp_nr_oversampled_trans;
        index /= ml->exp_nr_oversampled_trans;
        int over_rot = index % ml->exp_nr_oversampled_rot;
        index /= ml->exp_nr_oversampled_rot;
        int itrans = index % ml->exp_nr_trans;
        index /= ml->exp_nr_trans;
        int irot = index % ml->exp_nr_rot;
        index /= ml->exp_nr_rot;
        int iclass = index % h->nr_class;
        index /= h->nr_class;
        int image = index;

        int my_trans = over_trans + ml->exp_nr_oversampled_trans * itrans;
        int my_rot = over_rot + ml->exp_nr_oversampled_rot * irot;
        index = my_trans + h->nr_trans*(my_rot + h->nr_orient*(iclass + h->nr_class*image));
        weight[index] = DIRECT_MULTIDIM_ELEM(ml->exp_Mweight, i);
    }

    d->weight->FromHost(weight, mydim);
    d->weight_sig->FromHost(vector<float>(ml->exp_significant_weight.begin(), ml->exp_significant_weight.end()), h->nr_image);
    d->weight_sum->FromHost(vector<float>(ml->exp_sum_weight.begin(), ml->exp_sum_weight.end()), h->nr_image);
}

void ExpectationCudaSolver::sumWeights()
{
    fprintf(stderr, "ExpectationCudaSolver::sumWeights\n");
    assert(ml->do_map);
    assert(!ml->do_skip_maximization);
    assert(ml->exp_iseries == 0);

    int nr_particle = h->nr_image;

    std::vector< Matrix1D<double> > oversampled_orientations, oversampled_translations;
    std::vector<MultidimArray<double> > wsum_pdf_direction;
    std::vector<double> sumw_group;
    std::vector<double> wsum_pdf_class;
    std::vector<double> wsum_prior_offsetx_class;
    std::vector<double> wsum_prior_offsety_class;
    std::vector<double> max_weight;
    double wsum_sigma2_offset;
    MultidimArray<double> metadata, zeroArray;
    vector<MultidimArray<Complex>> fimg;
    vector<MultidimArray<double>> fweight;

    zeroArray.initZeros(ml->sampling.NrDirections(0, true));
    wsum_pdf_direction.resize(h->nr_class);
    for (int n = 0; n < h->nr_class; n++)
        wsum_pdf_direction[n] = zeroArray;
    sumw_group.resize(h->nr_group, 0.);
    wsum_pdf_class.resize(h->nr_class, 0.);
    if (ml->mymodel.ref_dim == 2) {
        wsum_prior_offsetx_class.resize(h->nr_class, 0.);
        wsum_prior_offsety_class.resize(h->nr_class, 0.);
    }
    max_weight.resize(nr_particle, 0.);
    metadata.initZeros(ml->exp_metadata);
    wsum_sigma2_offset = 0.;
    fimg.resize(h->nr_orient);
    fweight.resize(h->nr_orient);
    for (int i = 0; i < h->nr_orient; ++i) {
        fimg[i].initZeros(ml->exp_Fimgs[0]);
        fweight[i].initZeros(ml->exp_Fimgs[0]);
    }

    d->wsum_sigma2_noise->FromHost(vector<float>(h->nr_group*h->nr_spectrum, 0.f));
    d->wsum_norm_correction->FromHost(vector<float>(h->nr_image, 0.f));
    d->wsum_scale_correction_XA->FromHost(vector<float>(h->nr_image*h->nr_spectrum, 0.f));
    d->wsum_scale_correction_AA->FromHost(vector<float>(h->nr_image*h->nr_spectrum, 0.f));

    /* invSigma2[,0] should not be zero.  Bug?
    kComputeInvSigma2<THREAD_PER_BLOCK><<<h->nr_image, THREAD_PER_BLOCK>>>(
        h->page_size,
        h->image_size,
        h->nr_spectrum,
        ml->sigma2_fudge,
        true,
        *d->resolution,
        *d->image_to_group,
        *d->sigma2_noise,
        *d->invSigma2
    );
    */
#ifdef CHECK_RESULT
    {
        vector<float> invSigma2;
        d->invSigma2->ToHost(invSigma2, h->nr_image * h->page_size);
        int i = 0;
        int nr_error = 0;
        int nr_total = 0;
        for (auto &img : ml->exp_local_Minvsigma2s) {
            FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img) {
                float error = relative_diff(0.001 + abs(DIRECT_MULTIDIM_ELEM(img, n)),
                                            0.001 + abs(invSigma2[i * h->page_size + n]));
                if (error > 0.1) {
                    fprintf(stderr, "    invSigma2 error %.8f vs %.8f at (%d, %d)\n",
                            DIRECT_MULTIDIM_ELEM(img, n),
                            invSigma2[i * h->page_size + n], i, (int)n);
                    ++nr_error;
                }
                ++nr_total;
            }
            ++i;
        }
        float error_rate = 100.f * float(nr_error) / nr_total;
        printf("    invSigma2's error rate is %.5f%%\n", error_rate);
    }

    float fimg_error = 0.f;
    float fweight_error = 0.f;
    int fimg_count = 0;
    int fweight_count = 0;
#endif

    for (int iclass = ml->iclass_min; iclass <= ml->iclass_max; ++iclass) {
        vector< MultidimArray<Complex> > fref;
        fref.resize(h->nr_orient);

        int iorient_total =0;
        for (long int iorient = 0; iorient < ml->exp_nr_rot; iorient++) {
            long int idir = iorient / ml->exp_nr_psi;
            long int ipsi = iorient % ml->exp_nr_psi;
            ml->sampling.getOrientations(idir, ipsi, ml->adaptive_oversampling, oversampled_orientations);

            for (long int iover_rot = 0; iover_rot < ml->exp_nr_oversampled_rot; iover_rot++) {
                double rot =  XX(oversampled_orientations[iover_rot]);
                double tilt = YY(oversampled_orientations[iover_rot]);
                double psi =  ZZ(oversampled_orientations[iover_rot]);
                Matrix2D<double> A;
                Euler_angles2matrix(rot, tilt, psi, A);
                A = A.inv();
                ml->mymodel.PPref[iclass].get2DFourierTransform(fref[iorient_total], A, IS_INV);
                iorient_total++;
            }
        }

        d->data_vs_prior_class->FromHost(vector<float>(ml->mymodel.data_vs_prior_class[iclass].data,
                                                       ml->mymodel.data_vs_prior_class[iclass].data + ml->mymodel.data_vs_prior_class[iclass].nzyxdim));
        d->copyArrayToDeviceComplex(fref, d->fref_r, d->fref_i, true);
        kClearWeight<THREAD_PER_BLOCK>
                <<< dim3(h->nr_orient), THREAD_PER_BLOCK >>>(
            h->page_size,
            h->image_size,
            *d->fimg_sum_r,
            *d->fimg_sum_i,
            *d->fweight
        );

        kSumWeight<THREAD_PER_BLOCK>
                <<< dim3(h->nr_image, h->nr_orient), THREAD_PER_BLOCK >>>(
            h->page_size,
            h->image_size,
            h->nr_spectrum,
            h->nr_trans,
            h->nr_orient,
            h->nr_class,
            iclass,
            ml->do_ctf_correction,
            ml->refs_are_ctf_corrected,
            ml->do_scale_correction,
            *d->weight,
            *d->weight_sum,
            *d->weight_sig,
            *d->image_to_group,
            *d->resolution,
            *d->data_vs_prior_class,
            *d->fref_r,
            *d->fref_i,
            *d->fctf,
            *d->fshift_r,
            *d->fshift_i,
            *d->fshift_nomask_r,
            *d->fshift_nomask_i,
            *d->scale,
            *d->invSigma2,

            *d->wsum_sigma2_noise,
            *d->wsum_norm_correction,
            *d->wsum_scale_correction_XA,
            *d->wsum_scale_correction_AA,
            *d->fimg_sum_r,
            *d->fimg_sum_i,
            *d->fweight
        );

        d->copyArrayFromDeviceDouble(d->fweight, &fweight, h->page_size);
        d->copyArrayFromDeviceComplex(d->fimg_sum_r, d->fimg_sum_i, &fimg, h->page_size);

#ifdef CHECK_RESULT
        for (int i = 0; i < h->nr_orient; ++i) {
            int index = i + h->nr_orient * iclass;
            if (ml->exp_Fimgs_backup[index].nzyxdim == 0)
                continue;
            fweight_error += check_relative_diff(ml->exp_Fweight_backup[index], fweight[i], 0.001, 0.05, "Fweight", false);
            fimg_error += check_relative_diff(ml->exp_Fimgs_backup[index], fimg[i], 0.001, 0.05, "Fimgs", false);
            ++fweight_count;
            ++fimg_count;
        }
#endif

        iorient_total = 0;
        for (long int iorient = 0; iorient < ml->exp_nr_rot; iorient++) {
            long int iorientclass = iclass * ml->exp_nr_rot + iorient;
            long int idir = iorient / ml->exp_nr_psi;
            long int ipsi = iorient % ml->exp_nr_psi;
            ml->sampling.getOrientations(idir, ipsi, ml->adaptive_oversampling, oversampled_orientations);

            if (!ml->isSignificantAnyParticleAnyTranslation(iorientclass)) {
                std::cerr << "    isNotSignificantAnyParticleAnyTranslation " << iorientclass << std::endl;
                continue;
            }

            for (long int iover_rot = 0; iover_rot < ml->exp_nr_oversampled_rot; iover_rot++, iorient_total++) {
                double rot =  XX(oversampled_orientations[iover_rot]);
                double tilt = YY(oversampled_orientations[iover_rot]);
                double psi =  ZZ(oversampled_orientations[iover_rot]);
                Matrix2D<double> A;
                Euler_angles2matrix(rot, tilt, psi, A);
                A = A.inv();
                ml->wsum_model.BPref[iclass].set2DFourierTransform(fimg[iorient_total], A, IS_INV, &fweight[iorient_total]);

                for (long int ori_part_id = ml->exp_my_first_ori_particle, ipart = 0; ori_part_id <= ml->exp_my_last_ori_particle; ori_part_id++) {
                    for (long int i = 0; i < ml->mydata.ori_particles[ori_part_id].particles_id.size(); i++, ipart++) {
                        long int part_id = ml->mydata.ori_particles[ori_part_id].particles_id[i];
                        long int my_image_no = ml->exp_starting_image_no[ipart] + ml->exp_iseries;
                        int group_id = ml->mydata.getGroupId(part_id, ml->exp_iseries);

                        long int ihidden = iorientclass * ml->exp_nr_trans;
                        for (long int itrans = 0; itrans < ml->exp_nr_trans; itrans++, ihidden++) {
                            ml->sampling.getTranslations(itrans, ml->adaptive_oversampling, oversampled_translations);
                            for (long int iover_trans = 0; iover_trans < ml->exp_nr_oversampled_trans; iover_trans++) {
                                long int ihidden_over = ihidden * ml->exp_nr_oversampled_trans * ml->exp_nr_oversampled_rot +
                                    iover_rot * ml->exp_nr_oversampled_trans + iover_trans;
                                double weight = DIRECT_A2D_ELEM(ml->exp_Mweight, ipart, ihidden_over);

                                if (weight >= ml->exp_significant_weight[ipart]) {
                                    weight /= ml->exp_sum_weight[ipart];

                                    sumw_group[group_id] += weight;

                                    /*
                                    static int count = 0;
                                    if (ml->exp_my_first_ori_particle == 8) {
                                        printf("weigh3 : %.8f %6d %6d %6d %6d %6d\n", weight, ipart, iorient, iover_rot, itrans, iover_trans, group_id);
                                    }
                                    */

                                    wsum_pdf_class[iclass] += weight;
                                    if (ml->mymodel.ref_dim ==2) {
                                        wsum_prior_offsetx_class[iclass] += weight * XX(ml->exp_old_offset[my_image_no] + oversampled_translations[iover_trans]);
                                        wsum_prior_offsety_class[iclass] += weight * YY(ml->exp_old_offset[my_image_no] + oversampled_translations[iover_trans]);
                                        wsum_sigma2_offset += weight * ((ml->mymodel.prior_offset_class[iclass] - ml->exp_old_offset[my_image_no] - oversampled_translations[iover_trans]).sum2());
                                    } else {
                                        wsum_sigma2_offset += weight * ((ml->exp_prior[my_image_no] - ml->exp_old_offset[my_image_no] - oversampled_translations[iover_trans]).sum2());
                                    }

                                    if (ml->mymodel.orientational_prior_mode == NOPRIOR) {
                                        DIRECT_MULTIDIM_ELEM(wsum_pdf_direction[iclass], idir) += weight;
                                    } else {
                                        long int mydir = ml->sampling.getDirectionNumberAlsoZeroPrior(idir);
                                        DIRECT_MULTIDIM_ELEM(wsum_pdf_direction[iclass], mydir) += weight;
                                    }

                                    if (weight > max_weight[ipart]) {
                                        max_weight[ipart] = weight;
                                        Euler_matrix2angles(A.inv(), rot, tilt, psi);
                                        DIRECT_A2D_ELEM(metadata, my_image_no, METADATA_ROT) = rot;
                                        DIRECT_A2D_ELEM(metadata, my_image_no, METADATA_TILT) = tilt;
                                        DIRECT_A2D_ELEM(metadata, my_image_no, METADATA_PSI) = psi;
                                        DIRECT_A2D_ELEM(metadata, my_image_no, METADATA_XOFF) = XX(ml->exp_old_offset[my_image_no]) + XX(oversampled_translations[iover_trans]);
                                        DIRECT_A2D_ELEM(metadata, my_image_no, METADATA_YOFF) = YY(ml->exp_old_offset[my_image_no]) + YY(oversampled_translations[iover_trans]);
                                        DIRECT_A2D_ELEM(metadata, my_image_no, METADATA_CLASS) = (double)iclass + 1;
                                        DIRECT_A2D_ELEM(metadata, my_image_no, METADATA_PMAX) = max_weight[ipart];
                                    }

                                }
                            }
                        }
                    }
                }
            }
        }
    }

    vector<float> wsum_norm_correction;
    std::vector<MultidimArray<double> > wsum_sigma2_noise;
    std::vector<MultidimArray<double> > wsum_scale_correction_XA;
    std::vector<MultidimArray<double> > wsum_scale_correction_AA;

    zeroArray.initZeros(h->nr_spectrum);
    wsum_sigma2_noise.resize(h->nr_group);
    for (int n = 0; n < h->nr_group ; n++)
        wsum_sigma2_noise[n] = zeroArray;
    wsum_scale_correction_XA.resize(nr_particle);
    wsum_scale_correction_AA.resize(nr_particle);
    for (int n = 0; n < nr_particle; n++) {
        wsum_scale_correction_XA[n] = zeroArray;
        wsum_scale_correction_AA[n] = zeroArray;
    }

    d->wsum_norm_correction->ToHost(wsum_norm_correction, nr_particle);
    d->copyArrayFromDeviceDouble(d->wsum_sigma2_noise, &wsum_sigma2_noise, h->nr_spectrum);
    d->copyArrayFromDeviceDouble(d->wsum_scale_correction_XA, &wsum_scale_correction_XA, h->nr_spectrum);
    d->copyArrayFromDeviceDouble(d->wsum_scale_correction_AA, &wsum_scale_correction_AA, h->nr_spectrum);

#ifdef CHECK_RESULT
    fprintf(stderr, "    fimg has error rate: %.5f%%\n", 100.f * fimg_error / fimg_count);
    fprintf(stderr, "    fweight has error rate: %.5f%%\n", 100.f * fweight_error / fimg_count);

    if (ml->do_scale_correction) {
        assert(ml->exp_wsum_scale_correction_XA_backup.size() == nr_particle);
        assert(wsum_scale_correction_XA.size() == nr_particle);
        assert(wsum_scale_correction_AA.size() == nr_particle);
        float XA_error = 0.f;
        float AA_error = 0.f;
        for (int n = 0; n < nr_particle; n++) {
            XA_error += check_relative_diff(ml->exp_wsum_scale_correction_XA_backup[n], wsum_scale_correction_XA[n], 0.001, 0.05, "wsum_scale_correction_XA", false);
            AA_error += check_relative_diff(ml->exp_wsum_scale_correction_AA_backup[n], wsum_scale_correction_AA[n], 0.001, 0.05, "wsum_scale_correction_AA", false);
        }
        fprintf(stderr, "    wsum_scale_correction_XA has error rate: %.5f%%\n", 100.f * XA_error / nr_particle);
        fprintf(stderr, "    wsum_scale_correction_AA has error rate: %.5f%%\n", 100.f * AA_error / nr_particle);
    }

    assert(ml->exp_wsum_norm_correction_backup.size() == nr_particle);
    assert(wsum_norm_correction.size() == nr_particle);
    int norm_error = 0;
    for (int n = 0; n < nr_particle; n++) {
        norm_error += !check_relative_diff(ml->exp_wsum_norm_correction_backup[n], wsum_norm_correction[n], 0.001, 0.5, "exp_wsum_norm_correction", false);
    }
    fprintf(stderr, "    wsum_norm_correction has error rate: %.5f%%\n", 100.f * float(norm_error) / nr_particle);
    assert(ml->wsum_model.sigma2_noise.size() == ml->mymodel.nr_groups);
    assert(wsum_sigma2_noise.size() == ml->mymodel.nr_groups);
    assert(ml->wsum_model.sumw_group.size() == ml->mymodel.nr_groups);
    assert(sumw_group.size() == ml->mymodel.nr_groups);
    for (int n = 0; n < ml->mymodel.nr_groups; n++) {
        check_relative_diff(ml->exp_wsum_sigma2_noise_backup[n], wsum_sigma2_noise[n], 0.001, 0.05, "sigma2_noise");
        check_relative_diff(ml->exp_sumw_group_backup[n], sumw_group[n], 0.0001, 0.05, "sumw_group");
    }
    for (int n = 0; n < ml->mymodel.nr_classes; n++) {
        check_relative_diff(ml->exp_wsum_pdf_class_backup[n], wsum_pdf_class[n], 0.0001, 0.05, "pdf_class");
        if (ml->mymodel.ref_dim == 2) {
            check_relative_diff(ml->exp_wsum_prior_offsetx_class_backup[n], wsum_prior_offsetx_class[n], 0.0001, 0.05, "prior_offsetx_class");
            check_relative_diff(ml->exp_wsum_prior_offsety_class_backup[n], wsum_prior_offsety_class[n], 0.0001, 0.05, "prior_offsety_class");
        }
        check_relative_diff(ml->exp_wsum_pdf_direction_backup[n], wsum_pdf_direction[n], 0.0001, 0.05, "pdf_direction");
    }
    check_relative_diff(ml->exp_wsum_sigma2_offset_backup, wsum_sigma2_offset, 0.0001, 0.05, "wsum_sigma2_offset");

    assert(max_weight.size() == nr_particle);
    for (int n = 0; n < nr_particle; n++) {
        check_relative_diff(ml->exp_max_weight[n], max_weight[n], 0.0001, 0.05, "max_weight");
    }
#endif

    if (ml->do_scale_correction) {
        for (int n = 0; n < nr_particle; n++) {
            ml->exp_wsum_scale_correction_XA[n] += wsum_scale_correction_XA[n];
            ml->exp_wsum_scale_correction_AA[n] += wsum_scale_correction_AA[n];
        }
    }
    for (int n = 0; n < nr_particle; n++) {
        ml->exp_wsum_norm_correction[n] += wsum_norm_correction[n];
    }
    for (int n = 0; n < ml->mymodel.nr_groups; n++) {
        ml->wsum_model.sigma2_noise[n] += wsum_sigma2_noise[n];
        ml->wsum_model.sumw_group[n] += sumw_group[n];
    }
    for (int n = 0; n < ml->mymodel.nr_classes; n++) {
        ml->wsum_model.pdf_class[n] += wsum_pdf_class[n];
        if (ml->mymodel.ref_dim == 2) {
            XX(ml->wsum_model.prior_offset_class[n]) += wsum_prior_offsetx_class[n];
            YY(ml->wsum_model.prior_offset_class[n]) += wsum_prior_offsety_class[n];
        }
        ml->wsum_model.pdf_direction[n] += wsum_pdf_direction[n];
    }
    ml->wsum_model.sigma2_offset += wsum_sigma2_offset;
    for (int n = 0; n < nr_particle; n++) {
        if (max_weight[n] >= ml->exp_max_weight[n]) {
            ml->exp_max_weight[n] = max_weight[n];
            long int my_image_no = ml->exp_starting_image_no[n] + ml->exp_iseries;
            DIRECT_A2D_ELEM(ml->exp_metadata, my_image_no, METADATA_ROT)  = DIRECT_A2D_ELEM(metadata, my_image_no, METADATA_ROT);
            DIRECT_A2D_ELEM(ml->exp_metadata, my_image_no, METADATA_TILT) = DIRECT_A2D_ELEM(metadata, my_image_no, METADATA_TILT);
            DIRECT_A2D_ELEM(ml->exp_metadata, my_image_no, METADATA_PSI)  = DIRECT_A2D_ELEM(metadata, my_image_no, METADATA_PSI);
            DIRECT_A2D_ELEM(ml->exp_metadata, my_image_no, METADATA_XOFF) = DIRECT_A2D_ELEM(metadata, my_image_no, METADATA_XOFF);
            DIRECT_A2D_ELEM(ml->exp_metadata, my_image_no, METADATA_YOFF) = DIRECT_A2D_ELEM(metadata, my_image_no, METADATA_YOFF);
            DIRECT_A2D_ELEM(ml->exp_metadata, my_image_no, METADATA_CLASS)= DIRECT_A2D_ELEM(metadata, my_image_no, METADATA_CLASS);
            DIRECT_A2D_ELEM(ml->exp_metadata, my_image_no, METADATA_PMAX) = DIRECT_A2D_ELEM(metadata, my_image_no, METADATA_PMAX);
        }
    }
}
