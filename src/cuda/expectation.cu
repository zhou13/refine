#include "cuda/utils.cuh"
#include "cuda/expectation.cuh"
#include "cuda/kernel.cuh"
#include "ml_optimiser.h"

const int VT = 128 * 3;

#define CHECK_RESULT

struct HostData {
    int nr_image;
    int nr_trans;
    int nr_rotate;
    int xdim;
    int ydim;
    int image_size;
};

struct DeviceData {
    MGPU_MEM(float) fimg_r;
    MGPU_MEM(float) fimg_i;
    MGPU_MEM(float) fimg_nomask_r;
    MGPU_MEM(float) fimg_nomask_i;

    MGPU_MEM(float) fshift_r;
    MGPU_MEM(float) fshift_i;
    MGPU_MEM(float) fshift_nomask_r;
    MGPU_MEM(float) fshift_nomask_i;

    MGPU_MEM(float) sqrtXi2;

    MGPU_MEM(float) trans_x;
    MGPU_MEM(float) trans_y;

    ContextPtr context;

    void copyArrayToDeviceComplex2D(
            const vector<MultidimArray<Complex>> arr,
            MGPU_MEM(float) d_real,
            MGPU_MEM(float) d_imag)
    {
        int len = arr.size();
        assert(len > 0);
        int x = arr[0].xdim;
        int y = arr[0].xdim;
        vector<float> h_real, h_imag;
        int size = len * x * y;
        h_real.reserve(size);
        h_imag.reserve(size);

        for (int i = 0; i < len; ++i) {
            FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(arr[i]) {
                h_real.push_back(DIRECT_MULTIDIM_ELEM(arr[i], n).real);
                h_imag.push_back(DIRECT_MULTIDIM_ELEM(arr[i], n).imag);
            }
        }

        d_real->FromHost(h_real);
        d_imag->FromHost(h_imag);
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

void ExpectationCudaSolver::initialize()
{
    if (d)
        delete d;
    d = new DeviceData();

    cudaDeviceSynchronize();
    cudaDeviceReset();
    d->context = CreateCudaDevice(0);

    assert(ml->exp_Fimgs.size());
    int nr_image = ml->exp_Fimgs.size();
    int max_xdim = ml->exp_Fimgs[0].xdim;
    int max_ydim = ml->exp_Fimgs[0].ydim;
    int max_nr_trans_samples = ml->exp_nr_trans *
            ml->sampling.oversamplingFactorTranslations(ml->adaptive_oversampling);

    std::cerr << "ExpectationCudaSolver::initialize" << std::endl
              << "    nr_image: " << nr_image << std::endl
              << "    max_xdim: " << max_xdim << std::endl
              << "    max_ydim: " << max_ydim << std::endl
              << "    max_nr_trans_samples" << max_nr_trans_samples << std::endl;

    int image_size = nr_image * max_xdim * max_ydim;
    d->fimg_r = d->context->Malloc<float>(image_size);
    d->fimg_i = d->context->Malloc<float>(image_size);
    d->fimg_nomask_r = d->context->Malloc<float>(image_size);
    d->fimg_nomask_i = d->context->Malloc<float>(image_size);
    d->sqrtXi2 = d->context->Malloc<float>(nr_image);

    d->trans_x = d->context->Malloc<float>(max_nr_trans_samples);
    d->trans_y = d->context->Malloc<float>(max_nr_trans_samples);
    d->fshift_r = d->context->Malloc<float>(image_size * max_nr_trans_samples);
    d->fshift_i = d->context->Malloc<float>(image_size * max_nr_trans_samples);
    d->fshift_nomask_r = d->context->Malloc<float>(image_size * max_nr_trans_samples);
    d->fshift_nomask_i = d->context->Malloc<float>(image_size * max_nr_trans_samples);
}

void ExpectationCudaSolver::copyWindowsedImagesToGPU()
{
    h->xdim = ml->exp_current_image_size / 2 + 1;
    h->ydim = ml->exp_current_image_size;
    h->image_size = h->xdim * h->ydim;

    vector<MultidimArray<Complex>> fimg_win(h->nr_image);
    vector<MultidimArray<Complex>> fimg_nomask_win(h->nr_image);
    for (int i = 0; i < h->nr_image; ++i) {
        windowFourierTransform(ml->exp_Fimgs[i], fimg_win[i], ml->exp_current_image_size);
        windowFourierTransform(ml->exp_Fimgs_nomask[i], fimg_nomask_win[i], ml->exp_current_image_size);
    }
    assert(fimg_win[0].xdim == h->xdim);
    assert(fimg_win[0].ydim == h->ydim);
    d->copyArrayToDeviceComplex2D(fimg_win, d->fimg_r, d->fimg_i);
    d->copyArrayToDeviceComplex2D(fimg_nomask_win, d->fimg_nomask_r, d->fimg_nomask_i);

    bool do_cc = (ml->iter==1 && ml->do_firstiter_cc) || ml->do_always_cc;
    if (do_cc) {
        kReduceSqrtXi2<VT><<<h->nr_image, VT>>>(
            h->image_size,
            *d->fimg_r,
            *d->fimg_i,
            *d->sqrtXi2
        );

#ifdef CHECK_RESULT
        {
            vector<float> _sqrtXi2;
            d->sqrtXi2->ToHost(_sqrtXi2, h->nr_image);
            checkDiffArray(ml->exp_local_sqrtXi2, _sqrtXi2);
            std::cerr << "sqrtXi seems to be correct" << std::endl;
        }
#endif
    }
}

void ExpectationCudaSolver::getShiftedImages()
{
    std::vector<float> translation_x;
    std::vector<float> translation_y;
    for (int i = 0; i < ml->exp_nr_trans; ++i) {
        std::vector<Matrix1D<double>> tmp_trans;
        ml->sampling.getTranslations(i, ml->exp_current_oversampling, tmp_trans);
        for (auto &dir : tmp_trans) {
            translation_x.push_back(XX(dir));
            translation_y.push_back(YY(dir));
        }
    }
    kShiftPhase<VT><<<h->nr_image, VT>>>(
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
    kShiftPhase<VT><<<h->nr_image, VT>>>(
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
        vector<float> std_r, std_i;
        assert(h->nr_trans == ml->exp_local_Fimgs_shifted.size());
        d->fshift_r->ToHost(shifted_r, h->nr_trans);
        d->fshift_i->ToHost(shifted_i, h->nr_trans);
        for (auto &img : ml->exp_local_Fimgs_shifted) {
            FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img) {
                std_r.push_back(DIRECT_MULTIDIM_ELEM(img, n).real);
                std_i.push_back(DIRECT_MULTIDIM_ELEM(img, n).imag);
            }
        }
    }
#endif
}

void ExpectationCudaSolver::getSquareDifference()
{

//            MultidimArray<double> Fctf;
//            windowFourierTransform(exp_Fctfs[my_image_no], Fctf, exp_current_image_size);
//            exp_local_Fctfs[my_image_no] = Fctf;

//            // Get micrograph id (for choosing the right sigma2_noise)
//            int group_id = mydata.getGroupId(part_id, exp_iseries);

//            MultidimArray<double> Minvsigma2;
//            Minvsigma2.initZeros(YSIZE(Fimg), XSIZE(Fimg));
//            MultidimArray<int> * myMresol = (YSIZE(Fimg) == coarse_size) ? &Mresol_coarse : &Mresol_fine;

//#ifdef DEBUG_CHECKSIZES
//            if (!Minvsigma2.sameShape(*myMresol))
//            {
//                std::cerr<< "!Minvsigma2.sameShape(*myMresol)= "<<!Minvsigma2.sameShape(*myMresol) <<std::endl;
//                REPORT_ERROR("!Minvsigma2.sameShape(*myMresol)");
//            }
//#endif
//            // With group_id and relevant size of Fimg, calculate inverse of sigma^2 for relevant parts of Mresol
//            FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(*myMresol)
//            {
//                int ires = DIRECT_MULTIDIM_ELEM(*myMresol, n);
//                // Exclude origin (ires==0) from the Probability-calculation
//                // This way we are invariant to additive factors
//                if (ires > 0)
//                {
//                    DIRECT_MULTIDIM_ELEM(Minvsigma2, n) = 1. / (sigma2_fudge * DIRECT_A1D_ELEM(mymodel.sigma2_noise[group_id], ires));
//                }
//            }

//#ifdef DEBUG_CHECKSIZES
//            if (my_image_no >= exp_local_Minvsigma2s.size())
//            {
//                std::cerr<< "my_image_no= "<<my_image_no<<" exp_local_Minvsigma2s.size()= "<< exp_local_Minvsigma2s.size() <<std::endl;
//                REPORT_ERROR("my_image_no >= exp_local_Minvsigma2s.size()");
//            }
//#endif
//            exp_local_Minvsigma2s[my_image_no] = Minvsigma2;
//        }
//    }

}

void ExpectationCudaSolver::convertSquaredDifferencesToWeights()
{
}
