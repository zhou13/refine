/***************************************************************************
 *
 * Author: "Sjors H.W. Scheres"
 * MRC Laboratory of Molecular Biology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

#ifndef POSTPROCESSING_H_
#define POSTPROCESSING_H_

#include "image.h"
#include "multidim_array.h"
#include "metadata_table.h"
#include <src/fftw.h>
#include <src/timer.h>
#include <src/mask.h>
#include <src/funcs.h>



class Postprocessing
{
public:

    // I/O Parser
    IOParser parser;

    // Verbosity
    int verb;

    // Input & Output rootname
    FileName fn_in, fn_out, fn_I1, fn_I2;

    // Images for the two half-reconstructions and the mask
    Image<double> I1, I2, I1phi, I2phi, Im;

    // Pixel size in Angstroms
    double angpix;

    /////// Masking

    // Perform automated masking (based on a density threshold)
    bool do_auto_mask;

    // Density threshold below which to calculate initial mask seed
    double ini_mask_density_threshold;

    // Number of pixels to extend the mask beyond the initial mask seed
    double extend_ini_mask;

    // Width (in pixels) for soft mask edge
    double width_soft_mask_edge;

    // From the resolution where the FSC drops below this value, randomize the phases in the two maps
    double randomize_fsc_at;

    // Filename for a user-provided mask
    FileName fn_mask;

    //////// Sharpening

    // Filename for the STAR-file with the MTF of the detector
    FileName fn_mtf;

    // Flag to indicate whether to perform Rosenthal&Henderson-2003 like B-factor estimation
    bool do_auto_bfac;

    // Flag to indicate whether we'll do masking
    bool do_mask;

    // Minimum and maximum resolution to use in the fit
    double fit_minres, fit_maxres;

    // User-provided (ad hoc) B-factor
    double adhoc_bfac;

    ///////// Filtering

    // Flag to indicate whether to use FSC-weighting before B-factor sharpening
    bool do_fsc_weighting;

    // Frequency at which to low-pass filter the final map
    double low_pass_freq;

    // Width of raised cosine edge on low-pass filter
    int filter_edge_width;

    // Arrays to store FSC, Guinier curves etc
    MultidimArray<double> fsc_unmasked;
    MultidimArray<double> fsc_masked, fsc_random_masked, fsc_true;
    double global_intercept, global_slope, global_corr_coeff, global_bfactor, global_resol;
    // The Guinier plots
    std::vector<fit_point2D>  guinierin, guinierinvmtf, guinierweighted, guiniersharpen;

public:
    // Read command line arguments
    void read(int argc, char **argv);

    // Print usage instructions
    void usage();

    // Set parameters to some useful defaults
    void clear();

    // Initialise some stuff after reading
    void initialise();

    // Generate the mask (or read it from file)
    // Returns true if masking needs to be done, false otherwise
    bool getMask();

    // Make a mask automatically based on initial density threshold
    void getAutoMask();

    // Divide by MTF and perform FSC-weighted B-factor sharpening, as in Rosenthal and Henderson, 2003
    void sharpenMap();

    // This divides the input FT by the mtf (if fn_mtf !="")
    void divideByMtf(MultidimArray<Complex > &FT);

    // Make a Guinier plot from the Fourier transform of an image
    void makeGuinierPlot(MultidimArray<Complex > &FT, std::vector<fit_point2D> &guinier);

    // Apply sqrt(2FSC/(FSC=1)) weighting prior to B-factor sharpening
    void applyFscWeighting(MultidimArray<Complex > &FT, MultidimArray<double> my_fsc);

    // Output map and STAR files with metadata, also write final resolution to screen
    void writeOutput();

    // Write XML file for EMDB submission
    void writeFscXml(MetaDataTable &MDfsc);

    // General Running
    void run();

};

#endif /* POSTPROCESSING_H_ */
