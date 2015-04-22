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
/***************************************************************************
*
* Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
*
* Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
* 02111-1307  USA
*
*  All comments concerning this program package may be sent to the
*  e-mail address 'xmipp@cnb.csic.es'
***************************************************************************/

#ifndef FUNCS_H
#define FUNCS_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <climits>
#include <algorithm>
#include <vector>
#include <typeinfo>

#include "numerical_recipes.h"
#include "macros.h"
#include "error.h"

#define FILENAMENUMBERLENGTH 6

/** Structure of the points to do least-squares fitting
 */
struct fit_point2D
{
    /// x coordinate
    double x;
    /// y coordinate (assumed to be a function of x)
    double y;
    /// Weight of the point in the Least-Squares problem
    double w;
};

void fitStraightLine(std::vector<fit_point2D> &points, double &slope, double &intercept, double &corr_coeff);

/* ========================================================================= */
/* BLOBS                                                                     */
/* ========================================================================= */
/**@defgroup Blobs Blobs
   @ingroup BasisFunction */
//@{
// Blob structure ----------------------------------------------------------
/** Blob definition.
    The blob is a space limited function (click here for a theoretical
    explanation) which is used as basis function for the ART reconstructions.
    There are several parameters which define the shape of the blob.
    The following structure holds all needed information for a blob, a
    variable can be of this type and it is passed to the different functions
    containing all we need to know about the blob. As a type definition,
    we can work with several kind of blobs in the same program at the same
    time.

    The common way of defining a blob is as follows:
    @code
    struct blobtype blob;                  // Definition of the blob
    blob.radius = 2;                       // Blob radius in voxels
    blob.order  = 2;                       // Order of the Bessel function
    blob.alpha  = 3.6;                     // Smoothness parameter
    @endcode

    Sometimes it is useful to plot any quantity related to the blobs. In the
    following example you have how to plot their Fourier transform in the
    continuous frequency space.

    @code

      int main(int argc, char **argv) {
         struct blobtype blob;                  // Definition of the blob
         blob.radius = 2;                       // Blob radius in voxels
         blob.order  = 2;                       // Order of the Bessel function
         blob.alpha  = textToFloat(argv[1]);    // Smoothness parameter

         double M=blob_Fourier_val (0, blob);
         for (double w=0; w<=2; w += 0.05)
            std::cout << w << " " <<  blob_Fourier_val (w, blob)/M << std::endl;
         return 0;
      }
    @endcode
*/
struct blobtype
{
    /// Spatial radius in Universal System units
    double radius;

    /// Derivation order and Bessel function order
    int   order;

    /// Smoothness parameter
    double alpha;
};

// Blob value --------------------------------------------------------------
/** Blob value.
    This function returns the value of a blob at a given distance from its
    center (in Universal System units). The distance must be
    always positive. Remember that a blob is spherically symmetrycal so
    the only parameter to know the blob value at a point is its distance
    to the center of the blob. It doesn't matter if this distance is
    larger than the real blob spatial extension, in this case the function
    returns 0 as blob value.
    \\ Ex:
    @code
    struct blobtype blob; blob.radius = 2; blob.order = 2; blob.alpha = 3.6;
    Matrix1D<double> v=vectorR3(1,1,1);
    std::cout << "Blob value at (1,1,1) = " << blob_val(v.mod(),blob) << std::endl;
    @endcode */
#define blob_val(r, blob) kaiser_value(r, blob.radius, blob.alpha, blob.order)

/** Function actually computing the blob value. */
double kaiser_value(double r, double a, double alpha, int m);

// Blob projection ---------------------------------------------------------
/** Blob projection.
    This function returns the value of the blob line integral through a
    straight line which passes at a distance 'r' (in Universal System
    units) from the center of the
    blob. Remember that a blob is spherically symmetrycal so
    the only parameter to know this blob line integral is its distance
    to the center of the blob. It doesn't matter if this distance is
    larger than the real blob spatial extension, in this case the function
    returns 0.
    \\ Ex:
    @code
    struct blobtype blob; blob.radius = 2; blob.order = 2; blob.alpha = 3.6;
    Matrix1D<double> v=vectorR3(1,1,1);
    std::cout << "Blob line integral through (1,1,1) = " << blob_proj(v.mod(),blob)
         << std::endl;
    @endcode */
#define blob_proj(r, blob) kaiser_proj(r, blob.radius, blob.alpha, blob.order)

/** Function actually computing the blob projection. */
double kaiser_proj(double r, double a, double alpha, int m);

/** Fourier transform of a blob.
    This function returns the value of the Fourier transform of the blob
    at a given frequency (w). This frequency must be normalized by the
    sampling rate. For instance, for computing the Fourier Transform of
    a blob at 1/Ts (Ts in Amstrongs) you must provide the frequency Tm/Ts,
    where Tm is the sampling rate.

    The Fourier Transform can be computed only for blobs with m=2 or m=0. */
#define blob_Fourier_val(w, blob) \
    kaiser_Fourier_value(w, blob.radius, blob.alpha, blob.order)

/** Function actually computing the blob Fourier transform. */
double kaiser_Fourier_value(double w, double a, double alpha, int m);

/** Formula for a volume integral of a blob (n is the blob dimension) */
#define blob_mass(blob) \
    basvolume(blob.radius, blob.alpha, blob.order,3)

/** Function actually computing the blob integral */
double  basvolume(double a, double alpha, int m, int n);

/** Limit (z->0) of (1/z)^n I_n(z) (needed by basvolume)*/
double in_zeroarg(int n);

/** Limit (z->0) of (1/z)^(n+1/2) I_(n+1/2) (z) (needed by basvolume)*/
double inph_zeroarg(int n);

/** Bessel function I_(n+1/2) (x),  n = 0, 1, 2, ... */
double i_nph(int n, double x);

/** Bessel function I_n (x),  n = 0, 1, 2, ...
 Use ONLY for small values of n */
double i_n(int n, double x);

/** Blob pole.
    This is the normalized frequency at which the blob goes to 0. */
double blob_freq_zero(struct blobtype b);

/** Attenuation of a blob.
    The Fourier transform of the blob at w is the Fourier transform at w=0
    multiplied by the attenuation. This is the value returned. Remind that
    the frequency must be normalized by the sampling rate. Ie, Tm*w(cont) */
double blob_att(double w, struct blobtype b);

/** Number of operations for a blob.
    This is a number proportional to the number of operations that ART
    would need to make a reconstruction with this blob. */
double blob_ops(double w, struct blobtype b);

/** 1D gaussian value
 *
 * This function returns the value of a univariate gaussian function at the
 * point x.
 */
double gaussian1D(double x, double sigma, double mu = 0);

/** 1D t-student value
 *
 * This function returns the value of a univariate t-student function at the
 * point x, and with df degrees of freedom
 */
double tstudent1D(double x, double df, double sigma, double mu = 0);

/** Inverse Cumulative distribution function for a Gaussian
 *
 * This function returns the z of a N(0,1) such that the probability below z is p
 *
 * The function employs an fast approximation to z which is valid up to 1e-4.
 * See http://www.johndcook.com/normal_cdf_inverse.html
 */
double icdf_gauss(double p);

/** Cumulative distribution function for a Gaussian
 *
 * This function returns the value of the CDF of a univariate gaussian function at the
 * point x.
 */
double cdf_gauss(double x);

/** Cumulative distribution function for a t-distribution
 *
 * This function returns the value of the CDF of a univariate t-distribution
 * with k degrees of freedom  at the point t.
 *  Adapted by Sjors from: http://www.alglib.net/specialfunctions/distributions/student.php
 */
double cdf_tstudent(int k, double t);

/** Cumulative distribution function for a Snedecor's F-distribution.
 *
 * This function returns the value of the CDF of a univariate Snedecor's
 * F-distribution
 * with d1, d2 degrees of freedom  at the point x.
 */
double cdf_FSnedecor(int d1, int d2, double x);

/** Inverse Cumulative distribution function for a Snedecor's F-distribution.
 *
 * This function returns the value of the ICDF of a univariate Snedecor's
 * F-distribution
 * with d1, d2 degrees of freedom with probability p, i.e., it returns
 * x such that CDF(d1,d2,x)=p
 */
double icdf_FSnedecor(int d1, int d2, double p);

/** 2D gaussian value
 *
 * This function returns the value of a multivariate (2D) gaussian function at
 * the point (x,y) when the X axis of the gaussian is rotated ang
 * (counter-clockwise) radians (the angle is positive when measured from the
 * universal X to the gaussian X). X and Y are supposed to be independent.
 */
double gaussian2D(double x,
                  double y,
                  double sigmaX,
                  double sigmaY,
                  double ang,
                  double muX = 0,
                  double muY = 0);
//@}

/** Compute the logarithm in base 2
 */
// Does not work with xlc compiler
#ifndef __xlC__
double log2(double value);
#endif
//@}

/** @name Random functions
 *
 * These functions allow you to work in an easier way with the random functions
 * of the Numerical Recipes. Only an uniform and a gaussian random number
 * generators have been implemented. In fact only a uniform generator exists and
 * the gaussian one is based on a call to it. For this reason, if you initialize
 * the gaussian random generator, you are also initialising the uniform one.
 *
 * Here goes an example for uniform random numbers to show how to use this set
 * of functions.
 *
 * @code
 * // Initialise according to the clock
 * randomize_random_generator();
 *
 * // Show 10 random numbers between -1 and 1
 * for (int i=0; i<10; i++)
 *     std::cout << rnd_unif(-1,1) << std::endl;
 * @endcode
 */
//@{
/** Reset uniform random generator to a known point
 *
 * If you initialize the random generator with this function each time, then the
 * same random sequence will be generated
 *
 * @code
 * init_rnd_unif();
 * init_rnd_unif(17891)
 * @endcode
 */
void init_random_generator(int seed = -1);

/** Reset random generator according to the clock.
 *
 * This time the initialisation itself assures a random sequence different each
 * time the program is run. Be careful not to run the program twice within the
 * same second as the initialisation will be the same for both runs.
 */
void randomize_random_generator();

/** Produce a uniform random number between 0 and 1
 *
 * @code
 * std::cout << "This random number should be between 0 and 1: " << rnd_unif()
 * << std::endl;
 * @endcode
 */
float rnd_unif();

/** Produce a uniform random number between a and b
 *
 * @code
 * std::cout << "This random number should be between 0 and 10: " << rnd_unif(0,10)
 * << std::endl;
 * @endcode
 */
float rnd_unif(float a, float b);

/** Produce a t-distributed random number with mean 0 and standard deviation 1 and nu degrees of freedom
 *
 * @code
 * std::cout << "This random number should follow t(0,1) with 3 degrees of freedon: " << rnd_student_t(3.)
 * << std::endl;
 * @endcode
 */
float rnd_student_t(double nu);

/** Produce a gaussian random number with mean a and standard deviation b and nu degrees of freedom
 *
 * @code
 * std::cout << "This random number should follow t(1,4) with 3 d.o.f.: " << rnd_gaus(3,1,2)
 * << std::endl;
 * @endcode
 */
float rnd_student_t(double nu, float a, float b);

/** Produce a gaussian random number with mean 0 and standard deviation 1
 *
 * @code
 * std::cout << "This random number should follow N(0,1): " << rnd_gaus()
 * << std::endl;
 * @endcode
 */
float rnd_gaus();

/** Produce a gaussian random number with mean a and standard deviation b
 *
 * @code
 * std::cout << "This random number should follow N(1,4): " << rnd_gaus(1,2)
 * << std::endl;
 * @endcode
 */
float rnd_gaus(float a, float b);

/** Gaussian area from -x0 to x0
 *
 * By default the gaussian mean is 0 and the gaussian standard deviation is 1.
 * x0 must be positive
 */
float gaus_within_x0(float x0, float mean = 0, float stddev = 1);

/** Gaussian area outisde -x0 to x0
 *
 * By default the gaussian mean is 0 and the gaussian standard deviation is 1.
 * x0 must be positive
 */
float gaus_outside_x0(float x0, float mean = 0, float stddev = 1);

/** Gaussian area from -inf to x0
 *
 * By default the gaussian mean is 0 and the gaussian standard deviation is 1.
 * There is no restriction over the sign of x0
 */
float gaus_up_to_x0(float x0, float mean = 0, float stddev = 1);

/** Gaussian area from x0 to inf
 *
 * By default the gaussian mean is 0 and the gaussian standard deviation is 1.
 * There is no restriction over the sign of x0
 */
float gaus_from_x0(float x0, float mean = 0, float stddev = 1);

/** t0 for a given two-sided probability
 *
 * This function returns t0 such that the student probability outside t0 is
 * equal to p
 */
float student_outside_probb(float p, float degrees_of_freedom);

/** student area from -t0 to t0
 *
 * By default the student mean is 0 and the student standard deviation is 1.
 * t0 must be positive
 */
float student_within_t0(float t0, float degrees_of_freedom);

/** student area outisde -t0 to t0
 *
 * By default the student mean is 0 and the student standard deviation is 1.
 * t0 must be positive
 */
float student_outside_t0(float t0, float degrees_of_freedom);

/** student area from -inf to t0
 *
 * By default the student mean is 0 and the student standard deviation is 1.
 * There is no restriction over the sign of t0
 */
float student_up_to_t0(float t0, float degrees_of_freedom);

/** student area from t0 to inf
 *
 * By default the student mean is 0 and the student standard deviation is 1.
 * There is no restriction over the sign of t0
 */
float student_from_t0(float t0, float degrees_of_freedom);

/** chi2 area from -inf to t0
 *
 * By default the chi2 mean is 0 and the chi2 standard deviation is 1.
 * There is no restriction over the sign of t0
 */
float chi2_up_to_t0(float t0, float degrees_of_freedom);

/** chi2 area from t0 to inf
 *
 * By default the chi2 mean is 0 and the chi2 standard deviation is 1.
 * There is no restriction over the sign of t0
 */
float chi2_from_t0(float t0, float degrees_of_freedom);

/** Produce a log uniform random number between a and b
 *
 * Watch out that the following inequation must hold 0<a<=b.
 *
 * @code
 * std::cout << "This random number should be between 1 and 1000: "
 * << rnd_log(10,1000)<< std::endl;
 * @endcode
 */
float rnd_log(float a, float b);
//@}

/** Conversion little-big endian
 */
void swapbytes(char* v, unsigned long n);


//@}

//@}
#endif


