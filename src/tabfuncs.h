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

#ifndef TABFUNCS_H_
#define TABFUNCS_H_

#include "multidim_array.h"
#include "funcs.h"

// Class to tabulate some functions
class TabFunction
{

protected:
    MultidimArray<double> tabulatedValues;
    double sampling;
    double inv_sampling;

public:
    // Empty constructor
    TabFunction() {}

    // Destructor
    virtual ~TabFunction()
    {
        tabulatedValues.clear();
    }

    /** Copy constructor
     *
     * The created TabFunction is a perfect copy of the input array but with a
     * different memory assignment.
     */
    TabFunction(const TabFunction& op)
    {
        tabulatedValues.clear();
        *this = op;
    }

    /** Assignment.
     *
     * You can build as complex assignment expressions as you like. Multiple
     * assignment is allowed.
     */
    TabFunction& operator=(const TabFunction& op)
    {
        if (&op != this)
        {
            // Projector stuff (is this necessary in C++?)
            tabulatedValues = op.tabulatedValues;
            sampling = op.sampling;
        }
        return *this;
    }


};

class TabSine : public TabFunction
{
public:
    const int NUM_ELEM = 4096;

    // Empty constructor
    TabSine() {}

    // Constructor (with parameters)
    void initialise() {
        sampling = 2 * PI / (double) NUM_ELEM;
        inv_sampling = 1 / sampling;
        TabSine::fillTable();
    }

    //Pre-calculate table values
    void fillTable() {
        tabulatedValues.resize(NUM_ELEM);
        for (int i = 0; i < NUM_ELEM; i++)
        {
            double xx = (double) i * sampling;
            tabulatedValues(i) = sin(xx);
        }
    }

    // Value access
    double operator()(double val) const
    {
        int idx = (int)(ABS(val) * inv_sampling);
        double retval = DIRECT_A1D_ELEM(tabulatedValues, idx % NUM_ELEM);
        return (val < 0 ) ? -retval : retval;
    }
};

class TabCosine : public TabFunction
{
public:
    const int NUM_ELEM = 4096;

    // Empty constructor
    TabCosine() {}

    void initialise() {
        sampling = 2 * PI / (double) NUM_ELEM;
        inv_sampling = 1 / sampling;
        TabCosine::fillTable();
    }

    //Pre-calculate table values
    void fillTable() {
        tabulatedValues.resize(NUM_ELEM);
        for (int i = 0; i < NUM_ELEM; i++)
        {
            double xx = (double) i * sampling;
            tabulatedValues(i) = cos(xx);
        }
    }

    // Value access
    double operator()(double val) const
    {
        int idx = (int)(ABS(val) * inv_sampling);
        return DIRECT_A1D_ELEM(tabulatedValues, idx % NUM_ELEM);
    }
};

class TabBlob : public TabFunction
{
private:
    double radius;
    double alpha;
    int order;

public:
    // Empty constructor
    TabBlob() {}

    // Constructor (with parameters)
    void initialise(double _radius, double _alpha, int _order, const int _nr_elem = 10000);

    //Pre-calculate table values
    void fillTable(const int _nr_elem = 5000);

    // Value access
    double operator()(double val) const;

};

class TabFtBlob : public TabFunction
{

private:
    double radius;
    double alpha;
    int order;

public:
    // Empty constructor
    TabFtBlob() {}

    // Constructor (with parameters)
    void initialise(double _radius, double _alpha, int _order, const int _nr_elem = 10000);

    //Pre-calculate table values
    void fillTable(const int _nr_elem = 5000);

    // Value access
    double operator()(double val) const;

};


#endif /* TABFUNCS_H_ */
