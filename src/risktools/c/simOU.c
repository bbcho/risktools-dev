#include <stdio.h>
#include <math.h>

// https://stackoverflow.com/questions/5862915/passing-numpy-arrays-to-a-c-function-for-input-and-output
// https://stackoverflow.com/questions/4355524/getting-data-from-ctypes-array-into-numpy
// cc -fPIC -shared -o simOU.so simOU.c

void csimOU(
    double *x,
    const double theta,
    const double *mu,
    const double dt,
    const double sigma,
    const size_t rows,
    const size_t cols)
{
    size_t i;
    size_t j = 0;

    // pre-compute to make faster
    const int ll = rows * cols;
    const double ss = 0.5 * sigma * sigma;
    const double sq = sqrt(dt);

    // input x is a 2D array that has been reshaped to be 1D.
    // Loop through entire 1D array of length rows*cols,
    // skips value every time counter j is reaches row
    // count - effectively a new sim.

    for (i = 1; i < ll; ++i)
    {
        if (j >= cols - 1)
        {
            j = 0;
        }
        else
        {
            j = j + 1;
            x[i] = x[i - 1] + (theta * (mu[i] - x[i - 1]) - ss) * dt + sigma * sq * x[i];
        }
    }
    // return x;
}

// void csimOU(
//     double *x,
//     const double theta,
//     const double *mu,
//     const double dt,
//     const double sigma,
//     const size_t rows,
//     const size_t cols)
// {
//     size_t i;
//     size_t j = 0;
//     int idx;

//     // pre-compute to make faster
//     const double dt_theta = dt * theta;

//     // numpy 2D arrays are stored as 1D array in memory
//     // so this works fine

//     for (i = 0; i < rows; i++)
//     {
//         for (j = 1; j < cols; ++j)
//         {
//             idx = i * cols + j;
//             x[idx] = x[idx - 1] + (mu[idx] - x[idx - 1]) * dt_theta + sigma * x[idx];
//         }
//     }
//     // return x;
// }

// void cfun(const double *indatav, size_t size, double *outdatav)
// {
//     size_t i;
//     for (i = 0; i < size; ++i)
//         outdatav[i] = indatav[i] * 2.0;
// }
