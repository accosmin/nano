#include "utest.h"
#include "math/epsilon.h"
#include "vision/color.h"
#include "vision/gradient.h"
#include "vision/convolve.h"

using namespace nano;

NANO_BEGIN_MODULE(test_vision)

NANO_CASE(convolve)
{
        vector_t kernel(3);
        kernel << -1, 2, -1;

        matrix_t cimage(4, 5);
        cimage << 0, 1, 2, 0, 1,
                  1, 2, 0, 1, 2,
                  2, 0, 1, 2, 0,
                  0, 1, 2, 0, 1;

        convolve(kernel, cimage);

        matrix_t timage(4, 5);
        timage << 0, -3, 6, -3, 0,
                  -3, 9, -9, 0, 3,
                  6, -9, 0, 9, -6,
                  -3, 3, 3, -6, 3;

        NANO_CHECK_EIGEN_CLOSE(cimage, timage, epsilon0<scalar_t>());
}

NANO_CASE(gradientx)
{
        matrix_t image(4, 5);
        image << 0, 1, 2, 0, 1,
                 1, 2, 0, 1, 2,
                 2, 0, 1, 2, 0,
                 0, 1, 2, 0, 1;

        const auto gimage = gradientx(image);

        matrix_t timage(4, 5);
        timage << 1, 2, -1, -1, 1,
                  1, -1, -1, 2, 1,
                  -2, -1, 2, -1, -2,
                  1, 2, -1, -1, 1;

        NANO_CHECK_EIGEN_CLOSE(gimage, timage, epsilon0<scalar_t>());
}

NANO_CASE(gradienty)
{
        matrix_t image(4, 5);
        image << 0, 1, 2, 0, 1,
                 1, 2, 0, 1, 2,
                 2, 0, 1, 2, 0,
                 0, 1, 2, 0, 1;

        const auto gimage = gradienty(image);

        matrix_t timage(4, 5);
        timage << 1, 1, -2, 1, 1,
                  2, -1, -1, 2, -1,
                  -1, -1, 2, -1, -1,
                  -2, 1, 1, -2, 1;

        NANO_CHECK_EIGEN_CLOSE(gimage, timage, epsilon0<scalar_t>());
}

NANO_END_MODULE()
