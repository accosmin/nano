#include "utest.h"
#include "math/epsilon.h"
#include "layers/conv_utils.h"
#include "layers/conv_params.h"

using namespace nano;

NANO_BEGIN_MODULE(test_conv)

NANO_CASE(img2col_vs_img2colx)
{
        const auto irows = 27;
        const auto icols = 25;

        matrix_t imat(irows, icols); imat.setRandom();

        for (auto krows = 1; krows <= 3; ++ krows)
        {
                for (auto kcols = 1; kcols <= 3; ++ kcols)
                {
                        for (auto drows = 1; drows <= 3; ++ drows)
                        {
                                for (auto dcols = 1; dcols <= 3; ++ dcols)
                                {
                                        const auto params = conv_params_t{1, irows, icols, 1, 1, krows, kcols, drows, dcols};

                                        const auto orows = params.orows();
                                        const auto ocols = params.ocols();

                                        matrix_t omat0(krows * kcols, orows * ocols); omat0.setRandom();
                                        matrix_t omatx(krows * kcols, orows * ocols); omatx.setRandom();

                                        img2col0(imat, orows, ocols, krows, kcols, drows, dcols, omat0);
                                        img2colx(imat, orows, ocols, krows, kcols, drows, dcols, omatx);

                                        NANO_CHECK_EIGEN_CLOSE(omat0.array(), omatx.array(), epsilon0<scalar_t>());
                                }
                        }
                }
        }
}

NANO_END_MODULE()
