#include "utest.h"
#include "math/epsilon.h"
#include "layers/conv3d.h"
#include "layers/conv3d_naive.h"

using namespace nano;

NANO_BEGIN_MODULE(test_conv3d)

NANO_CASE(params_valid)
{
        const auto imaps = 8;
        const auto irows = 11;
        const auto icols = 15;
        const auto omaps = 4;
        const auto kconn = 2;
        const auto krows = 3;
        const auto kcols = 5;
        const auto kdrow = 2;
        const auto kdcol = 1;

        const auto params = conv3d_params_t{imaps, irows, icols, omaps, kconn, krows, kcols, kdrow, kdcol};

        NANO_CHECK(params.valid_kernel());
        NANO_CHECK(params.valid_connectivity());
        NANO_CHECK(params.valid());

        NANO_CHECK_EQUAL(params.imaps(), imaps);
        NANO_CHECK_EQUAL(params.irows(), irows);
        NANO_CHECK_EQUAL(params.icols(), icols);

        NANO_CHECK_EQUAL(params.omaps(), omaps);
        NANO_CHECK_EQUAL(params.orows(), (irows - krows + 1) / kdrow);
        NANO_CHECK_EQUAL(params.ocols(), (icols - kcols + 1) / kdcol);

        NANO_CHECK_EQUAL(params.bdims(), omaps);
}

NANO_CASE(params_invalid)
{
        const auto imaps = 8;
        const auto irows = 11;
        const auto icols = 15;
        const auto omaps = 6;
        const auto kconn = 3;
        const auto krows = 13;
        const auto kcols = 5;
        const auto kdrow = 2;
        const auto kdcol = 7;

        const auto params = conv3d_params_t{imaps, irows, icols, omaps, kconn, krows, kcols, kdrow, kdcol};

        NANO_CHECK(!params.valid_kernel());
        NANO_CHECK(!params.valid_connectivity());
        NANO_CHECK(!params.valid());
}

NANO_END_MODULE()
