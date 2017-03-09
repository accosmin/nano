#include "utest.h"
#include "math/epsilon.h"
#include "layers/conv3d.h"
#include "layers/conv3d_naive.h"

using namespace nano;

NANO_BEGIN_MODULE(test_conv3d)

NANO_CASE(params_valid)
{
        const tensor_size_t imaps = 8;
        const tensor_size_t irows = 11;
        const tensor_size_t icols = 15;
        const tensor_size_t omaps = 4;
        const tensor_size_t kconn = 2;
        const tensor_size_t krows = 3;
        const tensor_size_t kcols = 5;
        const tensor_size_t kdrow = 2;
        const tensor_size_t kdcol = 1;

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
        const tensor_size_t imaps = 8;
        const tensor_size_t irows = 11;
        const tensor_size_t icols = 15;
        const tensor_size_t omaps = 6;
        const tensor_size_t kconn = 3;
        const tensor_size_t krows = 13;
        const tensor_size_t kcols = 5;
        const tensor_size_t kdrow = 2;
        const tensor_size_t kdcol = 7;

        const auto params = conv3d_params_t{imaps, irows, icols, omaps, kconn, krows, kcols, kdrow, kdcol};

        NANO_CHECK(!params.valid_kernel());
        NANO_CHECK(!params.valid_connectivity());
        NANO_CHECK(!params.valid());
}

NANO_END_MODULE()
