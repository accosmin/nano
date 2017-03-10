#include "utest.h"
#include "function.h"
#include "math/epsilon.h"
#include "layers/conv3d_naive.h"
#include "layers/conv3d_toeplitz.h"

using namespace nano;

template <typename top>
struct wrt_params_function_t final : public function_t
{
        wrt_params_function_t(const top& op) : function_t("conv3d",
                op.params().psize(), op.params().psize(), op.params().psize(), convexity::no, 1e+6),
                m_op(op),
                m_idata(op.params().make_idata()),
                m_kdata(op.params().make_kdata()),
                m_bdata(op.params().make_bdata()),
                m_odata(op.params().make_odata())
        {
                m_bdata.setRandom();
                m_idata.vector().setRandom();
                m_kdata.vector().setRandom();
                m_odata.vector().setRandom();
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
                m_kdata = map_tensor(x.data(), m_kdata.dims());
                m_bdata = map_vector(x.data() + m_kdata.size(), m_bdata.size());
                m_op.output(m_idata, m_kdata, m_bdata, m_odata);
                if (gx)
                {
                        gx->resize(x.size());
                        m_op.gparam(m_idata, map_tensor(gx->data(), m_kdata.dims()),
                                map_vector(gx->data() + m_kdata.size(), m_bdata.size()), m_odata);
                }
                return m_odata.array().square().sum() / 2;
        }

        mutable top             m_op;
        tensor3d_t              m_idata;
        mutable tensor4d_t      m_kdata;
        mutable vector_t        m_bdata;
        mutable tensor3d_t      m_odata;
};

template <typename top>
struct wrt_inputs_function_t final : public function_t
{
        wrt_inputs_function_t(const top& op) : function_t("conv3d",
                op.params().isize(), op.params().isize(), op.params().isize(), convexity::no, 1e+6),
                m_op(op),
                m_idata(op.params().make_idata()),
                m_kdata(op.params().make_kdata()),
                m_bdata(op.params().make_bdata()),
                m_odata(op.params().make_odata())
        {
                m_bdata.setRandom();
                m_idata.vector().setRandom();
                m_kdata.vector().setRandom();
                m_odata.vector().setRandom();
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
                m_idata = map_tensor(x.data(), m_idata.dims());
                m_op.output(m_idata, m_kdata, m_bdata, m_odata);
                if (gx)
                {
                        gx->resize(x.size());
                        m_op.ginput(map_tensor(gx->data(), m_idata.dims()), m_kdata, m_bdata, m_odata);
                }
                return m_odata.array().square().sum() / 2;
        }

        mutable top             m_op;
        mutable tensor3d_t      m_idata;
        tensor4d_t              m_kdata;
        vector_t                m_bdata;
        mutable tensor3d_t      m_odata;
};

auto make_default_params()
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

        return conv3d_params_t{imaps, irows, icols, omaps, kconn, krows, kcols, kdrow, kdcol};
}

template <typename top>
auto make_wrt_params_function(const conv3d_params_t& params)
{
        return wrt_params_function_t<top>(top{params});
}

template <typename top>
auto make_wrt_inputs_function(const conv3d_params_t& params)
{
        return wrt_inputs_function_t<top>(top{params});
}

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

NANO_CASE(gparam_accuracy)
{
        const auto params = make_default_params();
        NANO_REQUIRE(params.valid());

        const auto pfunct = make_wrt_params_function<conv3d_naive_t>(params);
        for (int i = 0; i < 16; ++ i)
        {
                vector_t px(pfunct.size()); px.setRandom();
                NANO_CHECK_LESS(pfunct.grad_accuracy(px), epsilon2<scalar_t>());
        }
}

NANO_CASE(ginput_accuracy)
{
        const auto params = make_default_params();
        NANO_REQUIRE(params.valid());

        const auto ifunct = make_wrt_inputs_function<conv3d_naive_t>(params);
        for (int i = 0; i < 16; ++ i)
        {
                vector_t ix(ifunct.size()); ix.setRandom();
                NANO_CHECK_LESS(ifunct.grad_accuracy(ix), epsilon2<scalar_t>());
        }
}

NANO_CASE(naive_vs_toeplitz)
{
        const auto params = make_default_params();
        NANO_REQUIRE(params.valid());

        const auto op_naive = conv3d_naive_t{params};
        const auto op_toeplitz = conv3d_toeplitz_t{params};

        for (int i = 0; i < 16; ++ i)
        {
                auto bdata = params.make_bdata(); bdata.setRandom();
                auto idata = params.make_idata(); idata.vector().setRandom();
                auto kdata = params.make_kdata(); kdata.vector().setRandom();
                auto odata = params.make_odata(); odata.vector().setRandom();

                auto bdatax = params.make_bdata(); bdatax.setRandom();
                auto idatax = params.make_idata(); idatax.vector().setRandom();
                auto kdatax = params.make_kdata(); kdatax.vector().setRandom();
                auto odatax = params.make_odata(); odatax.vector().setRandom();

                op_naive.output(idata, kdata, bdata, odata);
                op_toeplitz.output(idata, kdata, bdata, odatax);
                NANO_CHECK_EIGEN_CLOSE(odata.array(), odatax.array(), 10 * epsilon0<scalar_t>());

                // NB: ::gparam() before ::ginput()!
                op_naive.gparam(idata, kdata, bdata, odata);
                op_toeplitz.gparam(idata, kdatax, bdatax, odata);
                NANO_CHECK_EIGEN_CLOSE(kdata.array(), kdatax.array(), 10 * epsilon0<scalar_t>());
                NANO_CHECK_EIGEN_CLOSE(bdata.array(), bdatax.array(), 10 * epsilon0<scalar_t>());

                op_naive.ginput(idata, kdata, bdata, odata);
                op_toeplitz.ginput(idatax, kdata, bdata, odata);
                NANO_CHECK_EIGEN_CLOSE(idata.array(), idatax.array(), epsilon1<scalar_t>());
        }
}

NANO_END_MODULE()
