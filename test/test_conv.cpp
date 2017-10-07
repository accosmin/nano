#include "utest.h"
#include "function.h"
#include "math/epsilon.h"
#include "layers/conv3d.h"
#include "layers/conv4d.h"

using namespace nano;

auto make_default_params(const tensor_size_t kconn = 1, const tensor_size_t drows = 2, const tensor_size_t dcols = 1)
{
        const auto imaps = 6;
        const auto irows = 9;
        const auto icols = 8;
        const auto omaps = 6;
        const auto krows = 2;
        const auto kcols = 3;

        return conv_params_t{imaps, irows, icols, omaps, kconn, krows, kcols, drows, dcols};
}

auto make_buffers(const conv_params_t& params, const tensor_size_t count)
{
        auto bdata = params.make_bdata(); bdata.setRandom();
        auto kdata = params.make_kdata(); kdata.setRandom();
        auto idata = params.make_idata(count); idata.setRandom();
        auto odata = params.make_odata(count); odata.setRandom();
        return std::make_tuple(bdata, idata, kdata, odata);
}

template <typename top>
struct wrt_params_function_t final : public function_t
{
        explicit wrt_params_function_t(const top& op) :
                function_t("conv", op.params().psize(), op.params().psize(), op.params().psize(), convexity::no, 1e+6),
                m_op(op)
        {
                std::tie(m_bdata, m_idata, m_kdata, m_odata) = make_buffers(op.params(), 3);
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
                m_kdata = map_tensor(x.data(), m_kdata.dims());
                m_bdata = map_vector(x.data() + m_kdata.size(), m_bdata.size());
                m_op.output(m_idata, m_kdata, m_bdata, m_odata);
                if (gx)
                {
                        gx->resize(x.size());
                        auto kdata = map_tensor(gx->data(), m_kdata.dims());
                        auto bdata = map_vector(gx->data() + m_kdata.size(), m_bdata.size());
                        m_op.gparam(m_idata, kdata, bdata, m_odata);
                }
                return m_odata.array().square().sum() / 2;
        }

        mutable top             m_op;
        tensor4d_t              m_idata;
        mutable tensor4d_t      m_kdata;
        mutable vector_t        m_bdata;
        mutable tensor4d_t      m_odata;
};

template <typename top>
struct wrt_inputs_function_t final : public function_t
{
        explicit wrt_inputs_function_t(const top& op) :
                function_t("conv", op.params().isize(), op.params().isize(), op.params().isize(), convexity::no, 1e+6),
                m_op(op)
        {
                std::tie(m_bdata, m_idata, m_kdata, m_odata) = make_buffers(op.params(), 1);
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
                m_idata = map_tensor(x.data(), m_idata.dims());
                m_op.output(m_idata, m_kdata, m_bdata, m_odata);
                if (gx)
                {
                        gx->resize(x.size());
                        auto idata = map_tensor(gx->data(), m_idata.dims());
                        m_op.ginput(idata, m_kdata, m_bdata, m_odata);
                }
                return m_odata.array().square().sum() / 2;
        }

        mutable top             m_op;
        mutable tensor4d_t      m_idata;
        tensor4d_t              m_kdata;
        vector_t                m_bdata;
        mutable tensor4d_t      m_odata;
};

template <typename top>
auto make_wrt_params_function(const conv_params_t& params)
{
        return wrt_params_function_t<top>(top{params});
}

template <typename top>
auto make_wrt_inputs_function(const conv_params_t& params)
{
        return wrt_inputs_function_t<top>(top{params});
}

NANO_BEGIN_MODULE(test_conv)

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

        const auto params = conv_params_t{imaps, irows, icols, omaps, kconn, krows, kcols, kdrow, kdcol};

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

        const auto params = conv_params_t{imaps, irows, icols, omaps, kconn, krows, kcols, kdrow, kdcol};

        NANO_CHECK(!params.valid_kernel());
        NANO_CHECK(!params.valid_connectivity());
        NANO_CHECK(!params.valid());
}

NANO_CASE(gparam_accuracy)
{
        for (const auto kconn : {1, 2, 3})
        for (const auto drows : {1, 2})
        for (const auto dcols : {1, 2})
        {
                const auto params = make_default_params(kconn, drows, dcols);
                NANO_REQUIRE(params.valid());

                const auto pfunct = make_wrt_params_function<conv3d_t>(params);

                vector_t px(pfunct.size()); px.setRandom();
                NANO_CHECK_LESS(pfunct.grad_accuracy(px), epsilon1<scalar_t>());
        }
}

NANO_CASE(ginput_accuracy)
{
        for (const auto kconn : {1, 2, 3})
        for (const auto drows : {1, 2})
        for (const auto dcols : {1, 2})
        {
                const auto params = make_default_params(kconn, drows, dcols);
                NANO_REQUIRE(params.valid());

                const auto ifunct = make_wrt_inputs_function<conv3d_t>(params);

                vector_t ix(ifunct.size()); ix.setRandom();
                NANO_CHECK_LESS(ifunct.grad_accuracy(ix), epsilon1<scalar_t>());
        }
}

NANO_CASE(3d_vs_4d_output)
{
        for (const auto kconn : {1, 2, 3})
        for (const auto drows : {1, 2})
        for (const auto dcols : {1, 2})
        {
                const auto params = make_default_params(kconn, drows, dcols);
                NANO_REQUIRE(params.valid());

                auto op3d = conv3d_t{params};
                auto op4d = conv4d_t{params};

                for (int i = 0; i < 3; ++ i)
                {
                        tensor4d_t idata, kdata, odata3, odata4;
                        vector_t bdata;

                        std::tie(bdata, idata, kdata, odata3) = make_buffers(params, i + 2);
                        std::tie(bdata, idata, kdata, odata4) = make_buffers(params, i + 2);

                        NANO_REQUIRE(op3d.output(idata, kdata, bdata, odata3));
                        NANO_REQUIRE(op4d.output(idata, kdata, bdata, odata4));

                        NANO_CHECK_EIGEN_CLOSE(odata3.array(), odata4.array(), epsilon1<scalar_t>());
                }
        }
}

NANO_CASE(3d_vs_4d_gparam)
{
        for (const auto kconn : {1, 2, 3})
        for (const auto drows : {1, 2})
        for (const auto dcols : {1, 2})
        {
                const auto params = make_default_params(kconn, drows, dcols);
                NANO_REQUIRE(params.valid());

                auto op3d = conv3d_t{params};
                auto op4d = conv4d_t{params};

                for (int i = 0; i < 3; ++ i)
                {
                        tensor4d_t idata, kdata3, kdata4, odata;
                        vector_t bdata3, bdata4;

                        std::tie(bdata3, idata, kdata3, odata) = make_buffers(params, i + 2);
                        std::tie(bdata4, idata, kdata4, odata) = make_buffers(params, i + 2);

                        NANO_REQUIRE(op4d.output(idata, kdata4, bdata4, odata));// NB: needed to update the internal buffers!
                        NANO_REQUIRE(op3d.gparam(idata, kdata3, bdata3, odata));
                        NANO_REQUIRE(op4d.gparam(idata, kdata4, bdata4, odata));

                        NANO_CHECK_EIGEN_CLOSE(bdata3.array(), bdata4.array(), epsilon1<scalar_t>());
                        NANO_CHECK_EIGEN_CLOSE(kdata3.array(), kdata4.array(), epsilon1<scalar_t>());
                }
        }
}

NANO_CASE(3d_vs_4d_ginput)
{
        for (const auto kconn : {1, 2, 3})
        for (const auto drows : {1, 2})
        for (const auto dcols : {1, 2})
        {
                const auto params = make_default_params(kconn, drows, dcols);
                NANO_REQUIRE(params.valid());

                auto op3d = conv3d_t{params};
                auto op4d = conv4d_t{params};

                for (int i = 0; i < 3; ++ i)
                {
                        tensor4d_t idata3, idata4, kdata, odata;
                        vector_t bdata;

                        std::tie(bdata, idata3, kdata, odata) = make_buffers(params, i + 2);
                        std::tie(bdata, idata4, kdata, odata) = make_buffers(params, i + 2);

                        NANO_REQUIRE(op4d.output(idata4, kdata, bdata, odata));// NB: needed to update the internal buffers!
                        NANO_REQUIRE(op3d.ginput(idata3, kdata, bdata, odata));
                        NANO_REQUIRE(op4d.ginput(idata4, kdata, bdata, odata));

                        NANO_CHECK_EIGEN_CLOSE(idata3.array(), idata4.array(), epsilon1<scalar_t>());
                }
        }
}

NANO_END_MODULE()
