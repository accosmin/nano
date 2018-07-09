#include "utest.h"
#include "function.h"
#include "core/epsilon.h"
#include "layers/affine3d.h"
#include "layers/affine4d.h"

using namespace nano;

auto make_default_params()
{
        const auto imaps = 2;
        const auto irows = 3;
        const auto icols = 4;
        const auto omaps = 3;
        const auto orows = 4;
        const auto ocols = 5;

        return affine_params_t{imaps, irows, icols, omaps, orows, ocols};
}

auto make_buffers(const affine_params_t& params, const tensor_size_t count)
{
        auto bdata = params.make_bdata(); bdata.setRandom();
        auto wdata = params.make_wdata(); wdata.setRandom();
        auto idata = params.make_idata(count); idata.setRandom();
        auto odata = params.make_odata(count); odata.setRandom();
        return std::make_tuple(idata, wdata, bdata, odata);
}

template <typename top>
struct wrt_params_function_t final : public function_t
{
        explicit wrt_params_function_t(const top& op) :
                function_t("affine", op.params().psize(), op.params().psize(), op.params().psize(), convexity::no, 1e+6),
                m_op(op)
        {
                std::tie(m_idata, m_wdata, m_bdata, m_odata) = make_buffers(op.params(), 3);
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
                m_wdata = map_matrix(x.data(), m_wdata.rows(), m_wdata.cols());
                m_bdata = map_vector(x.data() + m_wdata.size(), m_bdata.size());
                m_op.output(m_idata, m_wdata, m_bdata, m_odata);
                if (gx)
                {
                        gx->resize(x.size());
                        auto wdata = map_matrix(gx->data(), m_wdata.rows(), m_wdata.cols());
                        auto bdata = map_vector(gx->data() + m_wdata.size(), m_bdata.size());
                        m_op.gparam(m_idata, wdata, bdata, m_odata);
                }
                return m_odata.array().square().sum() / 2;
        }

        mutable top             m_op;
        tensor4d_t              m_idata;
        mutable matrix_t        m_wdata;
        mutable vector_t        m_bdata;
        mutable tensor4d_t      m_odata;
};

template <typename top>
struct wrt_inputs_function_t final : public function_t
{
        explicit wrt_inputs_function_t(const top& op) :
                function_t("affine", op.params().isize(), op.params().isize(), op.params().isize(), convexity::no, 1e+6),
                m_op(op)
        {
                std::tie(m_idata, m_wdata, m_bdata, m_odata) = make_buffers(op.params(), 1);
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
                m_idata = map_tensor(x.data(), m_idata.dims());
                m_op.output(m_idata, m_wdata, m_bdata, m_odata);
                if (gx)
                {
                        gx->resize(x.size());
                        auto idata = map_tensor(gx->data(), m_idata.dims());
                        m_op.ginput(idata, m_wdata, m_bdata, m_odata);
                }
                return m_odata.array().square().sum() / 2;
        }

        mutable top             m_op;
        mutable tensor4d_t      m_idata;
        matrix_t                m_wdata;
        vector_t                m_bdata;
        mutable tensor4d_t      m_odata;
};

template <typename top>
auto make_wrt_params_function(const affine_params_t& params)
{
        return wrt_params_function_t<top>(top{params});
}

template <typename top>
auto make_wrt_inputs_function(const affine_params_t& params)
{
        return wrt_inputs_function_t<top>(top{params});
}

NANO_BEGIN_MODULE(test_affine)

NANO_CASE(params)
{
        const auto imaps = 8;
        const auto irows = 11;
        const auto icols = 15;
        const auto omaps = 4;
        const auto orows = 2;
        const auto ocols = 3;

        const auto params = affine_params_t{imaps, irows, icols, omaps, orows, ocols};
        NANO_CHECK(params.valid());

        NANO_CHECK_EQUAL(params.imaps(), imaps);
        NANO_CHECK_EQUAL(params.irows(), irows);
        NANO_CHECK_EQUAL(params.icols(), icols);

        NANO_CHECK_EQUAL(params.omaps(), omaps);
        NANO_CHECK_EQUAL(params.orows(), orows);
        NANO_CHECK_EQUAL(params.ocols(), ocols);
}

NANO_CASE(gparam_accuracy)
{
        const auto params = make_default_params();
        NANO_REQUIRE(params.valid());

        const auto pfunct = make_wrt_params_function<affine3d_t>(params);

        vector_t px(pfunct.size()); px.setRandom();
        NANO_CHECK_LESS(pfunct.grad_accuracy(px), epsilon2<scalar_t>());
}

NANO_CASE(ginput_accuracy)
{
        const auto params = make_default_params();
        NANO_REQUIRE(params.valid());

        const auto ifunct = make_wrt_inputs_function<affine3d_t>(params);

        vector_t ix(ifunct.size()); ix.setRandom();
        NANO_CHECK_LESS(ifunct.grad_accuracy(ix), epsilon2<scalar_t>());
}

NANO_CASE(3d_vs_4d_output)
{
        const auto params = make_default_params();
        NANO_REQUIRE(params.valid());

        const auto op3d = affine3d_t{params};
        const auto op4d = affine4d_t{params};

        for (int i = 0; i < 8; ++ i)
        {
                tensor4d_t idata, odata3, odata4;
                matrix_t wdata;
                vector_t bdata;

                std::tie(idata, wdata, bdata, odata3) = make_buffers(params, i + 2);
                std::tie(idata, wdata, bdata, odata4) = make_buffers(params, i + 2);

                op3d.output(idata, wdata, bdata, odata3);
                op4d.output(idata, wdata, bdata, odata4);

                NANO_CHECK_EIGEN_CLOSE(odata3.array(), odata4.array(), epsilon1<scalar_t>());
        }
}

NANO_CASE(3d_vs_4d_gparam)
{
        const auto params = make_default_params();
        NANO_REQUIRE(params.valid());

        const auto op3d = affine3d_t{params};
        const auto op4d = affine4d_t{params};

        for (int i = 0; i < 8; ++ i)
        {
                tensor4d_t idata, odata;
                matrix_t wdata3, wdata4;
                vector_t bdata3, bdata4;

                std::tie(idata, wdata3, bdata3, odata) = make_buffers(params, i + 2);
                std::tie(idata, wdata4, bdata4, odata) = make_buffers(params, i + 2);

                op3d.gparam(idata, wdata3, bdata3, odata);
                op4d.gparam(idata, wdata4, bdata4, odata);

                NANO_CHECK_EIGEN_CLOSE(wdata3, wdata4, epsilon1<scalar_t>());
                NANO_CHECK_EIGEN_CLOSE(bdata3, bdata4, epsilon1<scalar_t>());
        }
}

NANO_CASE(3d_vs_4d_ginput)
{
        const auto params = make_default_params();
        NANO_REQUIRE(params.valid());

        const auto op3d = affine3d_t{params};
        const auto op4d = affine4d_t{params};

        for (int i = 0; i < 8; ++ i)
        {
                tensor4d_t idata3, idata4, odata;
                matrix_t wdata;
                vector_t bdata;

                std::tie(idata3, wdata, bdata, odata) = make_buffers(params, i + 2);
                std::tie(idata4, wdata, bdata, odata) = make_buffers(params, i + 2);

                op3d.ginput(idata3, wdata, bdata, odata);
                op4d.ginput(idata4, wdata, bdata, odata);

                NANO_CHECK_EIGEN_CLOSE(idata3.array(), idata4.array(), epsilon1<scalar_t>());
        }
}

NANO_END_MODULE()
