#include "utest.h"
#include "function.h"
#include "math/stats.h"
#include "math/epsilon.h"
#include "layers/norm4d.h"

using namespace nano;

template <typename ttensor>
static auto get_stats(const ttensor& xdata)
{
        stats_t<scalar_t> stats;
        stats(xdata.data(), xdata.data() + xdata.size());
        return stats;
}

auto make_buffers(const norm_params_t& params, const tensor_size_t count)
{
        auto idata = params.make_xdata(count); idata.setRandom();
        auto odata = params.make_xdata(count); odata.setRandom();
        return std::make_tuple(idata, odata);
}

template <typename top>
struct wrt_inputs_function_t final : public function_t
{
        explicit wrt_inputs_function_t(const top& op) :
                function_t("norm4d", op.params().xsize(), op.params().xsize(), op.params().xsize(), convexity::no, 1e+6),
                m_op(op)
        {
                std::tie(m_idata, m_odata) = make_buffers(op.params(), 1);
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
                m_idata = map_tensor(x.data(), m_idata.dims());
                m_op.output(m_idata, m_odata);
                if (gx)
                {
                        gx->resize(x.size());
                        auto idata = map_tensor(gx->data(), m_idata.dims());
                        m_op.ginput(idata, m_odata);
                }
                return m_odata.array().square().sum() / 2;
        }

        mutable top             m_op;
        mutable tensor4d_t      m_idata;
        mutable tensor4d_t      m_odata;
};

auto make_wrt_inputs_function(const norm_params_t& params)
{
        return wrt_inputs_function_t<norm4d_t>(norm4d_t{params});
}

NANO_BEGIN_MODULE(test_norm4d)

NANO_CASE(globally)
{
        const auto count = 9, xmaps = 3, xrows = 7, xcols = 5;
        const auto params = norm_params_t{xmaps, xrows, xcols, norm_type::global};

        NANO_CHECK(params.valid());
        NANO_CHECK_EQUAL(params.xdims(), make_dims(xmaps, xrows, xcols));
        NANO_CHECK_EQUAL(params.psize(), 0);

        tensor4d_t idata(count, xmaps, xrows, xcols);
        idata.random(-1, +1);

        auto odata = idata;
        norm4d_t norm(params);
        norm.output(idata, odata);

        for (auto x = 0; x < count; ++ x)
        {
                const auto stats = get_stats(odata.tensor(x));

                NANO_CHECK_EQUAL(stats.count(), static_cast<size_t>(xmaps * xrows * xcols));
                NANO_CHECK_LESS(std::fabs(stats.avg() - scalar_t(0)), epsilon0<scalar_t>());
                NANO_CHECK_LESS(std::fabs(stats.var() - scalar_t(1)), epsilon0<scalar_t>());
        }
}

NANO_CASE(globally_ginput_accuracy)
{
        const auto xmaps = 3, xrows = 7, xcols = 5;
        const auto params = norm_params_t{xmaps, xrows, xcols, norm_type::global};
        NANO_REQUIRE(params.valid());

        const auto ifunct = make_wrt_inputs_function(params);

        vector_t ix(ifunct.size()); ix.setRandom();
        NANO_CHECK_LESS(ifunct.grad_accuracy(ix), epsilon1<scalar_t>());
}

NANO_CASE(by_plane)
{
        const auto count = 9, xmaps = 3, xrows = 7, xcols = 5;
        const auto params = norm_params_t{xmaps, xrows, xcols, norm_type::plane};

        NANO_CHECK(params.valid());
        NANO_CHECK_EQUAL(params.xdims(), make_dims(xmaps, xrows, xcols));
        NANO_CHECK_EQUAL(params.psize(), 0);

        tensor4d_t idata(count, xmaps, xrows, xcols);
        idata.random(-1, +1);

        auto odata = idata;
        norm4d_t norm(params);
        norm.output(idata, odata);

        for (auto x = 0; x < count; ++ x)
        {
                for (auto i = 0; i < xmaps; ++ i)
                {
                        const auto stats = get_stats(odata.matrix(x, i));

                        NANO_CHECK_EQUAL(stats.count(), static_cast<size_t>(xrows * xcols));
                        NANO_CHECK_LESS(std::fabs(stats.avg() - scalar_t(0)), epsilon0<scalar_t>());
                        NANO_CHECK_LESS(std::fabs(stats.var() - scalar_t(1)), epsilon0<scalar_t>());
                }
        }
}

NANO_CASE(by_plane_ginput_accuracy)
{
        const auto xmaps = 3, xrows = 7, xcols = 5;
        const auto params = norm_params_t{xmaps, xrows, xcols, norm_type::plane};
        NANO_REQUIRE(params.valid());

        const auto ifunct = make_wrt_inputs_function(params);

        vector_t ix(ifunct.size()); ix.setRandom();
        NANO_CHECK_LESS(ifunct.grad_accuracy(ix), epsilon1<scalar_t>());
}

 NANO_END_MODULE()
