#include "utest.h"
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

NANO_BEGIN_MODULE(test_normalize)

NANO_CASE(globally)
{
        const auto count = 9, xmaps = 3, xrows = 7, xcols = 5;

        tensor4d_t idata(count, xmaps, xrows, xcols);
        idata.random(-1, +1);

        auto params = norm_params_t{xmaps, xrows, xcols, norm_type::global};
        NANO_CHECK(params.valid());

        NANO_CHECK_EQUAL(make_dims(xmaps, xrows, xcols), params.xdims());
        NANO_CHECK_EQUAL(0, params.psize());

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

NANO_CASE(by_plane)
{
        const auto count = 9, xmaps = 3, xrows = 7, xcols = 5;

        tensor4d_t idata(count, xmaps, xrows, xcols);
        idata.random(-1, +1);

        auto params = norm_params_t{xmaps, xrows, xcols, norm_type::plane};
        NANO_CHECK(params.valid());

        NANO_CHECK_EQUAL(make_dims(xmaps, xrows, xcols), params.xdims());
        NANO_CHECK_EQUAL(0, params.psize());

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

 NANO_END_MODULE()
