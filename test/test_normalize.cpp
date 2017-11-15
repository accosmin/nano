#include "layer.h"
#include "utest.h"
#include "math/stats.h"
#include "math/epsilon.h"
#include "layers/layer_normalize.h"

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

        const auto layer = get_layers().get("norm");
        layer->config(json_writer_t().object("type", norm_type::global).get());

        NANO_CHECK_EQUAL(layer->resize(make_dims(xmaps, xrows, xcols), ""), true);
        NANO_CHECK_EQUAL(make_dims(xmaps, xrows, xcols), layer->idims());
        NANO_CHECK_EQUAL(make_dims(xmaps, xrows, xcols), layer->odims());
        NANO_CHECK_EQUAL(0, layer->psize());

        const auto& odata = layer->output(idata);
        NANO_CHECK_EQUAL(odata.dims(), make_dims(count, xmaps, xrows, xcols));

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

        const auto layer = get_layers().get("norm");
        layer->config(json_writer_t().object("type", norm_type::plane).get());

        NANO_CHECK_EQUAL(layer->resize(make_dims(xmaps, xrows, xcols), ""), true);
        NANO_CHECK_EQUAL(make_dims(xmaps, xrows, xcols), layer->idims());
        NANO_CHECK_EQUAL(make_dims(xmaps, xrows, xcols), layer->odims());
        NANO_CHECK_EQUAL(0, layer->psize());

        const auto& odata = layer->output(idata);
        NANO_CHECK_EQUAL(odata.dims(), make_dims(count, xmaps, xrows, xcols));

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
