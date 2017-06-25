#include "layer.h"
#include "utest.h"
#include "math/stats.h"
#include "math/epsilon.h"

using namespace nano;

auto make_tensor(const tensor3d_dims_t& dims)
{
        tensor3d_t t(dims);
        t.random(-1, +1);
        return t;
}

NANO_BEGIN_MODULE(test_whitening)

NANO_CASE(full)
{
        const auto idims = tensor3d_dims_t{3, 7, 5};
        const auto odims = idims;
        const auto pdims = tensor1d_dims_t{0};

        const auto layer = get_layers().get("whitening");
        layer->configure(idims, "");
        NANO_CHECK_EQUAL(idims, layer->idims());
        NANO_CHECK_EQUAL(odims, layer->odims());
        NANO_CHECK_EQUAL(0, layer->psize());

        const auto idata = make_tensor(idims);
        const auto param = tensor1d_t{pdims};
        auto odata = tensor3d_t{odims};

        layer->output(map_tensor(idata.data(), idims), map_tensor(param.data(), pdims), map_tensor(odata.data(), odims));

        stats_t<double> stats;
        stats(odata.data(), odata.data() + odata.size());

        NANO_CHECK_EQUAL(stats.count(), static_cast<size_t>(nano::size(odims)));
        NANO_CHECK_LESS(std::fabs(stats.avg() - double(0)), epsilon0<double>());
        NANO_CHECK_LESS(std::fabs(stats.var() - double(1)), epsilon0<double>());
}

NANO_END_MODULE()
