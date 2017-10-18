#include <mutex>
#include "losses/square.h"
#include "losses/cauchy.h"
#include "losses/logistic.h"
#include "losses/classnll.h"
#include "losses/exponential.h"

using namespace nano;

tensor1d_t loss_t::error(const tensor4d_t& targets, const tensor4d_t& scores) const
{
        assert(targets.dims() == scores.dims());

        tensor1d_t errors(targets.size<0>());
        for (auto x = 0; x < targets.size<0>(); ++ x)
        {
                error(targets.vector(x), scores.vector(x), errors(x));
        }
        return errors;
}

tensor1d_t loss_t::value(const tensor4d_t& targets, const tensor4d_t& scores) const
{
        assert(targets.dims() == scores.dims());

        tensor1d_t values(targets.size<0>());
        for (auto x = 0; x < targets.size<0>(); ++ x)
        {
                value(targets.vector(x), scores.vector(x), values(x));
        }
        return values;
}

tensor4d_t loss_t::vgrad(const tensor4d_t& targets, const tensor4d_t& scores) const
{
        assert(targets.dims() == scores.dims());

        tensor4d_t vgrads(targets.dims());
        for (auto x = 0; x < targets.size<0>(); ++ x)
        {
                vgrad(targets.vector(x), scores.vector(x), vgrads.vector(x));
        }
        return vgrads;
}

loss_factory_t& nano::get_losses()
{
        static loss_factory_t manager;

        static std::once_flag flag;
        std::call_once(flag, [] ()
        {
                manager.add<square_loss_t>("square",              "multivariate regression:     l(y, t) = 1/2 * L2(y, t)");
                manager.add<cauchy_loss_t>("cauchy",              "multivariate regression:     l(y, t) = log(1 + L2(y, t))");
                manager.add<mlogistic_loss_t>("m-logistic",       "multi-class classification:  l(y, t) = log(1 + exp(-t.dot(y)))");
                manager.add<mexponential_loss_t>("m-exponential", "multi-class classification:  l(y, t) = exp(-t.dot(y))");
                manager.add<classnll_loss_t>("classnll",          "single-class classification: l(y, t) = log(y.exp().sum()) + 1/2 * (1 + t).dot(y)");
                manager.add<slogistic_loss_t>("s-logistic",       "single-class classification: l(y, t) = log(1 + exp(-t.dot(y)))");
                manager.add<sexponential_loss_t>("s-exponential", "single-class classification: l(y, t) = exp(-t.dot(y))");
        });

        return manager;
}
