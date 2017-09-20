#include <mutex>
#include "losses/square.h"
#include "losses/cauchy.h"
#include "losses/logistic.h"
#include "losses/classnll.h"
#include "losses/exponential.h"

using namespace nano;

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
