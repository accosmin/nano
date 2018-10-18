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
                manager.add<square_loss_t> ("square",
                        "multivariate regression:     l(y, t) = 1/2 * (y - t)^2");
                manager.add<cauchy_loss_t> ("cauchy",
                        "multivariate regression:     l(y, t) = 1/2 * log(1 + (y - t)^2)");

                manager.add<ssquare_loss_t>("s-square",
                        "single-label classification: l(y, t) = 1/2 * (1 - y*t)^2");
                manager.add<msquare_loss_t>("m-square",
                        "multi-label classification:  l(y, t) = 1/2 * (1 - y*t)^2");

                manager.add<scauchy_loss_t>("s-cauchy",
                        "single-label classification: l(y, t) = 1/2 * log(1 + (1 - y*t)^2)");
                manager.add<mcauchy_loss_t>("m-cauchy",
                        "multi-label classification:  l(y, t) = 1/2 * log(1 + (1 - y*t)^2)");

                manager.add<sexponential_loss_t>("s-exponential",
                        "single-label classification: l(y, t) = exp(-y*t)");
                manager.add<mexponential_loss_t>("m-exponential",
                        "multi-label classification:  l(y, t) = exp(-y*t)");

                manager.add<slogistic_loss_t>("s-logistic",
                        "single-label classification: l(y, t) = log(1 + exp(-y*t))");
                manager.add<mlogistic_loss_t>("m-logistic",
                        "multi-label classification:  l(y, t) = log(1 + exp(-y*t))");

                manager.add<classnll_loss_t>("classnll",
                        "single-label classification: l(y, t) = log(y.exp().sum()) + 1/2 * (1 + t).dot(y)");
        });

        return manager;
}
