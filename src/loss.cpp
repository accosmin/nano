#include "losses/square.h"
#include "losses/cauchy.h"
#include "losses/logistic.h"
#include "losses/classnll.h"
#include "losses/exponential.h"

using namespace nano;

loss_manager_t& nano::get_losses()
{
        static loss_manager_t manager;

        static std::once_flag flag;
        std::call_once(flag, [&m = manager] ()
        {
                m.add<square_loss_t>("square",              "multivariate regression:     l(y, t) = 1/2 * L2(y, t)");
                m.add<cauchy_loss_t>("cauchy",              "multivariate regression:     l(y, t) = log(1 + L2(y, t))");
                m.add<mlogistic_loss_t>("m-logistic",       "multi-class classification:  l(y, t) = log(1 + exp(-t.dot(y)))");
                m.add<mexponential_loss_t>("m-exponential", "multi-class classification:  l(y, t) = exp(-t.dot(y))");
                m.add<classnll_loss_t>("classnll",          "single-class classification: l(y, t) = log(y.exp().sum()) + 1/2 * (1 + t).dot(y)");
                m.add<slogistic_loss_t>("s-logistic",       "single-class classification: l(y, t) = log(1 + exp(-t.dot(y)))");
                m.add<sexponential_loss_t>("s-exponential", "single-class classification: l(y, t) = exp(-t.dot(y))");
        });

        return manager;
}
