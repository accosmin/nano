#include <mutex>
#include "stoch/ag.h"
#include "stoch/adam.h"
#include "stoch/adagrad.h"
#include "stoch/adadelta.h"
#include "stoch/ngd.h"
#include "stoch/sg.h"
#include "stoch/sgm.h"
#include "stoch/svrg.h"
#include "stoch/asgd.h"
#include "stoch/rmsprop.h"

using namespace nano;

stoch_optimizer_manager_t& nano::get_stoch_optimizers()
{
        static stoch_optimizer_manager_t manager;

        static std::once_flag flag;
        std::call_once(flag, [&m = manager] ()
        {
                m.add<stoch_sg_t>("sg", "stochastic gradient (descent)");
                m.add<stoch_sgm_t>("sgm", "stochastic gradient (descent) with momentum");
                m.add<stoch_ngd_t>("ngd", "stochastic normalized gradient");
                m.add<stoch_svrg_t>("svrg", "stochastic variance reduced gradient");
                m.add<stoch_asgd_t>("asgd", "averaged stochastic gradient (descent)");
                m.add<stoch_ag_t>("ag", "Nesterov's accelerated gradient");
                m.add<stoch_agfr_t>("agfr", "Nesterov's accelerated gradient with function value restarts");
                m.add<stoch_aggr_t>("aggr", "Nesterov's accelerated gradient with gradient restarts");
                m.add<stoch_adam_t>("adam", "Adam (see citation)");
                m.add<stoch_adagrad_t>("adagrad", "AdaGrad (see citation)");
                m.add<stoch_adadelta_t>("adadelta", "AdaDelta (see citation)");
                m.add<stoch_rmsprop_t>("rmsprop", "RMSProp (see citation)");
        });

        return manager;
}
