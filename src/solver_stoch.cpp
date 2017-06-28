#include <mutex>
#include "stoch/solver_stoch_ag.h"
#include "stoch/solver_stoch_adam.h"
#include "stoch/solver_stoch_adagrad.h"
#include "stoch/solver_stoch_adadelta.h"
#include "stoch/solver_stoch_ngd.h"
#include "stoch/solver_stoch_sg.h"
#include "stoch/solver_stoch_sgm.h"
#include "stoch/solver_stoch_svrg.h"
#include "stoch/solver_stoch_asgd.h"
#include "stoch/solver_stoch_rmsprop.h"

using namespace nano;

stoch_solver_factory_t& nano::get_stoch_solvers()
{
        static stoch_solver_factory_t manager;

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
