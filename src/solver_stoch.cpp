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
#include "stoch/solver_stoch_amsgrad.h"
#include "stoch/solver_stoch_cocob.h"

using namespace nano;

solver_state_t stoch_solver_t::tune(const stoch_params_t& params, const function_t& function, const vector_t& x0,
        const size_t trials_per_parameter)
{
        auto tuner = this->configs();

        if (tuner.n_params() == size_t(0))
        {
                // no tuning required (e.g. parameter-free optimization)
                return minimize(params, function, x0);
        }
        else
        {
                // tuning required: the number of trials is proportional with the number of parameters to tune
                string_t best_config;
                solver_state_t best_state;

                for (const auto& config : tuner.get(trials_per_parameter * tuner.n_params()))
                {
                        this->config(config);

                        const auto state = minimize(params, function, x0);
                        if (state < best_state)
                        {
                                best_state = state;
                                best_config = config;
                        }
                }

                this->config(best_config);
                return best_state;
        }
}

stoch_solver_factory_t& nano::get_stoch_solvers()
{
        static stoch_solver_factory_t manager;

        static std::once_flag flag;
        std::call_once(flag, [] ()
        {
                manager.add<stoch_sg_t>("sg", "stochastic gradient (descent)");
                manager.add<stoch_sgm_t>("sgm", "stochastic gradient (descent) with momentum");
                manager.add<stoch_ngd_t>("ngd", "stochastic normalized gradient");
                manager.add<stoch_svrg_t>("svrg", "stochastic variance reduced gradient");
                manager.add<stoch_asgd_t>("asgd", "averaged stochastic gradient (descent)");
                manager.add<stoch_ag_t>("ag", "Nesterov's accelerated gradient");
                manager.add<stoch_agfr_t>("agfr", "Nesterov's accelerated gradient with function value restarts");
                manager.add<stoch_aggr_t>("aggr", "Nesterov's accelerated gradient with gradient restarts");
                manager.add<stoch_adam_t>("adam", "Adam (see citation)");
                manager.add<stoch_adagrad_t>("adagrad", "AdaGrad (see citation)");
                manager.add<stoch_adadelta_t>("adadelta", "AdaDelta (see citation)");
                manager.add<stoch_rmsprop_t>("rmsprop", "RMSProp (see citation)");
                manager.add<stoch_amsgrad_t>("amsgrad", "AMSGrad (see citation)");
                manager.add<stoch_cocob_t>("cocob", "COCOB-Backprop (see citation)");
        });

        return manager;
}
