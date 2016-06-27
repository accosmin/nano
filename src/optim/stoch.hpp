#pragma once

#include "stoch/ag.h"
#include "stoch/sg.h"
#include "stoch/ngd.h"
#include "stoch/sgm.h"
#include "stoch/adam.h"
#include "stoch/adagrad.h"
#include "stoch/adadelta.h"

namespace nano
{
        ///
        /// \brief stochastic optimization
        ///
        template
        <
                typename topulog        ///< logging operator
        >
        auto minimize(
                const problem_t& problem,
                const topulog& fn_ulog,
                const vector_t& x0,
                const stoch_optimizer optimizer, const std::size_t epochs, const std::size_t epoch_size)
        {
                const stoch_params_t param(epochs, epoch_size, fn_ulog);

                switch (optimizer)
                {
                case stoch_optimizer::AG:       return stoch_ag_t(ag_restart::none)(param, problem, x0);
                case stoch_optimizer::AGFR:     return stoch_ag_t(ag_restart::function)(param, problem, x0);
                case stoch_optimizer::AGGR:     return stoch_ag_t(ag_restart::gradient)(param, problem, x0);

                case stoch_optimizer::ADAGRAD:  return stoch_adagrad_t()(param, problem, x0);
                case stoch_optimizer::ADADELTA: return stoch_adadelta_t()(param, problem, x0);
                case stoch_optimizer::ADAM:     return stoch_adam_t()(param, problem, x0);

                case stoch_optimizer::SGM:      return stoch_sgm_t()(param, problem, x0);
                case stoch_optimizer::NGD:      return stoch_ngd_t()(param, problem, x0);
                case stoch_optimizer::SG:
                default:                        return stoch_sg_t()(param, problem, x0);
                }
        }
}

