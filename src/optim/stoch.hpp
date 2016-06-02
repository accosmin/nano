#pragma once

#include "problem.h"

#include "stoch_types.h"
#include "stoch/stoch_ag.hpp"
#include "stoch/stoch_sg.hpp"
#include "stoch/stoch_ngd.hpp"
#include "stoch/stoch_sgm.hpp"
#include "stoch/stoch_adam.hpp"
#include "stoch/stoch_adagrad.hpp"
#include "stoch/stoch_adadelta.hpp"

namespace nano
{
        ///
        /// \brief stochastic optimization
        ///
        template
        <
                typename topulog,       ///< logging operator
                typename toptlog        ///< tuning operator
        >
        auto minimize(
                const problem_t& problem,
                const topulog& fn_ulog,
                const toptlog& fn_tlog,
                const vector_t& x0,
                const stoch_optimizer optimizer, const std::size_t epochs, const std::size_t epoch_size)
        {
                const stoch_params_t param(epochs, epoch_size, fn_ulog, fn_tlog);

                switch (optimizer)
                {
                case stoch_optimizer::SGM:
                        return stoch_sgm_t<tproblem>()(param, problem, x0);

                case stoch_optimizer::NGD:
                        return stoch_ngd_t<tproblem>()(param, problem, x0);

                case stoch_optimizer::AG:
                        return stoch_ag_t<tproblem>()(param, problem, x0);

                case stoch_optimizer::AGFR:
                        return stoch_agfr_t<tproblem>()(param, problem, x0);

                case stoch_optimizer::AGGR:
                        return stoch_aggr_t<tproblem>()(param, problem, x0);

                case stoch_optimizer::ADAGRAD:
                        return stoch_adagrad_t<tproblem>()(param, problem, x0);

                case stoch_optimizer::ADADELTA:
                        return stoch_adadelta_t<tproblem>()(param, problem, x0);

                case stoch_optimizer::ADAM:
                        return stoch_adam_t<tproblem>()(param, problem, x0);

                case stoch_optimizer::SG:
                default:
                        return stoch_sg_t<tproblem>()(param, problem, x0);
                }
        }
}

