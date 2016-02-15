#pragma once

#include "problem.hpp"

#include "stoch_types.h"
#include "stoch/ag.hpp"
#include "stoch/sg.hpp"
#include "stoch/sgm.hpp"
#include "stoch/sga.hpp"
#include "stoch/sia.hpp"
#include "stoch/adagrad.hpp"
#include "stoch/adadelta.hpp"

namespace math
{
        ///
        /// \brief stochastic optimization
        ///
        template
        <
                typename tproblem,
                typename tsize = typename tproblem::tsize,
                typename tvector = typename tproblem::tvector,
                typename topulog = typename tproblem::topulog
        >
        auto minimize(
                const tproblem& problem,
                const topulog& fn_ulog,
                const tvector& x0,
                const stoch_optimizer optimizer, const std::size_t epochs, const std::size_t epoch_size)
        {
                const stoch_params_t<tproblem> param(epochs, epoch_size, fn_ulog);

                switch (optimizer)
                {
                case stoch_optimizer::SGM:
                        return stoch_sgm_t<tproblem>()(param, problem, x0);

                case stoch_optimizer::SGA:
                        return stoch_sga_t<tproblem>()(param, problem, x0);

                case stoch_optimizer::SIA:
                        return stoch_sia_t<tproblem>()(param, problem, x0);

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

                case stoch_optimizer::SG:
                default:
                        return stoch_sg_t<tproblem>()(param, problem, x0);
                }
        }
}

