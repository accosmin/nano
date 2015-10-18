#pragma once

#include "problem.hpp"

#include "stoch_types.h"
#include "stoch/ag.hpp"
#include "stoch/sg.hpp"
#include "stoch/sga.hpp"
#include "stoch/sia.hpp"
#include "stoch/adagrad.hpp"
#include "stoch/adadelta.hpp"

namespace min
{
        ///
        /// \brief stochastic optimization
        ///
        template
        <
                typename tscalar,
                typename tproblem = problem_t<tscalar>,
                typename tsize = typename tproblem::tsize,
                typename tstate = typename tproblem::tstate,
                typename tvector = typename tproblem::tvector,
                typename topulog = typename tproblem::top_ulog
        >
        tstate minimize(
                const tproblem& problem,
                const topulog& fn_ulog,
                const tvector& x0,
                stoch_optimizer optimizer, std::size_t epochs, std::size_t epoch_size, tscalar alpha0, tscalar decay = 0.50)
        {
                switch (optimizer)
                {
                case stoch_optimizer::SGA:
                        return  stoch_sga_t<tproblem>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);

                case stoch_optimizer::SIA:
                        return  stoch_sia_t<tproblem>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);

                case stoch_optimizer::AG:
                        return  stoch_ag_t<tproblem>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);

                case stoch_optimizer::AGGR:
                        return  stoch_aggr_t<tproblem>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);

                case stoch_optimizer::ADAGRAD:
                        return  stoch_adagrad_t<tproblem>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);

                case stoch_optimizer::ADADELTA:
                        return  stoch_adadelta_t<tproblem>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);

                case stoch_optimizer::SG:
                default:
                        return  stoch_sg_t<tproblem>
                                (epochs, epoch_size, alpha0, decay, fn_ulog)
                                (problem, x0);
                }
        }
}

