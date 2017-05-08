#pragma once

#include "state.h"
#include "accumulator.h"
#include "task_iterator.h"
#include "trainer_result.h"
#include "text/to_string.h"
#include "logger.h"

namespace nano
{
        ///
        /// \brief log the current optimization state & check stopping criteria.
        ///
        inline bool ulog(accumulator_t& acc, task_iterator_t& it,
                size_t& epoch, const size_t epochs, trainer_result_t& result, const trainer_policy policy, const size_t patience,
                const timer_t& timer,
                const state_t& state, const string_t& config = string_t())
        {
                // evaluate the current state
                // NB: the training state is already estimated!
                const auto train = trainer_measurement_t{acc.vstats().avg(), acc.estats().avg()};

                acc.params(state.x);
                acc.mode(accumulator_t::type::value);
                acc.update(it.valid_fold());
                const auto valid = trainer_measurement_t{acc.vstats().avg(), acc.estats().avg()};

                acc.params(state.x);
                acc.mode(accumulator_t::type::value);
                acc.update(it.test_fold());
                const auto test = trainer_measurement_t{acc.vstats().avg(), acc.estats().avg()};

                // OK, update the optimum solution
                const auto milis = timer.milliseconds();
                const auto xnorm = state.x.lpNorm<2>();
                const auto gnorm = state.convergence_criteria();
                const auto ret = result.update(state, {milis, ++epoch, xnorm, gnorm, train, valid, test}, config, patience);

                log_info()
                        << "[" << epoch << "/" << epochs
                        << ":train=" << train
                        << ",valid=" << valid << "|" << nano::to_string(ret)
                        << ",test=" << test
                        << "," << config << ",batch=" << it.size()
                        << ",g=" << gnorm << ",x=" << xnorm
                        << "] " << timer.elapsed() << ".";

                return !nano::is_done(ret, policy);
        }
}
