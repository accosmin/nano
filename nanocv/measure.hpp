#pragma once

#include "timer.h"
#include "logger.h"
#include <limits>
#include <cstdlib>

namespace ncv
{
        ///
        /// \brief measure function call
        ///
        template
        <
                typename toperator,
                typename tstring
        >
        void measure_and_log(const toperator& op, const tstring& msg)
        {
                const timer_t timer;
                op();
                log_info() << msg << " [" << timer.elapsed() << "].";
        }

        ///
        /// \brief measure function call (and exit if any error)
        ///
        template
        <
                typename toperator,
                typename tstring_success,
                typename tstring_failure
        >
        void measure_critical_and_log(const toperator& op,
                const tstring_success& msg_success, const tstring_failure& msg_failure)
        {
                const timer_t timer;
                if (op())
                {
                        log_info() << msg_success << " (" << timer.elapsed() << ").";
                }
                else
                {
                        log_error() << msg_failure << " (" << timer.elapsed() << ")!";
                        exit(EXIT_FAILURE);
                }
        }

        ///
        /// \brief robustly measure a function call (in microseconds)
        ///
        template
        <
                typename toperator
        >
        std::size_t measure_robustly_usec(const toperator& op, std::size_t trials)
        {
                std::size_t best_time = std::numeric_limits<std::size_t>::max();

                for (std::size_t t = 0; t < trials; t ++)
                {
                        const timer_t timer;
                        op();
                        const std::size_t time = timer.microseconds();
                        if (time < best_time)
                        {
                                best_time = time;
                        }
                }

                return best_time;
        }
}
