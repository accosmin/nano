#pragma once

#include "timer.h"
#include "logger.h"
#include <limits>
#include <cstdlib>

namespace cortex
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
        /// \brief measure a function call (in microseconds)
        ///
        template
        <
                typename toperator
        >
        std::size_t measure_usec(const toperator& op, std::size_t count)
        {
                const timer_t timer;

                for (std::size_t i = 0; i < count; i ++)
                {
                        op();
                }

                return std::max(std::size_t(1), timer.microseconds());
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
                const std::size_t min_usec = 10 * 1000;

                // calibrate the number of function calls to achieve the minimum time resolution
                const auto time1 = measure_usec(op, 1);
                const auto count = std::max(std::size_t(1), (min_usec + time1 / 2) / time1);

                auto best_time = std::numeric_limits<std::size_t>::max();
                for (std::size_t t = 0; t < trials; t ++)
                {
                        const auto time = measure_usec(op, count);
                        if (time < best_time)
                        {
                                best_time = time;
                        }
                }

                return (best_time + count / 2) / count;
        }
}
