#pragma once

#include "timer.h"

namespace cortex
{
        ///
        /// \brief measure a function call (in microseconds)
        ///
        template
        <
                typename toperator
        >
        microseconds_t measure_usec(const toperator& op, const std::size_t trials)
        {
                const timer_t timer;

                for (std::size_t i = 0; i < trials; ++ i)
                {
                        op();
                }

                return std::max(microseconds_t(1), timer.microseconds());
        }

        ///
        /// \brief robustly measure a function call (in microseconds)
        ///
        template
        <
                typename toperator
        >
        microseconds_t measure_robustly_usec(const toperator& op, const std::size_t trials)
        {
                const microseconds_t min_usec(10 * 1000);

                // calibrate the number of function calls to achieve the minimum time resolution
                std::size_t count = 1;
                microseconds_t time1(0);
                while (time1 < min_usec)
                {
                        time1 = measure_usec(op, count);
                        count *= 2;
                }

                // stable measurements, so run the trials
                microseconds_t usec = microseconds_t::max();
                for (std::size_t t = 0; t < trials; ++ t)
                {
                        usec = std::min(usec, measure_usec(op, count));
                }

                return (usec + microseconds_t(count - 1)) / count;
        }
}
