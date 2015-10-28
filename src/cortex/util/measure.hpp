#pragma once

#include "timer.h"
#include "math/stats.hpp"

namespace cortex
{
        ///
        /// \brief measure a function call (in microseconds)
        ///
        template
        <
                typename toperator
        >
        std::size_t measure_usec(const toperator& op, const std::size_t count)
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
        std::size_t measure_robustly_usec(const toperator& op, const std::size_t trials)
        {
                const std::size_t min_usec = 10 * 1000;

                // calibrate the number of function calls to achieve the minimum time resolution
                std::size_t count = 1;
                std::size_t time1 = 0;
                while (time1 < min_usec)
                {
                        time1 = measure_usec(op, count);
                        count *= 2;
                }

                // stable measurements, so run the trials
                math::stats_t<std::size_t> measurements;
                for (std::size_t t = 0; t < trials; t ++)
                {
                        measurements(measure_usec(op, count));
                }

                return (measurements.min() + count / 2) / count;
        }
}
