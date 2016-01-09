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
        /// \brief robustly measure a function call (in the given period)
        ///
        template
        <
                typename tperiod,
                typename toperator
        >
        tperiod measure_robustly(const toperator& op, const std::size_t trials)
        {
                const microseconds_t min_usecs(100 * 1000);

                // calibrate the number of function calls to achieve the minimum time resolution
                std::size_t count = trials; 
                microseconds_t usecs(0);
                while (true)
                {
                        usecs = measure_usec(op, count);
                        if (usecs < min_usecs)
                        {
                                count *= 2;
                        }
                        else
                        {
                                break;
                        }
                }

                return (usecs + tperiod(count / 2)) / count;
        }

        ///
        /// \brief robustly measure a function call (in nanoseconds)
        ///
        template
        <
                typename toperator
        >
        nanoseconds_t measure_robustly_nsec(const toperator& op, const std::size_t trials)
        {
                return measure_robustly<nanoseconds_t>(op, trials);
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
                return measure_robustly<microseconds_t>(op, trials);
        }
        

        ///
        /// \brief robustly measure a function call (in miliseconds)
        ///
        template
        <
                typename toperator
        >
        milliseconds_t measure_robustly_msec(const toperator& op, const std::size_t trials)
        {
                return milliseconds_t((measure_robustly_usec(op, trials).count() + 500) / 1000);
        }
}
