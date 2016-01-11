#pragma once

#include "timer.h"

namespace cortex
{
        ///
        /// \brief measure a function call (in nanoseconds)
        ///
        template
        <
                typename toperator
        >
        nanoseconds_t measure_nsec(const toperator& op, const std::size_t trials)
        {
                const timer_t timer;

                for (std::size_t i = 0; i < trials; ++ i)
                {
                        op();
                }

                return timer.nanoseconds();
        }

        ///
        /// \brief robustly measure a function call (in the given period)
        ///
        template
        <
                typename toperator
        >
        nanoseconds_t measure_robustly(const toperator& op, const std::size_t trials)
        {
                const nanoseconds_t min_nsecs(10 * 1000 * 1000);

                // calibrate the number of function calls to achieve the minimum time resolution
                std::size_t count = std::max(std::size_t(1), trials / 2);
                nanoseconds_t nsecs(0);
                while (nsecs < min_nsecs)
                {
                        count *= 2;                
                        nsecs = measure_nsec(op, count);
                }
                
                return nsecs / count;
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
                return measure_robustly(op, trials);
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
                return std::chrono::duration_cast<microseconds_t>(measure_robustly(op, trials));
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
                return std::chrono::duration_cast<milliseconds_t>(measure_robustly(op, trials));
        }
}
