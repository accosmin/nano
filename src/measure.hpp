#pragma once

#include "timer.h"
#include "math/cast.hpp"
#include "math/numeric.hpp"

namespace nano
{
        ///
        /// \brief robustly measure a function call (in picoseconds)
        ///
        template
        <
                typename toperator
        >
        picoseconds_t measure_robustly_psec(const toperator& op, const std::size_t trials)
        {
                const microseconds_t min_usecs(10 * 1000);

                // calibrate the number of function calls to achieve the minimum time resolution
                std::size_t count = std::max(std::size_t(1), trials / 2);
                microseconds_t usecs(0);
                while (usecs < min_usecs)
                {
                        count *= 2;

                        const timer_t timer;
                        for (std::size_t i = 0; i < count; ++ i)
                        {
                                op();
                        }
                        usecs = timer.microseconds();
                }

                return picoseconds_t(nano::idiv(usecs.count() * 1000 * 1000, count));
        }

        ///
        /// \brief robustly measure a function call (in microseconds)
        ///
        template <typename toperator>
        nanoseconds_t measure_robustly_nsec(const toperator& op, const std::size_t trials)
        {
                return std::chrono::duration_cast<nanoseconds_t>(measure_robustly_psec(op, trials));
        }

        ///
        /// \brief robustly measure a function call (in microseconds)
        ///
        template <typename toperator>
        microseconds_t measure_robustly_usec(const toperator& op, const std::size_t trials)
        {
                return std::chrono::duration_cast<microseconds_t>(measure_robustly_psec(op, trials));
        }

        ///
        /// \brief robustly measure a function call (in milliseconds)
        ///
        template <typename toperator>
        milliseconds_t measure_robustly_msec(const toperator& op, const std::size_t trials)
        {
                return std::chrono::duration_cast<milliseconds_t>(measure_robustly_psec(op, trials));
        }

        ///
        /// \brief compute GFLOPS (giga floating point operations per seconds)
        ///     given the number of FLOPs run in the given duration
        ///
        template
        <
                typename tinteger,
                typename tduration
        >
        double gflops(const tinteger flops, const tduration& duration)
        {
                return  static_cast<double>(flops) /
                        static_cast<double>(std::chrono::duration_cast<picoseconds_t>(duration).count()) * 1e+3;
        }
}
