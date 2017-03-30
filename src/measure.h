#pragma once

#include "timer.h"
#include "math/numeric.h"

namespace nano
{
        ///
        /// \brief robustly measure a function call (in picoseconds)
        ///
        template <typename toperator>
        picoseconds_t measure_robustly_psec(const toperator& op, const std::size_t trials,
                const std::size_t min_trial_iterations = 1,
                const microseconds_t min_trial_duration = microseconds_t(10 * 1000))
        {
                const auto run_trial = [&] ()
                {
                        // calibrate the number of function calls to achieve the minimum time resolution
                        std::size_t count = std::max(std::size_t(1), min_trial_iterations);
                        microseconds_t usecs(0);
                        while (usecs < min_trial_duration)
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
                };

                // measure multiple times for robustness
                picoseconds_t duration = run_trial();
                for (std::size_t t = 1; t < trials; ++ t)
                {
                        duration = std::min(duration, run_trial());
                }

                return duration;
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
        ///     given the number of FLOPs run in the given duration.
        ///
        template <typename tinteger, typename tduration>
        int64_t gflops(const tinteger flops, const tduration& duration)
        {
                return  nano::idiv(
                        static_cast<int64_t>(flops) * 1000,
                        static_cast<int64_t>(std::chrono::duration_cast<picoseconds_t>(duration).count()));
        }
}
