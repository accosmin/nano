#pragma once

#include <cstdlib>
#include "logger.h"
#include "chrono/timer.h"

namespace nano
{
        ///
        /// \brief measure function call
        ///
        template <typename toperator, typename tstring>
        void measure_and_log(const toperator& op, const tstring& message)
        {
                const timer_t timer;
                op();
                log_info() << message << " [" << timer.elapsed() << "].";
        }

        ///
        /// \brief measure function call (and exit if any error)
        ///
        template <typename toperator, typename tstring>
        void measure_critical_and_log(const toperator& op, const tstring& message)
        {
                const timer_t timer;
                if (op())
                {
                        log_info() << message << " [" << timer.elapsed() << "].";
                }
                else
                {
                        log_error() << message << " [" << timer.elapsed() << "] failed!";
                        exit(EXIT_FAILURE);
                }
        }
}
