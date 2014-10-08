#pragma once

#include "common/logger.h"
#include "common/timer.h"
#include <cstdlib>

namespace ncv
{
        ///
        /// \brief measure function call
        ///
        template
        <
                typename toperator
        >
        void measure_call(const toperator& op, const std::string& msg)
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
                typename toperator
        >
        void measure_critical_call(const toperator& op, const std::string& msg_success, const std::string& msg_failure)
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
}
