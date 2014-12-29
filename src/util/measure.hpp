#pragma once

#include "file/logger.h"
#include "timer.h"
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
        void measure_call(const toperator& op, const tstring& msg)
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
        void measure_critical_call(const toperator& op, const tstring_success& msg_success, const tstring_failure& msg_failure)
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
