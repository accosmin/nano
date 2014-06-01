#ifndef NANOCV_H
#define NANOCV_H

#include "loss.h"
#include "layer.h"
#include "trainer.h"
#include "sampler.h"
#include "accumulator.h"
#include "optimize/opt_gd.hpp"
#include "optimize/opt_cgd.hpp"
#include "optimize/opt_lbfgs.hpp"
#include "common/logger.h"
#include "common/timer.h"
#include "common/math.hpp"
#include "common/stats.hpp"
#include "common/thread_loop.hpp"
#include "common/random.hpp"
#include <cstdlib>

namespace ncv
{
        ///
        /// \brief current version
        ///
        static const size_t MAJOR_VERSION = 0;
        static const size_t MINOR_VERSION = 1;

        inline string_t version()
        {
                return text::to_string(MAJOR_VERSION) + "." +
                       text::to_string(MINOR_VERSION);
        }

        ///
        /// measure function call
        ///
        template
        <
                typename toperator
        >
        void measure_call(const toperator& op, const string_t& msg)
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
        void measure_critical_call(const toperator& op, const string_t& msg_success, const string_t& msg_failure)
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
        /// \brief initialize library (register objects, start worker pool, initialize OpenCL ...)
        ///
        void init();

        ///
        /// \brief evaluate a model (compute the average loss value & error)
        ///
        size_t test(const task_t& task, const fold_t& fold, const loss_t& loss, const model_t& model,
                scalar_t& lvalue, scalar_t& lerror);
}

#endif // NANOCV_H
