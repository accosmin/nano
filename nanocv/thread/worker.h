#pragma once

#include "core/arch.h"

namespace ncv
{
        namespace thread
        {
                struct tasks_t;

                ///
                /// \brief worker to process tasks enqueued in a thread pool
                ///
                class NANOCV_PUBLIC pool_worker_t
                {
                public:

                        ///
                        /// \brief constructor
                        ///
                        explicit pool_worker_t(tasks_t& queue);

                        ///
                        /// \brief execute tasks when available
                        ///
                        void operator()();

                private:

                        // attributes
                        tasks_t&           m_queue;        ///< Tasks
                };
        }
}
