#pragma once

#include "arch.h"

namespace zob
{
        struct queue_t;

        ///
        /// \brief worker to process jobs enqueued in a thread pool
        ///
        class ZOB_PUBLIC worker_t
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit worker_t(queue_t& queue, const bool active = true);

                ///
                /// \brief execute jobs when available
                ///
                void operator()() const;

                ///
                /// \brief toggle the worker's activation state
                /// \return the previous activation state
                ///
                bool activate();
                bool deactivate();

                ///
                /// \brief check if the worker is active (aka for processing jobs)
                ///
                bool active() const;

        private:

                // attributes
                queue_t&        m_queue;        ///< job queue to process
                bool            m_active;       ///< is worker active for processing jobs?
        };
}
