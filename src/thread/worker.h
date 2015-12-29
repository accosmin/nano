#pragma once

#include "arch.h"

namespace thread
{
        struct tasks_t;

        ///
        /// \brief worker to process tasks en-queued in a thread pool
        ///
        class NANOCV_PUBLIC worker_t
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit worker_t(tasks_t& queue, const bool active = true);

                ///
                /// \brief execute tasks when available
                ///
                void operator()() const;

                ///
                /// \brief toggle the worker's activation state
                /// \return the previous activation state
                ///
                bool activate();
                bool deactivate();

                ///
                /// \brief check if the worker is active (aka for processing tasks)
                ///
                bool active() const;

        private:

                // attributes
                tasks_t&        m_queue;        ///< task queue to process
                bool            m_active;       ///< is worker active for processing tasks?
        };
}
