#pragma once

#include "arch.h"

namespace thread
{
        struct tasks_t;

        ///
        /// \brief configure & manipulate a worker thread
        ///
        class NANOCV_PUBLIC worker_config_t
        {
        public:
                ///
                /// \brief constructor
                ///
                explicit worker_config_t(bool active = true);

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
                bool            m_active;       ///< is worker active for processing tasks?
        };

        ///
        /// \brief worker to process tasks en-queued in a thread pool
        ///
        class NANOCV_PUBLIC worker_t
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit worker_t(tasks_t& queue, worker_config_t& config);

                ///
                /// \brief execute tasks when available
                ///
                void operator()();

        private:

                // attributes
                tasks_t&                m_queue;        ///< task queue to process
                worker_config_t&        m_config;       ///<
        };
}
