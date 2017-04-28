#pragma once

#include "queue.h"

namespace nano
{
        ///
        /// \brief worker to process tasks enqueued in a thread pool.
        ///
        class NANO_PUBLIC worker_t
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit worker_t(worker_queue_t& queue, const bool active = true);

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

                static bool toggle(bool& value, const bool flag)
                {
                        const bool changed = value == !flag;
                        value = flag;
                        return changed;
                }

                // attributes
                worker_queue_t& m_queue;        ///< task queue to process
                bool            m_active;       ///< is worker active for processing tasks?
        };

        inline worker_t::worker_t(worker_queue_t& queue, const bool active) :
                m_queue(queue),
                m_active(active)
        {
        }

        inline void nano::worker_t::operator()() const
        {
                while (true)
                {
                        worker_task_t task;

                        // wait for a new task to be available in the queue
                        {
                                std::unique_lock<std::mutex> lock(m_queue.m_mutex);

                                m_queue.m_condition.wait(lock, [&]
                                {
                                        return m_queue.m_stop || (active() && !m_queue.m_tasks.empty());
                                });

                                if (m_queue.m_stop)
                                {
                                        m_queue.m_tasks.clear();
                                        m_queue.m_condition.notify_all();
                                        break;
                                }

                                task = std::move(m_queue.m_tasks.front());
                                m_queue.m_tasks.pop_front();
                        }

                        // execute the task
                        task();
                }
        }

        inline bool nano::worker_t::activate()
        {
                return toggle(m_active, true);
        }

        inline bool nano::worker_t::deactivate()
        {
                return toggle(m_active, false);
        }

        inline bool nano::worker_t::active() const
        {
                return m_active;
        }
}
