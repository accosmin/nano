#pragma once

#include <deque>
#include <mutex>
#include <cstddef>
#include <functional>
#include <condition_variable>

namespace nano
{
        using job_t = std::function<void()>;

        ///
        /// \brief queue jobs to be run in a thread pool
        ///
        struct queue_t
        {
                ///
                /// \brief constructor
                ///
                queue_t() :     m_running(0),
                                m_stop(false)
                {
                }

                ///
                /// \brief enqueue a new job to execute
                ///
                template<class F>
                void enqueue(F f)
                {
                        const std::lock_guard<std::mutex> lock(m_mutex);

                        m_jobs.emplace_back(f);
                        m_condition.notify_all();
                }

                // attributes
                std::deque<job_t>               m_jobs;                 ///< jobs to execute
                std::size_t                     m_running;              ///< #running threads
                mutable std::mutex              m_mutex;                ///< synchronization
                mutable std::condition_variable m_condition;            ///< signaling
                bool                            m_stop;                 ///< stop requested
        };
}
