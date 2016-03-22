#pragma once

#include <deque>
#include <mutex>
#include <cstddef>
#include <future>
#include <condition_variable>

namespace nano
{
        using job_t = std::packaged_task<void()>;

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
                template <typename tfunc>
                std::future<void> enqueue(tfunc f)
                {
                        auto job = job_t(f);
                        auto fut = job.get_future();

                        const std::lock_guard<std::mutex> lock(m_mutex);
                        m_jobs.push_back(std::move(job));
                        m_condition.notify_one();

                        return fut;
                }

                // attributes
                std::deque<job_t>               m_jobs;                 ///< jobs to execute
                std::size_t                     m_running;              ///< #running threads
                mutable std::mutex              m_mutex;                ///< synchronization
                mutable std::condition_variable m_condition;            ///< signaling
                bool                            m_stop;                 ///< stop requested
        };
}
