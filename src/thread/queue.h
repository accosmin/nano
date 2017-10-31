#pragma once

#include <deque>
#include <mutex>
#include <future>
#include <condition_variable>

namespace nano
{
        using worker_task_t = std::packaged_task<void()>;
        using future_t = std::future<void>;

        ///
        /// \brief queue tasks to be run in a thread pool.
        ///
        struct worker_queue_t
        {
                ///
                /// \brief constructor
                ///
                worker_queue_t() = default;

                ///
                /// \brief enqueue a new task to execute
                ///
                template <typename tfunction>
                future_t enqueue(tfunction f)
                {
                        auto task = worker_task_t(f);
                        auto fut = task.get_future();

                        const std::lock_guard<std::mutex> lock(m_mutex);
                        m_tasks.emplace_back(std::move(task));
                        m_condition.notify_all();

                        return fut;
                }

                // attributes
                std::deque<worker_task_t>       m_tasks;                ///< tasks to execute
                mutable std::mutex              m_mutex;                ///< synchronization
                mutable std::condition_variable m_condition;            ///< signaling
                bool                            m_stop{false};          ///< stop requested
        };
}
