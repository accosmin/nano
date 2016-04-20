#pragma once

#include <deque>
#include <mutex>
#include <future>
#include <cstddef>
#include "section.hpp"
#include <condition_variable>

namespace thread
{
        using task_t = std::packaged_task<void()>;
        using future_t = std::future<void>;

        ///
        /// \brief queue tasks to be run in a thread pool.
        ///
        struct queue_t
        {
                ///
                /// \brief constructor
                ///
                queue_t() : m_stop(false)
                {
                }

                ///
                /// \brief enqueue a new task to execute
                ///
                template <typename tfunc>
                future_t enqueue(tfunc f)
                {
                        auto task = task_t(f);
                        auto fut = task.get_future();

                        const std::lock_guard<std::mutex> lock(m_mutex);
                        m_tasks.push_back(std::move(task));
                        m_condition.notify_all();

                        return fut;
                }

                // attributes
                std::deque<task_t>              m_tasks;                ///< tasks to execute
                mutable std::mutex              m_mutex;                ///< synchronization
                mutable std::condition_variable m_condition;            ///< signaling
                bool                            m_stop;                 ///< stop requested
        };
}
