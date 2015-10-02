#pragma once

#include <deque>
#include <mutex>
#include <cstddef>
#include <functional>
#include <condition_variable>

namespace thread
{
        typedef std::function<void()>   task_t;

        ///
        /// \brief queue tasks to be run in a thread pool
        ///
        struct tasks_t
        {
                ///
                /// \brief constructor
                ///
                tasks_t() :     m_running(0),
                                m_stop(false)
                {
                }

                ///
                /// \brief enqueue a new task to execute
                ///
                template<class F>
                void enqueue(F f)
                {
                        {
                                const std::lock_guard<std::mutex> lock(m_mutex);
                                m_tasks.push_back(task_t(f));
                        }
                        {
                                m_condition.notify_one();
                        }
                }

                // attributes
                std::deque<task_t>              m_tasks;                ///< tasks (functors) to execute
                std::size_t                     m_running;              ///< #running threads
                mutable std::mutex              m_mutex;                ///< synchronize task access
                mutable std::condition_variable m_condition;            ///< signaling
                bool                            m_stop;                 ///< stop requested
        };
}
