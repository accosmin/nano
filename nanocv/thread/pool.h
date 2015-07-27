#pragma once

#include <deque>
#include <thread>
#include <vector>
#include <memory>
#include "thread.h"
#include <condition_variable>

namespace ncv
{
        ///
        /// \brief asynchronously runs multiple workers/jobs/threads
        /// by enqueing and distribute them on all available threads
        ///
        /// NB: this is heavily copied/inspired by http://progsch.net/wordpress/?p=81
        ///
        class NANOCV_PUBLIC thread_pool_t
        {
        public:

                typedef std::function<void()>           task_t;
                typedef std::thread                     thread_t;
                typedef std::mutex                      mutex_t;
                typedef std::unique_lock<mutex_t>       lock_t;
                typedef std::condition_variable         condition_t;

                ///
                /// \brief constructor
                ///
                explicit thread_pool_t(std::size_t nthreads = 0);

                ///
                /// \brief disable copying
                ///
                thread_pool_t(const thread_pool_t&) = delete;
                thread_pool_t& operator=(const thread_pool_t&) = delete;

                ///
                /// \brief destructor
                ///
                ~thread_pool_t();

                ///
                /// \brief movable
                ///
                thread_pool_t(thread_pool_t&&) = default;

                ///
                /// \brief movable
                ///
                thread_pool_t& operator=(thread_pool_t&&) = default;

                ///
                /// \brief enqueue a new task to execute
                ///
                template<class F>
                void enqueue(F f)
                {
                        _enqueue(f);
                }

                ///
                /// \brief wait for all workers to finish running the tasks
                ///
                void wait();

                ///
                /// \brief number of available worker threads
                ///
                std::size_t n_workers() const;

                ///
                /// \brief number of tasks to run
                ///
                std::size_t n_tasks() const;

        private:

                ///
                /// \brief collect the tasks to run
                ///
                struct data_t
                {
                        ///
                        /// \brief constructor
                        ///
                        data_t() :      m_running(0),
                                        m_stop(false)
                        {
                        }

                        // attributes
                        std::deque<task_t>      m_tasks;                ///< tasks (functors) to execute
                        std::size_t             m_running;              ///< #running threads
                        mutex_t                 m_mutex;                ///< synchronize task access
                        condition_t             m_condition;            ///< signaling
                        bool                    m_stop;                 ///< stop requested
                };

                ///
                /// \brief worker unit (to execute tasks)
                ///
                struct worker_t
                {
                        ///
                        /// \brief constructor
                        ///
                        worker_t(data_t& data) : m_data(data)
                        {
                        }

                        ///
                        /// \brief execute tasks when available
                        ///
                        void operator()();

                        // attributes
                        data_t&                 m_data;                 ///< Tasks
                };

                ///
                /// \brief add a new task to execute (implementation)
                ///
                template<class F>
                void _enqueue(F f)
                {
                        {
                                const lock_t lock(m_data.m_mutex);                                
                                m_data.m_tasks.push_back(task_t(f));
                        }
                        {
                                m_data.m_condition.notify_one();
                        }
                }

        private:

                // attributes
                std::vector<thread_t>           m_workers;              ///< worker threads
                mutable data_t                  m_data;                 ///< tasks to execute + synchronization
        };
}
