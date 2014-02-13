#ifndef NANOCV_THREAD_POOL_H
#define NANOCV_THREAD_POOL_H

#include <thread>
#include <vector>
#include <condition_variable>
#include <deque>
#include "singleton.hpp"

namespace ncv
{
        ///
        /// \brief the number of threads available on the system
        ///
        inline size_t n_threads()
        {
                return static_cast<size_t>(std::thread::hardware_concurrency());
        }

        ///
        /// \brief asynchronously runs multiple workers/jobs/threads
        /// by enqueing and distribute them on all available threads
        ///
        class thread_pool_t : public singleton_t<thread_pool_t>
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
                thread_pool_t();

                ///
                /// \brief destructor
                ///
                ~thread_pool_t();

                ///
                /// \brief restrict the number of workers, if zero all workers will be active
                ///
                void resize(size_t threads = 0);

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

                // access functions
                size_t n_workers() const { return m_workers.size(); }
                size_t n_jobs() const { return m_data.m_tasks.size(); }
                size_t n_running() const { return m_data.m_running; }
                size_t max_running() const { return m_data.m_maxrunning; }

        private:

                ///
                /// \brief collect the tasks to run
                ///
                struct data_t
                {
                        ///
                        /// \brief constructor
                        ///
                        data_t(size_t maxrunning)
                                :       m_running(0),
                                        m_maxrunning(maxrunning),
                                        m_stop(false)
                        {
                        }

                        // attributes
                        std::deque<task_t>      m_tasks;                ///< Tasks (functors) to execute
                        size_t                  m_running;              ///< #running threads
                        size_t                  m_maxrunning;           ///< #maximum number of running threads
                        mutex_t                 m_mutex;                ///< Synchronize task access
                        condition_t             m_condition;            ///< Signaling
                        bool                    m_stop;                 ///< Stop requested
                };

                ///
                /// \brief worker unit (to execute tasks)
                ///
                struct worker_t
                {
                        ///
                        /// \brief constructor
                        ///
                        worker_t(data_t& data) :        m_data(data)
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
                                lock_t lock(m_data.m_mutex);
                                m_data.m_tasks.push_back(task_t(f));
                        }
                        m_data.m_condition.notify_one();
                }

        private:

                // attributes
                std::vector<thread_t>           m_workers;              ///< worker threads
                data_t                          m_data;                 ///< tasks to execute + synchronization
        };
}

#endif // NANOCV_THREAD_POOL_H

