#ifndef NANOCV_THREAD_POOL_H
#define NANOCV_THREAD_POOL_H

#include <thread>
#include <vector>
#include <condition_variable>
#include <deque>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // asynchronously runs multiple workers/jobs/threads
        //      by enqueing and distribute them on all available threads.
        /////////////////////////////////////////////////////////////////////////////////////////

        // return the number of threads available on the system
        inline size_t n_threads()
        {
                return static_cast<size_t>(std::thread::hardware_concurrency());
        }

        // thread pool
        class thread_pool_t
        {
        public:

                typedef std::function<void()>           task_t;
                typedef std::thread                     thread_t;
                typedef std::mutex                      mutex_t;
                typedef std::unique_lock<mutex_t>       lock_t;
                typedef std::condition_variable         condition_t;

                // constructor
                thread_pool_t(size_t threads = 0);

                // destructor
                ~thread_pool_t();

                // disable copying
                thread_pool_t(const thread_pool_t&) = delete;
                thread_pool_t& operator=(const thread_pool_t&) = delete;

                // add a new worker to execute
                template<class F>
                void enqueue(F f)
                {
                        _enqueue(f);
                }

                // wait for all workers to finish
                void wait();

                // access functions
                size_t n_workers() const { return m_workers.size(); }
                size_t n_jobs() const { return m_data.m_tasks.size(); }

        private:

                // task collection
                struct data_t
                {
                        // constructor
                        data_t() :      m_running(0),
                                        m_stop(false)
                        {
                        }

                        // attributes
                        std::deque<task_t>      m_tasks;                // Tasks (functors) to execute
                        size_t                  m_running;              // #running taks
                        mutex_t                 m_mutex;                // Synchronize task access
                        condition_t             m_condition;            // Signaling
                        bool                    m_stop;                 // Stop requested
                };

                // worker
                class worker_t
                {
                public:

                        // constructor
                        worker_t(data_t& data) : m_data(data) {}

                        // execute tasks when available
                        void operator()();

                private:

                        // attributes
                        data_t&         m_data;                 // Tasks
                };

                // add a new task to execute
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
                std::vector<thread_t>   m_workers;              // worker threads
                data_t                  m_data;                 // tasks to execute + synchronization
        };
}

#endif // NANOCV_THREAD_POOL_H

