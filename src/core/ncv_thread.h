#ifndef NANOCV_THREAD_H
#define NANOCV_THREAD_H

#include "ncv_types.h"
#include "ncv_singleton.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <deque>

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // utility functions to process a loop using multiple threads.
        //
        // assuming a function <op(i)> to process the i-th element out of N total,
        // then instead of:     for (size_t i = 0; i < N; i ++) { op(i); }
        // we can use:          thread_loop(N, op)
        //
        // to automatically split the loop using as many threads as available on the current platform.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        namespace thread_impl
        {
                typedef std::function<void()>           task_t;
                typedef std::thread                     thread_t;
                typedef std::mutex                      mutex_t;
                typedef std::unique_lock<mutex_t>       lock_t;
                typedef std::condition_variable         condition_t;

                // return the number of threads available on the system
                inline size_t n_threads()
                {
                        return static_cast<size_t>(std::thread::hardware_concurrency());
                }

                // worker pool
                class worker_pool_t : public singleton_t<worker_pool_t>
                {
                public:

                        // constructor
                        worker_pool_t();

                        // destructor
                        ~worker_pool_t();

                        // add a new worker to execute
                        template<class F>
                        void enqueue(F f)
                        {
                                _enqueue(f);
                        }

                        // wait for all workers to finish
                        void wait();

                        // access functions
                        size_t n_threads() const { return m_workers.size(); }
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
                        class worker
                        {
                        public:

                                // constructor
                                worker(data_t& data) : m_data(data) {}

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

        // split a loop computation of the given size using multiple threads
        template
        <
                typename tsize,
                class toperator
        >
        void thread_loop(tsize N, toperator op)
        {
                thread_impl::worker_pool_t& pool = thread_impl::worker_pool_t::instance();

                const tsize n_tasks = static_cast<tsize>(pool.n_threads());

                for (tsize t = 0; t < n_tasks; t ++)
                {
                        pool.enqueue([=,&op]()
                        {
                                for (tsize i = t; i < N; i += n_tasks)
                                {
                                        op(i);
                                }
                        });
                }

                pool.wait();
        }

        // split a loop computation of the given size using multiple threads
        //      toperator_init: initialize <tdata> partial results for each thread
        //      toperator_cumulate: cumulate <tdata> partial results
        template
        <                        
                typename tsize,
                typename tdata,
                class toperator_init,
                class toperator,
                class toperator_cumulate
        >
        void thread_loop_cumulate(tsize N, toperator_init op_init, toperator op, toperator_cumulate op_cumulate)
        {
                thread_impl::worker_pool_t& pool = thread_impl::worker_pool_t::instance();

                const tsize n_tasks = static_cast<tsize>(pool.n_threads());

                std::vector<tdata> data(n_tasks);
                for (tsize t = 0; t < n_tasks; t ++)
                {
                        op_init(data[t]);
                }

                for (tsize t = 0; t < n_tasks; t ++)
                {
                        pool.enqueue([=,&data,&op]()
                        {
                                for (tsize i = t; i < N; i += n_tasks)
                                {
                                        op(i, data[t]);
                                }
                        });
                }

                pool.wait();

                for (tsize t = 0; t < n_tasks; t ++)
                {
                        op_cumulate(data[t]);
                }
        }
}

#endif // NANOCV_THREAD_H

