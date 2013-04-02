#ifndef NANOCV_THREAD_H
#define NANOCV_THREAD_H

#include "ncv_singleton.h"
#include "ncv_types.h"
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
        // then instead of:
        //
        // for (size_t i = 0; i < N; i ++)
        //      op(i)
        //
        // we can use:
        //
        // thread_loop(N, op)
        //
        //      to automatically split the loop using as many threads as available on the current platform.
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
                class worker_pool : public singleton<worker_pool>
                {
                public:
                        // destructor
                        ~worker_pool();

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

                protected:

                        friend class singleton<worker_pool>;

                        // constructor
                        worker_pool();

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
                                index_t                 m_running;              // #running taks
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
                thread_impl::worker_pool& pool = thread_impl::worker_pool::instance();

                const tsize n_tasks = static_cast<tsize>(pool.n_threads());
                for (tsize t = 0; t < n_tasks; t ++)
                {
                        pool.enqueue([=]()
                        {
                                for (tsize i = t; i < N; i += n_tasks)
                                {
                                        op(i);
                                }
                        });
                }

                pool.wait();
        }

//        TODO: policy here (non-overlapping, interleaving memory access),

        // TODO: add also state workers: result = op(begin, end)

//        thread_loop with state:
//        pass two additional policies, one to initialize the result, the other to accumulate the result

//        accumulate the result for the [begin, end) region of the thread
//        accumulate the partial results for all the threads
}

#endif // NANOCV_THREAD_H

