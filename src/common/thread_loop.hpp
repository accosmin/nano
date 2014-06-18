#ifndef NANOCV_THREAD_LOOP_H
#define NANOCV_THREAD_LOOP_H

#include "thread_pool.h"

namespace ncv
{
        ///
        /// \brief split a loop computation of the given size using a thread pool
        /// NB: the operator receives the index of the sample to process: op(i)
        ///
        template
        <
                typename tsize,
                class toperator
        >
        void thread_loopi(tsize N, toperator op, thread_pool_t& pool)
        {
                const tsize n_tasks = static_cast<tsize>(pool.n_workers());
                const tsize task_size = N / n_tasks + 1;

                for (tsize t = 0; t < n_tasks; t ++)
                {
                        pool.enqueue([=,&op]()
                        {
                                for (tsize i = t * task_size, iend = std::min(i + task_size, N); i < iend; i ++)
                                {
                                        op(i);
                                }
                        });
                }

                pool.wait();
        }
        
        ///
        /// \brief split a loop computation of the given size using a thread pool
        /// NB: the operator receives the index of the sample to process and the assigned thread index: op(i, t)
        ///
        template
        <
                typename tsize,
                class toperator
        >
        void thread_loopit(tsize N, toperator op, thread_pool_t& pool)
        {
                const tsize n_tasks = static_cast<tsize>(pool.n_workers());
                const tsize task_size = N / n_tasks + 1;
                
                for (tsize t = 0; t < n_tasks; t ++)
                {
                        pool.enqueue([=,&op]()
                        {
                                for (tsize i = t * task_size, iend = std::min(i + task_size, N); i < iend; i ++)
                                {
                                        op(i, t);
                                }
                        });
                }
                
                pool.wait();
        }

        ///
        /// \brief split a loop computation of the given size using multiple threads
        ///
        template
        <
                typename tsize,
                class toperator
        >
        void thread_loopi(tsize N, toperator op, tsize nthreads = tsize(0))
        {
                thread_pool_t pool(nthreads);
                thread_loopi(N, op, pool);
        }
        
        ///
        /// \brief split a loop computation of the given size using multiple threads
        ///
        template
        <
                typename tsize,
                class toperator
        >
        void thread_loopit(tsize N, toperator op, tsize nthreads = tsize(0))
        {
                thread_pool_t pool(nthreads);
                thread_loopit(N, op, pool);
        }
}

#endif // NANOCV_THREAD_LOOP_H

