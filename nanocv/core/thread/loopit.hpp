#pragma once

#include "pool.h"

namespace ncv
{
        ///
        /// \brief split a loop computation of the given size using a thread pool
        /// NB: the operator receives the index of the sample to process and the assigned thread index: op(i, t)
        ///
        template
        <
                typename tsize,
                class toperator
        >
        void thread_loopit(tsize N, thread_pool_t& pool, toperator op)
        {
                const tsize n_tasks = static_cast<tsize>(pool.n_workers());
                const tsize task_size = (N + n_tasks - 1) / n_tasks;
                
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
        void thread_loopit(tsize N, tsize nthreads, toperator op)
        {
                thread_pool_t pool(nthreads);
                thread_loopit(N, pool, op);
        }
        
        ///
        /// \brief split a loop computation of the given size using all available threads
        ///
        template
        <
                typename tsize,
                class toperator
        >
        void thread_loopit(tsize N, toperator op)
        {
                thread_loopit(N, tsize(0), op);
        }
}
