#ifndef NANOCV_THREAD_LOOP_H
#define NANOCV_THREAD_LOOP_H

#include "thread_pool.h"

namespace ncv
{
        ///
        /// \brief split a loop computation of the given size using a thread pool
        ///
        template
        <
                typename tsize,
                class toperator
        >
        void thread_loop(tsize N, toperator op, thread_pool_t& pool)
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
        /// \brief split a loop computation of the given size using multiple threads
        ///
        template
        <
                typename tsize,
                class toperator
        >
        void thread_loop(tsize N, toperator op, tsize nthreads = tsize(0))
        {
                thread_pool_t pool(nthreads);
                thread_loop(N, op, pool);
        }

        ///
        /// \brief split a loop computation of the given size using a thread pool and cumulate partial results
        ///
        template
        <
                typename tdata,
                typename tsize,
                class toperator_init,
                class toperator,
                class toperator_cumulate
        >
        void thread_loop_cumulate(
                tsize N, toperator_init op_init, toperator op, toperator_cumulate op_cumulate, thread_pool_t& pool)
        {
                const tsize n_tasks = static_cast<tsize>(pool.n_workers());
                const tsize task_size = N / n_tasks + 1;

                std::vector<tdata> data(n_tasks);
                for (tsize t = 0; t < n_tasks; t ++)
                {
                        op_init(data[t]);
                }

                for (tsize t = 0; t < n_tasks; t ++)
                {
                        pool.enqueue([=,&data,&op]()
                        {
                                for (tsize i = t * task_size, iend = std::min(i + task_size, N); i < iend; i ++)
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

        ///
        /// \brief split a loop computation of the given size using multiple threads and cumulate partial results
        ///
        template
        <
                typename tdata,
                typename tsize,
                class toperator_init,
                class toperator,
                class toperator_cumulate
        >
        void thread_loop_cumulate(
                tsize N, toperator_init op_init, toperator op, toperator_cumulate op_cumulate, tsize nthreads = tsize(0))
        {
                thread_pool_t pool(nthreads);
                thread_loop_cumulate<tdata, tsize, toperator_init, toperator, toperator_cumulate>(N, op_init, op, op_cumulate, pool);
        }
}

#endif // NANOCV_THREAD_LOOP_H

