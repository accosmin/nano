#ifndef NANOCV_THREAD_LOOP_H
#define NANOCV_THREAD_LOOP_H

#include "thread_pool.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // utility functions to process a loop using multiple threads.
        //
        // assuming a function <op(i)> to process the i-th element out of N total,
        // then instead of:     for (size_t i = 0; i < N; i ++) { op(i); }
        // we can use:          thread_loop(N, op)
        //
        // to automatically split the loop using as many threads as available on the current platform.
        /////////////////////////////////////////////////////////////////////////////////////////

        // split a loop computation of the given size using multiple threads
        template
        <
                typename tsize,
                class toperator
        >
        void thread_loop(tsize N, toperator op, tsize threads = tsize(0))
        {
                thread_pool_t pool(threads);

                const tsize n_tasks = static_cast<tsize>(pool.n_workers());

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
                typename tdata,
                typename tsize,
                class toperator_init,
                class toperator,
                class toperator_cumulate
        >
        void thread_loop_cumulate(
                tsize N, toperator_init op_init, toperator op, toperator_cumulate op_cumulate, tsize threads = tsize(0))
        {
                thread_pool_t pool(threads);

                const tsize n_tasks = static_cast<tsize>(pool.n_workers());

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

#endif // NANOCV_THREAD_LOOP_H

