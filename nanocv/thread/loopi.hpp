#pragma once

#include "pool.h"

namespace ncv
{
        namespace thread
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
                void loopi(tsize N, pool_t& pool, toperator op)
                {
                        const tsize n_tasks = static_cast<tsize>(pool.n_workers());
                        const tsize task_size = (N + n_tasks - 1) / n_tasks;

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
                void loopi(tsize N, tsize nthreads, toperator op)
                {
                        pool_t pool(nthreads);
                        loopi(N, pool, op);
                }

                ///
                /// \brief split a loop computation of the given size using all availabe threads
                ///
                template
                <
                        typename tsize,
                        class toperator
                >
                void loopi(tsize N, toperator op)
                {
                        loopi(N, tsize(0), op);
                }
        }
}
