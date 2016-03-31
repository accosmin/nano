#pragma once

#include "pool.h"

namespace nano
{
        ///
        /// \brief split a loop computation of the given size using a thread pool.
        /// NB: the operator receives the index of the sample to process and the assigned thread index: op(i, t)
        ///
        template
        <
                typename tsize,
                class toperator
        >
        void loopit(const tsize size, pool_t& pool, const toperator op, const tsize split = 1)
        {
                assert(split > 0);
                const auto n_workers = static_cast<tsize>(pool.n_workers());
                const auto task_size = (size + n_workers * split - 1) / (n_workers * split);

                for (tsize begin = 0, end = 0, t = 0; end < size; t = (t + 1) % n_workers)
                {
                        begin = end;
                        end = std::min(begin + task_size, size);

                        pool.enqueue([=,&op]()
                        {
                                for (tsize i = begin; i < end; ++ i)
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
        void loopit(const tsize size, const tsize nthreads, const toperator op, const tsize split = 1)
        {
                pool_t pool(nthreads);
                loopit(size, pool, op, split);
        }

        ///
        /// \brief split a loop computation of the given size using all available threads
        ///
        template
        <
                typename tsize,
                class toperator
        >
        void loopit(const tsize size, const toperator op, const tsize split = 1)
        {
                loopit(size, tsize(0), op, split);
        }
}
