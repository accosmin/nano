#pragma once

#include "pool.h"

namespace thread
{
        ///
        /// \brief split a loop computation of the given size using a thread pool.
        /// NB: the operator receives the index of the sample to process: op(i)
        ///
        template
        <
                typename tsize,
                class toperator
        >
        void loopi(const tsize size, const toperator op, const tsize split = 1)
        {
                auto& pool = pool_t::instance();

                assert(split > 0);
                const auto n_workers = static_cast<tsize>(pool.n_active_workers());
                const auto task_size = (size + n_workers * split - 1) / (n_workers * split);

                section_t<future_t> futures;
                for (tsize begin = 0, end = 0; end < size; )
                {
                        begin = end;
                        end = std::min(begin + task_size, size);

                        futures.push_back(pool.enqueue([=,&op]()
                        {
                                for (tsize i = begin; i < end; ++ i)
                                {
                                        op(i);
                                }
                        }));
                }
        }

        ///
        /// \brief split a loop computation of the given size using the given number of threads.
        ///
        template
        <
                typename tsize,
                class toperator
        >
        void loopi(const tsize size, const tsize nthreads, const toperator op, const tsize split = 1)
        {
                pool_t::instance().activate(nthreads);
                loopi(size, op, split);
        }
}
