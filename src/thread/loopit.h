#pragma once

#include "pool.h"
#include <cassert>

namespace nano
{
        ///
        /// \brief split a loop computation of the given size using a thread pool.
        /// NB: the operator receives the index of the sample to process and the assigned thread index: op(i, t)
        ///
        template <typename tsize, typename toperator>
        void loopit(const tsize size, const toperator op, const tsize split = 1)
        {
                auto& pool = thread_pool_t::instance();

                assert(split > 0);
                const auto n_workers = static_cast<tsize>(pool.n_workers());
                const auto task_size = (size + n_workers * split - 1) / (n_workers * split);

                section_t<future_t> futures;
                for (tsize begin = 0, end = 0, t = 0; end < size; t = (t + 1) % n_workers)
                {
                        begin = end;
                        end = std::min(begin + task_size, size);

                        futures.push_back(pool.enqueue([=,&op]()
                        {
                                for (tsize i = begin; i < end; ++ i)
                                {
                                        op(i, t);
                                }
                        }));
                }
        }

        ///
        /// \brief split a loop computation of the given size using multiple threads
        ///
        template <typename tsize, typename toperator>
        void loopit(const tsize size, const tsize nthreads, const toperator op, const tsize split = 1)
        {
                thread_pool_t::instance().activate(nthreads);
                loopit(size, op, split);
        }
}
