#pragma once

#include "pool.h"
#include <cassert>

namespace nano
{
        ///
        /// \brief split a loop computation of the given size using a thread pool.
        /// NB: the operator receives the range [begin, end) to process and the assigned thread index:
        ///     op(begin, end, thread)
        ///
        template <typename tsize, typename toperator>
        void loopit(const tsize size, const tsize max_thread_chunk, const toperator& op)
        {
                auto& pool = thread_pool_t::instance();

                const auto workers = static_cast<tsize>(pool.workers());
                const auto thread_chunk = (size + workers - 1) / workers;
                if (thread_chunk > tsize(0))
                {
                        section_t<future_t> section;
                        for (tsize thread = 0; thread < workers; ++ thread)
                        {
                                const auto begin = thread * thread_chunk;
                                const auto end = std::min(begin + thread_chunk, size);
                                const auto chunk = std::min(thread_chunk, max_thread_chunk);

                                if (begin >= end)
                                {
                                        // not enough data to split to all threads
                                        break;
                                }

                                assert(begin < end && chunk > 0);
                                section.push_back(pool.enqueue([&, begin, end, chunk, thread]()
                                {
                                        for (auto ibegin = begin; ibegin < end; ibegin = std::min(ibegin + chunk, end))
                                        {
                                                op(ibegin, std::min(ibegin + chunk, end), thread);
                                        }
                                }));
                        }
                        // NB: the section is destroyed here waiting for all tasks to finish!
                }
        }

        ///
        /// \brief split a loop computation of the given size using a thread pool.
        /// NB: the operator receives the range [begin, end) to process:
        ///     op(begin, end)
        ///
        template <typename tsize, typename toperator>
        void loopi(const tsize size, const tsize max_thread_chunk, const toperator& op)
        {
                loopit(size, max_thread_chunk, [&] (const tsize begin, const tsize end, const tsize thread)
                {
                        NANO_UNUSED1(thread);
                        op(begin, end);
                });
        }
}
