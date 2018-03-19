#pragma once

#include <vector>

namespace nano
{
        ///
        /// \brief RAII object to wait for a given set of futures (aka barrier).
        ///
        template <typename tfuture>
        class section_t
        {
        public:
                ///
                /// \brief destructor
                ///
                ~section_t()
                {
                        // block until all futures are done
                        wait();
                }

                ///
                /// \brief add a new future to wait for.
                ///
                void push_back(tfuture future)
                {
                        m_futures.emplace_back(std::move(future));
                }

        private:

                void wait() const
                {
                        for (const auto& future : m_futures)
                        {
                                future.wait();
                        }
                }

                // attributes
                std::vector<tfuture>    m_futures;
        };
}
