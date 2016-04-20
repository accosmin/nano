#pragma once

#include <vector>

namespace thread
{
        ///
        /// \brief RAII object to wait for a given set of futures.
        ///
        template <typename tfuture>
        class section_t : public std::vector<tfuture>
        {
        public:

                ///
                /// \brief destructor
                ///
                ~section_t()
                {
                        wait();
                }

                ///
                /// \brief block until all futures are done
                ///
                void wait() const
                {
                        for (auto it = this->cbegin(), end = this->cend(); it != end; ++ it)
                        {
                                it->wait();
                        }
                }
        };
}
