#pragma once

#include "arch.h"
#include <chrono>
#include <string>

namespace cortex
{
        class NANOCV_PUBLIC timer_t
        {
        public:

                ///
                /// \brief constructor
                ///
                timer_t();

                ///
                /// \brief reset timer
                ///
                void start();

                ///
                /// \brief retrieve the elapsed time as a string
                ///
                std::string elapsed() const;

                ///
                /// \brief retrieve the elapsed time in seconds
                ///
                std::size_t seconds() const;

                ///
                /// \brief retrieve the elapsed time in miliseconds
                ///
                std::size_t miliseconds() const;

                ///
                /// \brief retrieve the elapsed time in microseconds
                ///
                std::size_t microseconds() const;

        private:

                typedef std::chrono::high_resolution_clock::time_point  time_t;

        private:

                // attributes
                time_t          m_start;        ///< starting time point
        };
}
