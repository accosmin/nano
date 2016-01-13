#pragma once

#include "arch.h"
#include <chrono>
#include <string>

namespace cortex
{
        using seconds_t = std::chrono::seconds;
        using milliseconds_t = std::chrono::milliseconds;
        using microseconds_t = std::chrono::microseconds;
        using nanoseconds_t = std::chrono::nanoseconds;

        using timepoint_t = std::chrono::high_resolution_clock::time_point;

        class NANOCV_PUBLIC timer_t
        {
        public:

                ///
                /// \brief constructor
                ///
                timer_t();

                ///
                /// \brief retrieve the elapsed time as a string
                ///
                std::string elapsed() const;

                ///
                /// \brief retrieve the elapsed time in seconds
                ///
                seconds_t seconds() const;

                ///
                /// \brief retrieve the elapsed time in miliseconds
                ///
                milliseconds_t milliseconds() const;

                ///
                /// \brief retrieve the elapsed time in microseconds
                ///
                microseconds_t microseconds() const;

                ///
                /// \brief retrieve the elapsed time in nanoseconds
                ///
                nanoseconds_t nanoseconds() const;

        private:

                // attributes
                timepoint_t     m_start;        ///< starting time point
        };
}
