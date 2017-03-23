#pragma once

#include "arch.h"
#include <chrono>
#include <string>

namespace nano
{
        using picoseconds_t = std::chrono::duration<long long, std::pico>;
        using nanoseconds_t = std::chrono::duration<long long, std::nano>;
        using microseconds_t = std::chrono::duration<long long, std::micro>;
        using milliseconds_t = std::chrono::duration<long long, std::milli>;
        using seconds_t = std::chrono::duration<long long>;

        using timepoint_t = std::chrono::high_resolution_clock::time_point;

        struct NANO_PUBLIC timer_t
        {
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
