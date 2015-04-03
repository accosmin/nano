#pragma once

#include "arch.h"
#include <chrono>
#include <ratio>
#include <string>

namespace ncv
{
        class NANOCV_DLL_PUBLIC timer_t
        {
        public:

                ///
                /// \brief constructor
                ///
                timer_t() : m_start(now())
                {
                }

                ///
                /// \brief reset timer
                ///
                void start()
                {
                        m_start = now();
                }

                ///
                /// \brief retrieve the elapsed time as a string
                ///
                std::string elapsed() const
                {
                        return miliseconds_to_string(miliseconds());
                }

                ///
                /// \brief retrieve the elapsed time in miliseconds
                ///
                std::size_t miliseconds() const
                {
                        const auto duration = std::chrono::duration_cast<milliseconds_t>(now() - m_start);
                        return duration.count();
                }

                ///
                /// \brief retrieve the elapsed time in microseconds
                ///
                std::size_t microseconds() const
                {
                        const auto duration = std::chrono::duration_cast<microseconds_t>(now() - m_start);
                        return duration.count();
                }

        private:

                typedef std::chrono::high_resolution_clock::time_point  time_t;
                typedef std::chrono::duration<std::size_t, std::milli>  milliseconds_t;
                typedef std::chrono::duration<std::size_t, std::micro>  microseconds_t;

                ///
                /// \brief current time point
                ///
                static time_t now()
                {
                        return std::chrono::high_resolution_clock::now();
                }

                ///
                /// \brief transform miliseconds to string (days, hours, minutes, seconds, miliseconds)
                ///
                static std::string miliseconds_to_string(std::size_t count);

        private:

                // attributes
                time_t          m_start;        ///< starting time point
        };
}
