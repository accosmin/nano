#pragma once

#include <chrono>
#include <string>
#include <cstdio>

namespace nano
{
        using picoseconds_t = std::chrono::duration<long long, std::pico>;
        using nanoseconds_t = std::chrono::duration<long long, std::nano>;
        using microseconds_t = std::chrono::duration<long long, std::micro>;
        using milliseconds_t = std::chrono::duration<long long, std::milli>;
        using seconds_t = std::chrono::duration<long long>;

        using timepoint_t = std::chrono::high_resolution_clock::time_point;

        struct timer_t
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
                /// \brief retrieve the elapsed time in milliseconds
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

                static timepoint_t now()
                {
                        return std::chrono::high_resolution_clock::now();
                }

                static void append(std::string& str, const char* format, const long long value)
                {
                        char buffer[32];
                        snprintf(buffer, sizeof(buffer), format, value);
                        str.append(buffer);
                }

                // attributes
                timepoint_t     m_start;        ///< starting time point
        };

        inline timer_t::timer_t() :
                m_start(now())
        {
        }

        inline seconds_t timer_t::seconds() const
        {
                return std::chrono::duration_cast<seconds_t>(nanoseconds());
        }

        inline milliseconds_t timer_t::milliseconds() const
        {
                return std::chrono::duration_cast<milliseconds_t>(nanoseconds());
        }

        inline microseconds_t timer_t::microseconds() const
        {
                return std::chrono::duration_cast<microseconds_t>(nanoseconds());
        }

        inline nanoseconds_t timer_t::nanoseconds() const
        {
                return std::chrono::duration_cast<nanoseconds_t>(now() - m_start);
        }

        inline std::string timer_t::elapsed() const
        {
                static constexpr long long size_second = 1000;
                static constexpr long long size_minute = 60 * size_second;
                static constexpr long long size_hour = 60 * size_minute;
                static constexpr long long size_day = 24 * size_hour;

                auto milliseconds = this->milliseconds().count();
                const auto days = milliseconds / size_day; milliseconds -= days * size_day;
                const auto hours = milliseconds / size_hour; milliseconds -= hours * size_hour;
                const auto minutes = milliseconds / size_minute; milliseconds -= minutes * size_minute;
                const auto seconds = milliseconds / size_second; milliseconds -= seconds * size_second;

                std::string str;
                if (days > 0)
                {
                        append(str, "%id:", days);
                }
                if (days > 0 || hours > 0)
                {
                        append(str, "%.2ih:", hours);
                }
                if (days > 0 || hours > 0 || minutes > 0)
                {
                        append(str, "%.2im:", minutes);
                }
                if (days > 0 || hours > 0 || minutes > 0 || seconds > 0)
                {
                        append(str, "%.2is:", seconds);
                }
                append(str, "%.3ims", milliseconds);

                return str;
        }
}
