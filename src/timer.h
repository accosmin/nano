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
                timer_t() : m_start(now()) {}

                ///
                /// \brief retrieve the elapsed time as a string
                ///
                std::string elapsed() const { return elapsed(milliseconds()); }

                ///
                /// \brief retrieve the elapsed time in seconds
                ///
                seconds_t seconds() const { return std::chrono::duration_cast<seconds_t>(nanoseconds()); }

                ///
                /// \brief retrieve the elapsed time in milliseconds
                ///
                milliseconds_t milliseconds() const { return std::chrono::duration_cast<milliseconds_t>(nanoseconds()); }

                ///
                /// \brief retrieve the elapsed time in microseconds
                ///
                microseconds_t microseconds() const { return std::chrono::duration_cast<microseconds_t>(nanoseconds()); }

                ///
                /// \brief retrieve the elapsed time in nanoseconds
                ///
                nanoseconds_t nanoseconds() const { return std::chrono::duration_cast<nanoseconds_t>(now() - m_start); }

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

                static std::string elapsed(const milliseconds_t duration)
                {
                        static constexpr long long size_second = 1000;
                        static constexpr long long size_minute = 60 * size_second;
                        static constexpr long long size_hour = 60 * size_minute;
                        static constexpr long long size_day = 24 * size_hour;

                        auto milliseconds = duration.count();
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

                // attributes
                timepoint_t     m_start;        ///< starting time point
        };
}
