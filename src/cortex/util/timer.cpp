#include "timer.h"
#include <ratio>
#include <cstddef>
#include <sstream>
#include <iomanip>

namespace cortex
{
        namespace
        {
                auto now()
                {
                        return std::chrono::high_resolution_clock::now();
                }

                std::string miliseconds_to_string(std::size_t miliseconds)
                {
                        static const std::size_t size_second = 1000;
                        static const std::size_t size_minute = 60 * size_second;
                        static const std::size_t size_hour = 60 * size_minute;
                        static const std::size_t size_day = 24 * size_hour;

                        const std::size_t days = miliseconds / size_day; miliseconds -= days * size_day;
                        const std::size_t hours = miliseconds / size_hour; miliseconds -= hours * size_hour;
                        const std::size_t minutes = miliseconds / size_minute; miliseconds -= minutes * size_minute;
                        const std::size_t seconds = miliseconds / size_second; miliseconds -= seconds * size_second;

                        std::stringstream stream;
                        if (days > 0)
                        {
                                stream << days << "d:";
                        }
                        if (days > 0 || hours > 0)
                        {
                                stream << std::setfill('0') << std::setw(2) << hours << "h:";
                        }
                        if (days > 0 || hours > 0 || minutes > 0)
                        {
                                stream << std::setfill('0') << std::setw(2) << minutes << "m:";
                        }
                        if (days > 0 || hours > 0 || minutes > 0 || seconds > 0)
                        {
                                stream << std::setfill('0') << std::setw(2) << seconds << "s:";
                        }
                        stream << std::setfill('0') << std::setw(3) << miliseconds << "ms";

                        return stream.str();
                }
        }

        timer_t::timer_t()
                : m_start(now())
        {
        }

        void timer_t::start()
        {
                m_start = now();
        }

        std::string timer_t::elapsed() const
        {
                return miliseconds_to_string(static_cast<std::size_t>(milliseconds().count()));
        }

        seconds_t timer_t::seconds() const
        {
                return std::chrono::duration_cast<seconds_t>(now() - m_start);
        }

        milliseconds_t timer_t::milliseconds() const
        {
                return std::chrono::duration_cast<milliseconds_t>(now() - m_start);
        }

        microseconds_t timer_t::microseconds() const
        {
                return std::chrono::duration_cast<microseconds_t>(now() - m_start);
        }
}
