#ifndef NANOCV_TIMER_H
#define NANOCV_TIMER_H

#include "ncv_types.h"
#include <chrono>
#include <ratio>
#include <utility>
#include <sstream>
#include <iomanip>

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // timer with milisecond resolution.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        class timer
        {
        public:

                // constructor
                timer() : m_start(now())
                {
                }

                // measure time
                void start()
                {
                        m_start = now();
                }
                string_t elapsed_string() const
                {
                        return miliseconds_to_string(elapsed_miliseconds());
                }
                index_t elapsed_miliseconds() const
                {
                        const auto duration = std::chrono::duration_cast<milliseconds_t>(now() - m_start);
                        return duration.count();
                }

        private:

                typedef std::chrono::steady_clock::time_point           time_t;
                typedef std::chrono::duration<size_t, std::milli>       milliseconds_t;

                // current time point
                static time_t now()
                {
                        return std::chrono::steady_clock::now();
                }

                // transform miliseconds to string (days, hours, minutes, seconds, miliseconds)
                static string_t miliseconds_to_string(index_t count)
                {
                        static const size_t size_second = 1000;
                        static const size_t size_minute = 60 * size_second;
                        static const size_t size_hour = 60 * size_minute;
                        static const size_t size_day = 24 * size_hour;

                        const size_t days = count / size_day; count -= days * size_day;
                        const size_t hours = count / size_hour; count -= hours * size_hour;
                        const size_t minutes = count / size_minute; count -= minutes * size_minute;
                        const size_t seconds = count / size_second; count -= seconds * size_second;
                        const size_t miliseconds = count;

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
                        if (days > 0 || hours > 0 || minutes > 0 || seconds > 0 || miliseconds >= 0)
                        {
                                stream << std::setfill('0') << std::setw(3) << miliseconds << "ms";
                        }

                        return stream.str();
                }

        private:

                // attributes
                time_t          m_start;
        };	
}

#endif // NANOCV_TIMER_H

