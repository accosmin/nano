#ifndef NANOCV_TIMER_H
#define NANOCV_TIMER_H

#include <chrono>
#include <ratio>
#include <utility>
#include <sstream>
#include <iomanip>

namespace ncv
{
        class timer_t
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
                static std::string miliseconds_to_string(std::size_t count)
                {
                        static const std::size_t size_second = 1000;
                        static const std::size_t size_minute = 60 * size_second;
                        static const std::size_t size_hour = 60 * size_minute;
                        static const std::size_t size_day = 24 * size_hour;

                        const std::size_t days = count / size_day; count -= days * size_day;
                        const std::size_t hours = count / size_hour; count -= hours * size_hour;
                        const std::size_t minutes = count / size_minute; count -= minutes * size_minute;
                        const std::size_t seconds = count / size_second; count -= seconds * size_second;
                        const std::size_t miliseconds = count;

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

        private:

                // attributes
                time_t          m_start;        ///< starting time point
        };
}

#endif // NANOCV_TIMER_H

