#ifndef NANOCV_LOGGER_H
#define NANOCV_LOGGER_H

#include <iostream>
#include <iomanip>
#include <ctime>
#include <string>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // logging object that can use any std::ostream (standard streaming & text files).
        /////////////////////////////////////////////////////////////////////////////////////////

        class logger_t
        {
        public:

                // constructor
                logger_t(std::ostream& stream, const char* header, bool flush = true)
                        :       m_stream(stream), m_flush(flush)
                {
                        log_time();
                        m_stream << "[" << header << "] ";
                }

                // destructor
                ~logger_t()
                {
                        m_flush ? endl() : newl();
                }

                // stream data
                template <typename T>
                logger_t& operator<<(const T& data)
                {
                        m_stream << data;
                        return *this;
                }
                logger_t& operator<<(const char* str)
                {
                        m_stream << str;
                        return *this;
                }
                logger_t& operator<<(std::ostream& (*pf)(std::ostream&))
                {
                        (*pf)(m_stream);
                        return *this;
                }

                logger_t& operator<<(logger_t& (*pf)(logger_t&))
                {
                        return (*pf)(*this);
                }

                // stream tags
                logger_t& newl()
                {
                        m_stream << "\n";
                        return *this;
                }
                logger_t& endl()
                {
                        m_stream << std::endl;
                        return *this;
                }

                logger_t& done()
                {
                        m_stream << "<<< program finished correctly >>>";
                        return *this;
                }

                logger_t& flush()
                {
                        m_stream.flush();
                        return *this;
                }

        private:

                // log current time
                void log_time()
                {
                        std::time_t t = std::time(nullptr);
                        {
                                //m_stream << "[" << std::put_time(std::localtime(&t), "%c %Z") << "] ";
                        }
                        {
                                char buffer[128];
                                strftime(buffer, 128, "%Y:%m:%d %H:%M:%S", localtime(&t));
                                m_stream << "[" << buffer << "] ";
                        }
                }


        private:

                // attributes
                std::ostream&   m_stream;
                bool            m_flush;
        };

        // streaming particular tags
        inline logger_t& newl(logger_t& logger_t)         { return logger_t.newl(); }
        inline logger_t& endl(logger_t& logger_t)         { return logger_t.endl(); }
        inline logger_t& done(logger_t& logger_t)         { return logger_t.done(); }
        inline logger_t& flush(logger_t& logger_t)        { return logger_t.flush(); }

        // specific [information, warning, error] line logger_ts
        inline logger_t log_info(std::ostream& os = std::cout, bool flush_at_destruction = true)
        {
                return logger_t(os, "info", flush_at_destruction);
        }
        inline logger_t log_warning(std::ostream& os = std::cout, bool flush_at_destruction = true)
        {
                return logger_t(os, "warning", flush_at_destruction);
        }
        inline logger_t log_error(std::ostream& os = std::cout, bool flush_at_destruction = true)
        {
                return logger_t(os, "error", flush_at_destruction);
        }
}

#endif // NANOCV_LOGGER_H

