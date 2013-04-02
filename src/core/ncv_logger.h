#ifndef NANOCV_LOGGER_H
#define NANOCV_LOGGER_H

#include <iostream>
#include <iomanip>
#include <ctime>
#include <string>

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // logging object that can use any std::ostream (standard streaming & text files).
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        class logger
        {
        public:

                // constructor
                logger(std::ostream& stream, const char* header, bool flush = true)
                        :       m_stream(stream), m_flush(flush)
                {
                        log_time();
                        m_stream << "[" << header << "] ";
                }

                // destructor
                ~logger()
                {
                        m_flush ? endl() : newl();
                }

                // stream data
                template <typename T>
                logger& operator<<(const T& data)
                {
                        m_stream << data;
                        return *this;
                }
                logger& operator<<(const char* str)
                {
                        m_stream << str;
                        return *this;
                }

                // stream tags
                logger& operator<<(std::ostream& (*pf)(std::ostream&))
                {
                        (*pf)(m_stream);
                        return *this;
                }

                logger& operator<<(logger& (*pf)(logger&))
                {
                        return (*pf)(*this);
                }

                logger& newl()
                {
                        m_stream << "\n";
                        return *this;
                }
                logger& endl()
                {
                        m_stream << std::endl;
                        return *this;
                }

                logger& done()
                {
                        m_stream << "<<< program finished correctly >>>";
                        return *this;
                }

                logger& flush()
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
        inline logger& newl(logger& logger)         { return logger.newl(); }
        inline logger& endl(logger& logger)         { return logger.endl(); }
        inline logger& done(logger& logger)         { return logger.done(); }
        inline logger& flush(logger& logger)        { return logger.flush(); }

        // specific [information, warning, error] line loggers
        inline logger log_info(std::ostream& os = std::cout, bool flush_at_destruction = false)
        {
                return logger(os, "info", flush_at_destruction);
        }
        inline logger log_warning(std::ostream& os = std::cout, bool flush_at_destruction = false)
        {
                return logger(os, "warning", flush_at_destruction);
        }
        inline logger log_error(std::ostream& os = std::cout, bool flush_at_destruction = false)
        {
                return logger(os, "error", flush_at_destruction);
        }
}

#endif // NANOCV_LOGGER_H

