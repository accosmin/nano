#include "logger.h"
#include <ctime>
#include <string>
#include <iomanip>

namespace cortex
{
        logger_t::logger_t(std::ostream& stream, const char* header, bool flush)
                :       m_stream(stream),
                        m_precision(stream.precision()),
                        m_flush(flush)
        {
                log_time();
                m_stream << "[" << header << "] ";
        }

        logger_t::~logger_t()
        {
                m_flush ? endl() : newl();
                m_stream.precision(m_precision);
        }

        logger_t& logger_t::operator<<(const char* str)
        {
                m_stream << str;
                return *this;
        }

        logger_t& logger_t::operator<<(std::ostream& (*pf)(std::ostream&))
        {
                (*pf)(m_stream);
                return *this;
        }

        logger_t& logger_t::operator<<(logger_t& (*pf)(logger_t&))
        {
                return (*pf)(*this);
        }        

        logger_t& logger_t::newl()
        {
                m_stream << "\n";
                return *this;
        }

        logger_t& logger_t::endl()
        {
                m_stream << std::endl;
                return *this;
        }

        logger_t& logger_t::done()
        {
                m_stream << "<<< program finished correctly >>>";
                return *this;
        }

        logger_t& logger_t::flush()
        {
                m_stream.flush();
                return *this;
        }

        void logger_t::log_time()
        {
                std::time_t t = std::time(nullptr);
                {
                        //m_stream << "[" << std::put_time(std::localtime(&t), "%c %Z") << "] ";
                }
                {
                        char buffer[128];
                        strftime(buffer, 128, "%Y:%m:%d %H:%M:%S", localtime(&t));
                        m_stream << "[" << buffer << "]";
                }
        }
}
