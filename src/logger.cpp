#include "logger.h"
#include <ctime>
#include <string>
#include <cassert>
#include <iomanip>
#include <iostream>

namespace nano
{
        static std::ostream& get_stream(const logger_t::type type)
        {
                switch (type)
                {
                case logger_t::type::info:      return std::cout;
                case logger_t::type::warn:      return std::cout;
                case logger_t::type::error:     return std::cerr;
                default:                        assert(false); return std::cout;
                }
        }

        static char get_header(const logger_t::type type)
        {
                switch (type)
                {
                case logger_t::type::info:      return 'i';
                case logger_t::type::warn:      return 'w';
                case logger_t::type::error:     return 'e';
                default:                        assert(false); return 'x';
                }
        }

        logger_t::logger_t(const logger_t::type type, bool flush) :
                m_stream(get_stream(type)),
                m_precision(m_stream.precision()),
                m_flush(flush)
        {
                const std::time_t t = std::time(nullptr);
                m_stream << "[";
//                m_stream << std::put_time(std::localtime(&t), "%F|%T");
                char mbstr[100];
                if (std::strftime(mbstr, sizeof(mbstr), "%F|%T", std::localtime(&t)))
                {
                        m_stream << mbstr;
                }
                m_stream << "|" << get_header(type) << "]: ";
                m_stream << std::fixed << std::setprecision(6);
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
}
