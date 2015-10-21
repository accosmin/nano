#pragma once

#include "arch.h"
#include <iostream>

namespace cortex
{
        ///
        /// \brief logging object that can use any std::ostream (standard streaming & text files)
        ///
        class NANOCV_PUBLIC logger_t
        {
        public:

                ///
                /// \brief constructor
                ///
                logger_t(std::ostream& stream, const char* header, bool flush = true);

                ///
                /// \brief destructor
                ///
                ~logger_t();

                ///
                /// \brief log element
                ///
                template <typename T>
                logger_t& operator<<(const T& data)
                {
                        m_stream << data;
                        return *this;
                }
                logger_t& operator<<(const char* str);
                logger_t& operator<<(std::ostream& (*pf)(std::ostream&));
                logger_t& operator<<(logger_t& (*pf)(logger_t&));

                ///
                /// \brief log tags
                ///
                logger_t& newl();
                logger_t& endl();
                logger_t& done();
                logger_t& flush();

        private:

                ///
                /// \brief log the current time
                ///
                void log_time();

        private:

                // attributes
                std::ostream&   m_stream;       ///< stream to write into
                std::streamsize m_precision;    ///< original precision to restore
                bool            m_flush;
        };

        ///
        /// \brief stream particular tags
        ///
        inline logger_t& newl(logger_t& logger_t)               { return logger_t.newl(); }
        inline logger_t& endl(logger_t& logger_t)               { return logger_t.endl(); }
        inline logger_t& done(logger_t& logger_t)               { return logger_t.done(); }
        inline logger_t& flush(logger_t& logger_t)              { return logger_t.flush(); }

        ///
        /// \brief specific [information, warning, error] line loggers
        ///
        inline logger_t log_info(std::ostream& os = std::cout, bool flush_at_destruction = true)
        {
                return logger_t(os, "info", flush_at_destruction);
        }
        inline logger_t log_warning(std::ostream& os = std::cout, bool flush_at_destruction = true)
        {
                return logger_t(os, "warning", flush_at_destruction);
        }
        inline logger_t log_error(std::ostream& os = std::cerr, bool flush_at_destruction = true)
        {
                return logger_t(os, "error", flush_at_destruction);
        }
}
