#pragma once

#include <cstdlib>
#include "logger.h"
#include "core/timer.h"

namespace nano
{
        ///
        /// \brief utility to log and measure (critical) function calls (aka steps).
        ///
        class checkpoint_t
        {
        public:

                void step(const std::string& step_name)
                {
                        m_step_name = step_name;
                        m_timer.reset();
                }

                ///
                /// \brief check if a critical step failed
                ///
                void critical(const bool ok) const
                {
                        if (ok)
                        {
                                log_info() << m_step_name << ".";
                        }
                        else
                        {
                                log_error() << m_step_name << " failed!";
                                exit(EXIT_FAILURE);
                        }
                }

                ///
                /// \brief measure a non-critical step
                ///
                void measure() const
                {
                        log_info() << m_step_name << " [" << m_timer.elapsed() << "].";
                }

                ///
                /// \brief measure a critical step (exit if failed)
                ///
                void measure(const bool ok) const
                {
                        if (ok)
                        {
                                log_info() << m_step_name << " [" << m_timer.elapsed() << "].";
                        }
                        else
                        {
                                log_error() << m_step_name << " [" << m_timer.elapsed() << "] failed!";
                                exit(EXIT_FAILURE);
                        }
                }

        private:

                // attributes
                timer_t         m_timer;
                std::string     m_step_name;
        };
}
