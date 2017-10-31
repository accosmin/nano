#pragma once

#include <tuple>
#include <cassert>

namespace nano
{
        ///
        /// \brief stores the tuning result: optimum and its associated tuple of parameters.
        ///
        template <typename toptimum, typename tparameters>
        struct tune_result_t
        {
                ///
                /// \brief constructor
                ///
                tune_result_t() : m_initialized(false), m_optimum(), m_parameters()
                {
                }

                ///
                /// \brief check if the result is initialized (aka tuned)
                ///
                operator bool() const
                {
                        return m_initialized;
                }

                ///
                /// \brief update the optimum (if possible)
                ///
                void update(const toptimum& value, const tparameters& params)
                {
                        if (!m_initialized || value < m_optimum)
                        {
                                m_optimum = value;
                                m_parameters = params;
                                m_initialized = true;
                        }
                }

                ///
                /// \brief returns the optimum configuration
                ///
                const auto& optimum() const
                {
                        assert(m_initialized);
                        return m_optimum;
                }

                ///
                /// \brief returns a particular optimum parameter
                ///
                template <unsigned int index>
                auto param() const
                {
                        assert(m_initialized);
                        return std::get<index>(m_parameters);
                }

                auto param0() const { return param<0>(); }
                auto param1() const { return param<1>(); }
                auto param2() const { return param<2>(); }
                auto param3() const { return param<3>(); }
                auto param4() const { return param<4>(); }

                ///
                /// \brief returns the list of optimum parameters
                ///
                const auto& params() const
                {
                        assert(m_initialized);
                        return m_parameters;
                }

        private:

                // attributes
                bool            m_initialized;
                toptimum        m_optimum;
                tparameters     m_parameters;
        };
}
