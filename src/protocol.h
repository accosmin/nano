#pragma once

#include "text/enum_string.h"

namespace nano
{
        ///
        /// \brief dataset splitting protocol.
        ///
        enum class protocol
        {
                train = 0,                      ///< training
                valid,                          ///< validation (for tuning hyper-parameters)
                test                            ///< testing
        };

        template <>
        inline std::map<protocol, std::string> enum_string<protocol>()
        {
                return
                {
                        { protocol::train,        "train" },
                        { protocol::valid,        "valid" },
                        { protocol::test,         "test" }
                };
        }

        ///
        /// \brief dataset splitting fold.
        ///
        struct fold_t
        {
                fold_t(const size_t index = 0, const protocol p = protocol::train) :
                        m_index(index), m_protocol(p)
                {
                }

                size_t          m_index;        ///< fold index
                protocol        m_protocol;     ///<
        };

        inline bool operator==(const fold_t& f1, const fold_t& f2)
        {
                return f1.m_index == f2.m_index && f1.m_protocol == f2.m_protocol;
        }

        inline bool operator<(const fold_t& f1, const fold_t& f2)
        {
                return f1.m_index < f2.m_index || (f1.m_index == f2.m_index && f1.m_protocol < f2.m_protocol);
        }
}
