#pragma once

#include "tensor.h"
#include "stringi.h"
#include "text/enum_string.hpp"

namespace nano
{
        ///
        /// \brief
        ///
        enum class protocol
        {
                train = 0,                      ///< training
                valid,                          ///< validation (for tuning hyper-parameters)
                test                            ///< testing
        };

        ///
        /// \brief
        ///
        struct fold_t
        {
                // attributes
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

        ///
        /// \brief
        ///
        struct target_t
        {
                ///
                /// \brief check if this sample is annotated
                ///
                bool annotated() const { return m_target.size() > 0; }

                // attributes
                string_t        m_label;        ///< label (e.g. if classification)
                vector_t        m_target;       ///< target vector to predict (if annotated)
        };
}

// string cast for enumerations
namespace nano
{
        template <>
        inline std::map<nano::protocol, std::string> enum_string<nano::protocol>()
        {
                return
                {
                        { nano::protocol::train,        "train" },
                        { nano::protocol::valid,        "valid" },
                        { nano::protocol::test,         "test" }
                };
        }
}


