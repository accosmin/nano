#pragma once

#include "tensor.h"
#include "core/cast.h"

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
        inline enum_map_t<protocol> enum_string<protocol>()
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
        /// \brief target value of the positive class
        ///
        inline scalar_t pos_target() { return +1; }

        ///
        /// \brief target value of the negative class
        ///
        inline scalar_t neg_target() { return -1; }

        ///
        /// \brief check if a target value maps to a positive class
        ///
        inline bool is_pos_target(const scalar_t target) { return target > 0; }

        ///
        /// \brief target value for multi-class single-label classification problems with [n_labels] classes
        ///
        inline vector_t class_target(const tensor_size_t n_labels)
        {
                return vector_t::Constant(n_labels, neg_target());
        }

        inline vector_t class_target(const tensor_size_t ilabel, const tensor_size_t n_labels)
        {
                auto target = class_target(n_labels);
                if (ilabel >= 0 && ilabel < n_labels)
                {
                        target(ilabel) = pos_target();
                }
                return target;
        }

        ///
        /// \brief target value for multi-class multi-label classification problems based on the sign of the target
        ///
        inline vector_t class_target(const vector_t& outputs)
        {
                vector_t target(outputs.size());
                for (auto i = 0; i < outputs.size(); ++ i)
                {
                        target(i) = is_pos_target(outputs(i)) ? pos_target() : neg_target();
                }
                return target;
        }

        ///
        /// \brief cast tensor dimensions to string.
        ///
        template <std::size_t trank>
        struct to_string_t<tensor_dims_t<trank>>
        {
                static string_t cast(const tensor_dims_t<trank>& dims)
                {
                        std::stringstream s;
                        s << dims;
                        return s.str();
                }
        };
}
