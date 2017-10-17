#pragma once

#include "tensor.h"
#include "text/cast.h"

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
        /// \brief dataset sample.
        ///
        struct sample_t
        {
                tensor3d_t      m_input;        ///< input: count x planes x rows x columns
                tensor3d_t      m_target;       ///< desired (ideal) output: count x planes x rows x columns
                string_t        m_label;        ///< classification label (optional)
        };

        ///
        /// \brief dataset samples composing a minibatch.
        ///
        struct minibatch_t
        {
                minibatch_t() = default;

                minibatch_t(const tensor_size_t count, const tensor3d_dims_t& idims, const tensor3d_dims_t& odims) :
                        m_idata(count, std::get<0>(idims), std::get<1>(idims), std::get<2>(idims)),
                        m_odata(count, std::get<0>(odims), std::get<1>(odims), std::get<2>(odims)),
                        m_labels(static_cast<size_t>(count))
                {
                }

                auto count() const { return m_idata.size<0>(); }
                auto idims() const { return tensor3d_dims_t{m_idata.size<1>(), m_idata.size<2>(), m_idata.size<3>()}; }
                auto odims() const { return tensor3d_dims_t{m_odata.size<1>(), m_odata.size<2>(), m_odata.size<3>()}; }

                template <typename titensor, typename totensor>
                void copy(const tensor_size_t index, const titensor& idata, const totensor& odata, const string_t& label)
                {
                        assert(index >= 0 && index < count());
                        assert(idata.dims() == idims());
                        assert(odata.dims() == odims());
                        m_idata.vector(index) = idata.vector();
                        m_odata.vector(index) = odata.vector();
                        m_labels[static_cast<size_t>(index)] = label;
                }

                const auto& idata() const { return m_idata; }
                const auto& odata() const { return m_odata; }
                const auto& labels() const { return m_labels; }

                auto idata() { return m_idata.tensor(); }
                auto odata() { return m_odata.tensor(); }

                auto idata(const tensor_size_t index) { return m_idata.tensor(index); }
                auto odata(const tensor_size_t index) { return m_odata.tensor(index); }

                auto idata(const tensor_size_t index) const { return m_idata.tensor(index); }
                auto odata(const tensor_size_t index) const { return m_odata.tensor(index); }

        private:

                // attributes
                tensor4d_t      m_idata;        ///< inputs: count x planes x rows x columns
                tensor4d_t      m_odata;        ///< desired (ideal) outputs/targets: count x planes x rows x columns
                strings_t       m_labels;       ///< classification labels (optional)
        };

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
        inline vector_t class_target(const tensor_size_t ilabel, const tensor_size_t n_labels)
        {
                vector_t target = vector_t::Constant(n_labels, neg_target());
                if (ilabel < n_labels)
                {
                        target(ilabel) = pos_target();
                }
                return target;
        }

        ///
        /// \brief target value for multi-class multi-label classification problems based on the sign of the target
        ///
        inline vector_t class_target(const vector_t& scores)
        {
                vector_t target(scores.size());
                for (auto i = 0; i < scores.size(); ++ i)
                {
                        target(i) = is_pos_target(scores(i)) ? pos_target() : neg_target();
                }
                return target;
        }
}
