#pragma once

#include "cortex/task.h"

namespace nano
{
        ///
        /// \brief task that stores in memory a collection of annotated 3D input tensors.
        ///
        /// parameters:
        ///     idims           - input size
        ///     irows           - input size
        ///     icols           - input size
        ///     osize           - output size
        ///     count           - number of samples (training + validation + test)
        ///
        class NANO_PUBLIC mem_tensor_task_t : public task_t
        {
        public:

                ///
                /// \brief constructor
                ///
                mem_tensor_task_t(
                        const tensor_size_t idims, const tensor_size_t irows, const tensor_size_t icols,
                        const tensor_size_t osize) :
                        m_idims(idims), m_irows(irows), m_icols(icols), m_osize(osize)
                {
                }

                ///
                /// \brief destructor
                ///
                virtual ~mem_tensor_task_t()
                {
                }

                ///
                /// \brief input size
                ///
                virtual tensor_size_t idims() const override final { return m_idims; }
                virtual tensor_size_t irows() const override final { return m_irows; }
                virtual tensor_size_t icols() const override final { return m_icols; }

                ///
                /// \brief output size
                ///
                virtual tensor_size_t osize() const override final { return m_osize; }

                ///
                /// \brief number of folds (not considering the protocol!)
                ///
                virtual size_t n_folds() const override final;

                ///
                /// \brief total number of samples
                ///
                virtual size_t n_samples() const override final;

                ///
                /// \brief number of samples for the given fold
                ///
                virtual size_t n_samples(const fold_t&) const override final;

                ///
                /// \brief randomly shuffle the samples associated for the given fold
                ///
                virtual void shuffle(const fold_t&) const override final;

                ///
                /// \brief retrieve the 3D input tensor for a given sample
                ///
                virtual tensor3d_t input(const fold_t&, const size_t index) const;

                ///
                /// \brief retrieve the target for a given sample
                ///
                virtual target_t target(const fold_t&, const size_t index) const;

        protected:

                void clear(const fold_t&, const size_t count);
                void reserve(const fold_t&, const size_t count);
                void push_back(const fold_t&, const tensor3d_t& input, const target_t& target);

        private:

                using sample_t  = std::pair<tensor3d_t, target_t>;
                using samples_t = std::vector<sample_t>;
                using storage_t = std::map<fold_t, samples_t>;

        private:

                // attributes
                tensor_size_t   m_idims;        ///< input size
                tensor_size_t   m_icols;
                tensor_size_t   m_irows;
                tensor_size_t   m_osize;        ///< output size
                storage_t       m_data;         ///< stored samples (training, validation, test)
        };
}
