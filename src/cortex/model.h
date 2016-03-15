#pragma once

#include "arch.h"
#include "tensor.h"
#include "stringi.h"
#include "util/manager.hpp"

namespace nano
{
        class task_t;
        class model_t;

        ///
        /// \brief stores registered prototypes
        ///
        using model_manager_t = manager_t<model_t>;
        using rmodel_t = model_manager_t::trobject;

        NANO_PUBLIC model_manager_t& get_models();

        ///
        /// \brief generic model used for computing:
        ///     - the output for an image patch
        //      - its parameters gradient
        ///
        class NANO_PUBLIC model_t : public clonable_t<model_t>
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit model_t(const string_t& parameters);

                ///
                /// \brief resize to process new inputs
                ///
                bool resize(const tensor_size_t idims, const tensor_size_t irows, const tensor_size_t icols,
                            const tensor_size_t osize,
                            const bool verbose);

                ///
                /// \brief resize to process new inputs compatible with the given task
                ///
                bool resize(const task_t& task, bool verbose);

                ///
                /// \brief compute the model's output
                ///
                const tensor3d_t& output(const vector_t& input);
                virtual const tensor3d_t& output(const tensor3d_t& input) = 0;

                ///
                /// \brief save its parameters to file
                ///
                bool save(const string_t& path) const;

                ///
                /// \brief load its parameters from file
                ///
                bool load(const string_t& path);

                ///
                /// \brief load its parameters from vector
                ///
                virtual bool load_params(const vector_t& x) = 0;

                ///
                /// \brief save its parameters to vector
                ///
                virtual bool save_params(vector_t& x) const = 0;

                ///
                /// \brief set parameters to zero
                ///
                virtual void zero_params() = 0;

                ///
                /// \brief set parameters to random values
                ///
                virtual void random_params() = 0;

                ///
                /// \brief compute the model's gradient wrt parameters
                ///
                virtual const vector_t& gparam(const vector_t& output) = 0;

                ///
                /// \brief compute the model's gradient wrt inputs
                ///
                virtual const tensor3d_t& ginput(const vector_t& output) = 0;

                // access functions
                tensor_size_t idims() const { return m_idims; }
                tensor_size_t irows() const { return m_irows; }
                tensor_size_t icols() const { return m_icols; }
                tensor_size_t isize() const { return idims() * irows() * icols(); }
                tensor_size_t osize() const { return m_osize; }
                virtual tensor_size_t psize() const = 0;

        protected:

                // resize to new inputs/outputs, returns the number of parameters
                virtual tensor_size_t resize(bool verbose) = 0;

        private:

                // attributes
                tensor_size_t   m_idims, m_irows, m_icols;      ///< input size
                tensor_size_t   m_osize;                        ///< output size
                tensor3d_t      m_idata;                        ///< buffer input tensor
        };
}

