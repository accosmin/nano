#pragma once

#include "arch.h"
#include "tensor.h"
#include "manager.h"
#include "math/stats.h"

namespace nano
{
        class task_t;
        class ibstream_t;
        class obstream_t;

        ///
        /// \brief stores registered prototypes
        ///
        class model_t;
        using model_manager_t = manager_t<model_t>;
        using rmodel_t = model_manager_t::trobject;

        NANO_PUBLIC model_manager_t& get_models();

        ///
        /// \brief generic model to process fixed-size 3D tensors.
        ///
        class NANO_PUBLIC model_t : public clonable_t
        {
        public:

                /// <entity, timing statistics in microseconds>
                using timing_t = stats_t<size_t>;
                using timings_t = std::map<string_t, timing_t>;

                ///
                /// \brief constructor
                ///
                explicit model_t(const string_t& parameters);

                ///
                /// \brief create a copy of the current object
                ///
                virtual rmodel_t clone() const = 0;

                ///
                /// \brief resize to process new inputs
                ///
                bool resize(const dim3d_t& idims, const dim3d_t& odims);

                ///
                /// \brief resize to process new inputs compatible with the given task
                ///
                bool resize(const task_t& task);

                ///
                /// \brief serialize to disk
                ///
                bool save(const string_t& path) const;
                bool load(const string_t& path);

                ///
                /// \brief number of parameters (to optimize)
                ///
                virtual tensor_size_t psize() const = 0;

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
                /// \brief compute the model's output
                ///
                virtual const tensor3d_t& output(const tensor3d_t& input) = 0;

                ///
                /// \brief compute the model's gradient wrt parameters
                ///
                virtual const vector_t& gparam(const vector_t& output) = 0;

                ///
                /// \brief compute the model's gradient wrt inputs
                ///
                virtual const tensor3d_t& ginput(const vector_t& output) = 0;

                ///
                /// \brief retrieve timing information (in microseconds) regarding various components
                ///      for the three basic operations (output, gradient wrt parameters, gradient wrt inputs)
                ///
                virtual timings_t timings() const = 0;

                ///
                /// \brief print a short description of the model
                ///
                virtual void describe() const = 0;

                ///
                /// \brief returns the input/output dimensions
                ///
                dim3d_t idims() const { return m_idims; }
                dim3d_t odims() const { return m_odims; }

        protected:

                virtual bool resize() = 0;
                virtual bool save(obstream_t&) const = 0;
                virtual bool load(ibstream_t&) = 0;

        private:

                // attributes
                dim3d_t         m_idims;
                dim3d_t         m_odims;
        };

        ///
        /// \brief check if the given model is compatible with the given task.
        ///
        NANO_PUBLIC bool operator==(const model_t& model, const task_t& task);
        NANO_PUBLIC bool operator!=(const model_t& model, const task_t& task);
}

