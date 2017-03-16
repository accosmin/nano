#pragma once

#include "task.h"
#include "math/stats.h"

namespace nano
{
        class ibstream_t;
        class obstream_t;

        ///
        /// \brief stores registered prototypes
        ///
        struct model_t;
        using model_manager_t = manager_t<model_t>;
        using rmodel_t = model_manager_t::trobject;

        NANO_PUBLIC model_manager_t& get_models();

        ///
        /// \brief generic model to process fixed-size 3D tensors.
        ///
        struct NANO_PUBLIC model_t : public clonable_t
        {
                /// <entity, timing statistics in microseconds>
                using timing_t = stats_t<size_t>;
                using timings_t = std::map<string_t, timing_t>;

                using clonable_t::clonable_t;

                ///
                /// \brief create a copy of the current object
                ///
                virtual rmodel_t clone() const = 0;

                ///
                /// \brief resize to process new inputs
                ///
                virtual bool configure(const tensor3d_dims_t& idims, const tensor3d_dims_t& odims) = 0;

                ///
                /// \brief resize to process new inputs compatible with the given task
                ///
                bool configure(const task_t& task) { return configure(task.idims(), task.odims()); }

                ///
                /// \brief serialize to disk
                ///
                virtual bool save(const string_t& path) const = 0;
                virtual bool load(const string_t& path) = 0;

                ///
                /// \brief serialize parameters to memory
                ///
                virtual bool save(vector_t& x) const = 0;
                virtual bool load(const vector_t& x) = 0;

                ///
                /// \brief set parameters to  values
                ///
                virtual void random() = 0;

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
                virtual tensor3d_dims_t idims() const = 0;
                virtual tensor3d_dims_t odims() const = 0;

                ///
                /// \brief number of parameters (to optimize)
                ///
                virtual tensor_size_t psize() const = 0;
        };

        ///
        /// \brief check if the given model is compatible with the given task.
        ///
        inline bool operator==(const model_t& model, const task_t& task)
        {
                return  model.idims() == task.idims() &&
                        model.odims() == task.odims();
        }

        inline bool operator!=(const model_t& model, const task_t& task)
        {
                return !(model == task);
        }
}
