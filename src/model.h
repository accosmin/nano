#pragma once

#include "task.h"
#include "chrono/probe.h"

namespace nano
{
        class ibstream_t;
        class obstream_t;

        ///
        /// \brief stores registered prototypes
        ///
        class model_t;
        using model_factory_t = factory_t<model_t>;
        using rmodel_t = model_factory_t::trobject;

        NANO_PUBLIC model_factory_t& get_models();

        ///
        /// \brief generic model to process fixed-size 3D tensors.
        ///
        class NANO_PUBLIC model_t : public configurable_t
        {
        public:
                using configurable_t::configurable_t;

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
                /// \brief serialize model to disk
                ///
                virtual bool save(const string_t& path) const = 0;
                virtual bool load(const string_t& path) = 0;

                ///
                /// \brief serialize parameters to memory
                ///
                virtual vector_t params() const = 0;
                virtual void params(const vector_t&) = 0;

                ///
                /// \brief set parameters to random values
                ///
                virtual void random() = 0;

                ///
                /// \brief compute the model's output given its input
                ///
                virtual const tensor4d_t& output(const tensor4d_t& idata) = 0;

                ///
                /// \brief compute the model's gradient wrt parameters given its output
                ///
                virtual const tensor1d_t& gparam(const tensor4d_t& odata) = 0;

                ///
                /// \brief compute the model's gradient wrt inputs given its output
                ///
                virtual const tensor4d_t& ginput(const tensor4d_t& odata) = 0;

                ///
                /// \brief retrieve timing information for all components
                ///
                virtual probes_t probes() const = 0;

                ///
                /// \brief print a short description of the model
                ///
                virtual void describe() const = 0;

                ///
                /// \brief returns the input/output dimensions
                ///
                virtual tensor3d_dims_t idims() const = 0;
                virtual tensor3d_dims_t odims() const = 0;

                auto isize() const { return nano::size(idims()); }
                auto osize() const { return nano::size(odims()); }

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
