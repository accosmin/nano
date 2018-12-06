#pragma once

#include "loss.h"
#include "task.h"
#include "core/stats.h"
#include "core/timer.h"

namespace nano
{
        class model_t;
        using model_factory_t = factory_t<model_t>;
        using rmodel_t = model_factory_t::trobject;

        NANO_PUBLIC model_factory_t& get_models();

        class ibstream_t;
        class obstream_t;

        ///
        /// \brief track the current/optimum model state.
        ///
        class NANO_PUBLIC training_t
        {
        public:

                enum class status
                {
                        better,         ///< performance improved
                        worse,          ///< performance decreased (but not critically)
                        overfit,        ///< overfitting detected (processing should stop)
                        diverge,        ///< divergence detected aka Nan/Inf (processing should stop)
                        failed,         ///< the training loss does not decrease (e.g. the optimization fails)
                };

                ///
                /// \brief measurement (e.g. after a training epoch).
                ///
                struct measurement_t
                {
                        static constexpr scalar_t max()
                        {
                                return std::numeric_limits<scalar_t>::max();
                        }

                        operator bool() const
                        {
                                return std::isfinite(m_value) && std::isfinite(m_error);
                        }

                        bool operator<(const measurement_t& other) const
                        {
                                return  ((*this) ? m_error : measurement_t::max()) <
                                        ((other) ? other.m_error : measurement_t::max());
                        }

                        // attributes
                        scalar_t        m_value{max()}; ///< average loss value
                        scalar_t        m_error{max()}; ///< average error
                };

                ///
                /// \brief training state after a training epoch.
                ///
                struct state_t
                {
                        operator bool() const
                        {
                                return m_train && m_valid && m_test;
                        }

                        bool operator<(const state_t& other) const
                        {
                                // compare (aka tune) on the validation dataset!
                                return m_valid < other.m_valid;
                        }

                        // attributes
                        milliseconds_t  m_milis{0};     ///< (cumulated) elapsed time since the optimization started
                        int             m_epoch{0};     ///<
                        measurement_t   m_train;        ///< measurement on the training dataset
                        measurement_t   m_valid;        ///< measurement on the validation dataset
                        measurement_t   m_test;         ///< measurement on the test dataset
                };
                using states_t = std::vector<state_t>;

                ///
                /// \brief default constructor
                ///
                training_t() = default;

                ///
                /// \brief constructor
                ///
                explicit training_t(string_t config) :
                        m_config(std::move(config))
                {
                }

                ///
                /// \brief update the current/optimum state with a possible better state
                ///
                status update(const state_t&, const int patience);

                ///
                /// \brief check if a valid result
                ///
                operator bool() const
                {
                        return !m_history.empty();
                }

                ///
                /// \brief compare two training histories (on the validation dataset!)
                ///
                bool operator<(const training_t& other) const
                {
                        return m_optimum < other.m_optimum;
                }

                ///
                /// \brief check if the training should be stopped
                ///
                bool is_done() const
                {
                        return  m_status == training_t::status::failed  ||
                                m_status == training_t::status::diverge ||
                                m_status == training_t::status::overfit;
                }

                ///
                /// \brief computes the convergence speed (aka. loss decrease ratio per unit time)
                ///
                scalar_t convergence_speed() const;

                ///
                /// \brief save the training history as csv
                ///
                bool save(const string_t& path) const;

                ///
                /// \brief access functions
                ///
                auto get_status() const { return m_status; }
                const auto& config() const { return m_config; }
                const auto& history() const { return m_history; }
                const auto& optimum() const { return m_optimum; }
                const auto& last() const { return *m_history.rbegin(); }

        private:

                // attributes
                string_t                m_config;                       ///<
                status                  m_status{status::better};       ///<
                state_t                 m_optimum;                      ///< optimum state
                states_t                m_history;                      ///< optimization history
        };

        ///
        /// \brief machine learning model.
        ///
        class NANO_PUBLIC model_t : public json_configurable_t
        {
        public:

                ///
                /// \brief train the model on the given task and using the given loss.
                ///
                virtual training_t train(const task_t&, const size_t fold, const loss_t&) = 0;

                ///
                /// \brief compute the prediction for an input.
                ///
                virtual tensor3d_t output(const tensor3d_t& input) const = 0;

                ///
                /// \brief compute statistics for a given task.
                ///
                struct evaluate_t
                {
                        operator bool() const { return m_values && m_errors; }

                        // attributes
                        stats_t         m_values;       ///< loss values
                        stats_t         m_errors;       ///< error values
                        milliseconds_t  m_millis{0};    ///< average milliseconds per sample
                };

                evaluate_t evaluate(const task_t&, const fold_t&, const loss_t&) const;

                ///
                /// \brief serialize a model to disk
                ///
                static bool save(const string_t& path, const string_t& id, const model_t&);
                static rmodel_t load(const string_t& path);

                ///
                /// \brief serialize the model to disk
                ///
                virtual bool save(obstream_t&) const = 0;
                virtual bool load(ibstream_t&) = 0;

                ///
                /// \brief returns the expected input/output dimensions
                ///
                virtual tensor3d_dim_t idims() const = 0;
                virtual tensor3d_dim_t odims() const = 0;
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

        ///
        /// \brief convenience function to compute the size of the inputs / outputs of the given model.
        ///
        inline auto isize(const model_t& model) { return nano::size(model.idims()); }
        inline auto osize(const model_t& model) { return nano::size(model.odims()); }

        inline auto isize(const model_t& model, const tensor_size_t count) { return count * isize(model); }
        inline auto osize(const model_t& model, const tensor_size_t count) { return count * osize(model); }

        inline auto idims(const model_t& model, const tensor_size_t count) { return cat_dims(count, model.idims()); }
        inline auto odims(const model_t& model, const tensor_size_t count) { return cat_dims(count, model.odims()); }

        template <>
        inline enum_map_t<training_t::status> enum_string<training_t::status>()
        {
                return
                {
                        { training_t::status::better,          "+better" },
                        { training_t::status::worse,           "--worse" },
                        { training_t::status::overfit,         "overfit" },
                        { training_t::status::diverge,         "diverge" },
                        { training_t::status::failed,          "!failed" }
                };
        }

        inline std::ostream& operator<<(std::ostream& os, const training_t::status status)
        {
                return os << to_string(status);
        }

        inline std::ostream& operator<<(std::ostream& os, const training_t::measurement_t& measure)
        {
                return os << measure.m_value << "|" << measure.m_error;
        }
}
