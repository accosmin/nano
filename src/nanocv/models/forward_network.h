#ifndef NANOCV_FORWARD_NETWORK_H
#define NANOCV_FORWARD_NETWORK_H

#include "model.h"
#include "layer.h"

namespace ncv
{
        ///
        /// multi-layer feed-forward network model
        ///
        class forward_network_t : public model_t
        {
        public:

                NANOCV_MAKE_CLONABLE(forward_network_t)

                using model_t::resize;
                
                ///
                /// \brief constructor
                ///
                forward_network_t(const string_t& parameters = string_t());

                ///
                /// \brief copy constructor
                ///
                forward_network_t(const forward_network_t& other);

                ///
                /// \brief assignment operator
                ///
                forward_network_t& operator=(forward_network_t other);

                ///
                /// \brief compute the model's output
                ///
                virtual const tensor_t& forward(const tensor_t& input) const;

                ///
                /// \brief compute the model's gradient wrt parameters
                ///
                virtual vector_t gradient(const vector_t& output) const;

                ///
                /// \brief compute the model's gradient (wrt parameters & inputs)
                ///
                virtual const tensor_t& backward(const vector_t& output) const;

                ///
                /// \brief save/load/initialize parameters
                ///
                virtual bool load_params(const vector_t& x);
                virtual void zero_params();
                virtual void random_params();

                ///
                /// \brief current parameters
                ///
                virtual vector_t params() const;

                ///
                /// \brief number of parameters
                ///
                virtual size_t psize() const;

                ///
                /// \brief manage layers
                ///
                size_t n_layers() const { return m_layers.size(); }

                bool toggable(size_t layer) const;
                bool enabled(size_t layer) const;

                void enable(size_t layer);
                void disable(size_t layer);

        protected:

                // save/load from file
                virtual bool save(boost::archive::binary_oarchive& oa) const;
                virtual bool load(boost::archive::binary_iarchive& ia);

                // resize to new inputs/outputs, returns the number of parameters
                virtual size_t resize(bool verbose);

        private:

                ///
                /// \brief display the model structure
                ///
                void print(const strings_t& layer_ids) const;

                ///
                /// \brief feed-forward layer
                ///
                struct flayer_t
                {
                        flayer_t(const rlayer_t& layer = rlayer_t(), bool enable = true)
                                :       m_layer(layer), m_enable(enable)
                        {
                        }

                        bool toggable() const { return m_layer->psize() > 0; }
                        bool enabled() const { return m_enable; }
                        void enable() { m_enable = true; }
                        void disable() { m_enable = false; }

                        // attributes
                        rlayer_t        m_layer;                ///< layer
                        bool            m_enable;               ///< enable flag
                };

                typedef std::vector<flayer_t>   flayers_t;

        private:

                // attributes
                flayers_t               m_layers;               ///< feed-forward layers
        };
}

#endif // NANOCV_FORWARD_NETWORK_H
