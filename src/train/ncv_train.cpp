#include "ncv_train.h"

namespace ncv
{
//        //-------------------------------------------------------------------------------------------------

//        class criterion_t : public function_t
//        {
//        public:
//                // Constructor
//                criterion_t(const dataset_t& data, const loss_t& loss, scalar_t lambda)
//                :       m_data(data), m_loss(loss), m_lambda(lambda), m_evals(0)
//                {
//                }

//                // Destructor
//                virtual ~criterion_t() {}

//                // Evaluate the function and its gradient
//                virtual scalar_t operator()(const vector_t& x, vector_t& g) const { return evaluate(x, g); }

//                // Evaluate the function
//                virtual scalar_t operator()(const vector_t& x) const { return evaluate(x); }

//                // Number of variables
//                virtual size_t n_params() const { return n_outputs() * n_inputs() + n_outputs(); }

//                // Retrieve the model parameters
//                void decode(const vector_t& x, matrix_t& weights, vector_t& bias) const
//                {
//                        _decode(x, weights, bias);
//                }

//                // Access functions
//                size_t n_evals() const { return m_evals; }
//                size_t n_inputs() const { return m_data.n_inputs(); }
//                size_t n_outputs() const { return m_data.n_outputs(); }
//                size_t n_samples() const { return m_data.n_samples(); }

//        private:

//                // Retrieve the model parameters
//                void _decode(const vector_t& x, matrix_t& weights, vector_t& bias) const
//                {
//                        weights.resize(n_outputs(), n_inputs());
//                        bias.resize(n_outputs());

//                        const scalar_t* px = x.data();
//                        px = array2mat(px, weights);
//                        px = array2vec(px, bias);
//                }

//                // Evaluate the function and its gradient
//                scalar_t evaluate(const vector_t& x, vector_t& g) const
//                {
//                        timer_t timer;

//                        // Split the computation (across multiple threads)
//                        scalars_t th_values;
//                        vectors_t th_grads;
//                        thread_loop(
//                                n_samples(),
//                                std::bind(&criterion_t::th_eval_vg, this, std::cref(x), _1, _2, _3, _4),
//                                th_values, th_grads);

//                        // Merge results
//                        scalar_t v = 0.0;
//                        g.resize(n_params());
//                        g.setZero();
//                        for (size_t t = 0; t < n_threads(); t ++)
//                        {
//                                v += th_values[t] * inverse(n_samples());
//                                g += th_grads[t] * inverse(n_samples());
//                        }

//                        // L2-regularization term
//                        v += 0.5 * m_lambda * x.array().square().sum();
//                        for (size_t i = 0; i < (size_t)x.size(); i ++)
//                        {
//                                g(i) += m_lambda * x(i);
//                        }

//                        // Debug
//                        info()  << "loss [=" << v << "], grad = [" << g.minCoeff() << " - " << g.maxCoeff()
//                                << "] evaluated in " << timer.elapsed() << "s.";

//                        m_evals ++;
//                        return v;
//                }

//                // Evaluate the function
//                scalar_t evaluate(const vector_t& x) const
//                {
//                        timer_t timer;

//                        // Split the computation (across multiple threads)
//                        scalars_t th_values;
//                        thread_loop(
//                                n_samples(),
//                                std::bind(&criterion_t::th_eval_v, this, std::cref(x), _1, _2, _3),
//                                th_values);

//                        // Merge results
//                        scalar_t v = 0.0;
//                        for (size_t t = 0; t < n_threads(); t ++)
//                        {
//                                v += th_values[t] * inverse(n_samples());
//                        }

//                        // L2-regularization term
//                        v += 0.5 * m_lambda * x.array().square().sum();

//                        // Debug
//                        info()  << "loss [= " << v << "]" << " evaluated in " << timer.elapsed() << "s.";

//                        m_evals ++;
//                        return v;
//                }

//                // Provides objective function and gradient evaluations for the [sbegin, send) samples.
//                void th_eval_vg(const vector_t& x, size_t sbegin, size_t send, scalar_t& v, vector_t& g) const
//                {
//                        matrix_t weights;
//                        vector_t bias;
//                        decode(x, weights, bias);

//                        vector_t lgrad(n_outputs()), output(n_outputs()), bgrad(n_outputs());
//                        matrix_t wgrad(n_outputs(), n_inputs());

//                        wgrad.setZero();
//                        bgrad.setZero();
//                        v = 0.0;
//                        for (size_t s = sbegin; s < send; s ++)
//                        {
//                                output = weights * m_data.inputs(s) + bias;
//                                v += m_loss.vgrad(m_data.targets(s), output, lgrad);
//                                wgrad.noalias() += lgrad * m_data.inputs(s).transpose();
//                                bgrad.noalias() += lgrad;
//                        }

//                        g.resize(n_params());
//                        std::copy(wgrad.data(), wgrad.data() + wgrad.size(), g.data());
//                        std::copy(bgrad.data(), bgrad.data() + bgrad.size(), g.data() + wgrad.size());
//                }

//                // Provides objective function evaluations for the [sbegin, send) samples.
//                void th_eval_v(const vector_t& x, size_t sbegin, size_t send, scalar_t& v) const
//                {
//                        matrix_t weights;
//                        vector_t bias;
//                        decode(x, weights, bias);

//                        vector_t output(n_outputs());

//                        v = 0.0;
//                        for (size_t s = sbegin; s < send; s ++)
//                        {
//                                output = weights * m_data.inputs(s) + bias;
//                                v += m_loss.value(m_data.targets(s), output);
//                        }
//                }

//        private:

//                // Attributes
//                const dataset_t&	m_data;		// Input dataset
//                const loss_t&		m_loss;		// Input loss
//                scalar_t                m_lambda;       // L1-regularization
//                mutable size_t		m_evals;	// Number of loss evaluations
//        };

//        //-------------------------------------------------------------------------------------------------

//        linear_model_t::linear_model_t(size_t rows, size_t cols, size_t outputs)
//                :       model_t(rows, cols, outputs)
//        {
//                resize(rows, cols, outputs);
//        }

//        //-------------------------------------------------------------------------------------------------

//        void linear_model_t::resize(size_t rows, size_t cols, size_t outputs)
//        {
//                msize_t::resize(rows, cols, outputs);
//                m_weights.resize(n_outputs(), n_inputs());
//                m_bias.resize(n_outputs());
//                m_input.resize(n_inputs());
//                m_output.resize(n_outputs());
//        }

//        //-------------------------------------------------------------------------------------------------

//        void linear_model_t::resize(const msize_t& msize)
//        {
//                resize(msize.n_rows(), msize.n_cols(), msize.n_outputs());
//        }

//        //-------------------------------------------------------------------------------------------------

//        bool linear_model_t::train(const dataset_t& tdata, const dataset_t& vdata, const loss_t& loss,
//                scalar_t eps, size_t iterations)
//        {
//                eps = clamp(eps, 1e-2, 1e-6);
//                iterations = clamp(iterations, 32, 8096);

//                // Check parameters
//                if (tdata != vdata)
//                {
//                        error() << "invalid training parameters!" << std::endl;
//                        return false;
//                }
//                resize(tdata);

//                return optimize(tdata, vdata, loss, eps, iterations);
//        }

//        //-------------------------------------------------------------------------------------------------

//        bool linear_model_t::optimize(const dataset_t& tdata, const dataset_t& vdata, const loss_t& loss,
//                scalar_t eps, size_t iterations)
//        {
//                generalizer_t<vector_t> gen;

//                // Tune the L2 regularization factor
//                static const scalar_t lamdas[] = { 0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 };
//                for (size_t i = 0; i < sizeof(lamdas) / sizeof(scalar_t); i ++)
//                {
//                        const scalar_t lambda = lamdas[i];

//                        criterion_t crit(tdata, loss, lambda);

//                        // Minimize loss
//                        vector_t x(crit.n_params());
//                        x.setZero();
//                        minimize(crit, x, eps, iterations);
//                        crit.decode(x, m_weights, m_bias);

//                        // Check the generalization error
//                        scalar_t tvalue, terror;
//                        scalar_t vvalue, verror;

//                        evaluate(tdata, loss, *this, tvalue, terror);
//                        evaluate(vdata, loss, *this, vvalue, verror);

//                        gen.process(    terror, verror, x,
//                                        "lambda = " + to_string(lambda) +
//                                        ", train = " + to_string(tvalue) + "/" + to_string(terror) +
//                                        ", valid = " + to_string(vvalue) + "/" + to_string(verror) +
//                                        ", evals = " + to_string(crit.n_evals()));

//                        // Debug
//                        info() << gen.last_description() << ".";
//                }

//                // Store the optimum parameters
//                criterion_t crit(tdata, loss, 0.0);
//                crit.decode(gen.model(), m_weights, m_bias);

//                // Debug
//                info() << "optimal [" << gen.description() << "].";

//                return true;
//        }

//        //-------------------------------------------------------------------------------------------------

//        const vector_t& linear_model_t::process(const vector_t& input)
//        {
//                return m_output = m_weights * input + m_bias;
//        }

//        //-------------------------------------------------------------------------------------------------

//        const vector_t& linear_model_t::process(const matrix_t& image, int x, int y)
//        {
//                vector_t input = ivector();
//                mat2vec(image.block(y, x, n_rows(), n_cols()), input);
//                return process(input);
//        }

//        //-------------------------------------------------------------------------------------------------

//        bool linear_model_t::save(const string_t& path) const
//        {
//                std::ofstream ofs(path, std::ios::binary);

//                boost::archive::binary_oarchive oa(ofs);
//                oa << m_weights;
//                oa << m_bias;
//                oa << m_input;
//                oa << m_output;

//                return ofs.good();
//        }

//        //-------------------------------------------------------------------------------------------------

//        bool linear_model_t::load(const string_t& path)
//        {
//                std::ifstream ifs(path, std::ios::binary);

//                boost::archive::binary_iarchive ia(ifs);
//                ia >> m_weights;
//                ia >> m_bias;
//                ia >> m_input;
//                ia >> m_output;

//                return ifs.good();
//        }

//        //-------------------------------------------------------------------------------------------------
}

