#include "ncv_task_class.h"

namespace ncv
{
//        //-------------------------------------------------------------------------------------------------

//        class_task_t::class_task_t(size_t rows, size_t cols, size_t outputs)
//                :       task_t(rows, cols, outputs)
//        {
//        }

//        //-------------------------------------------------------------------------------------------------

//        bool class_task_t::sample(data_enum dtype, size_t n_samples, dataset_t& dataset) const
//        {
//                // Compute sampling statistics
//                sampler_t sampler(n_labels());

//                const indices_t& ilabels = this->ilabels(dtype);
//                for (size_t i = 0; i < ilabels.size(); i ++)
//                {
//                        const index_t ilabel = ilabels[i];
//                        sampler.add(ilabel, 1.0);
//                }

//                sampler.norm(n_samples);
                
//                // Select samples
//                indices_t isamples;
//                for (size_t i = 0; i < ilabels.size(); i ++)
//                {
//                        const size_t ilabel = ilabels[i];
//                        const size_t times = sampler.select(ilabel, 1.0);
                        
//                        isamples.insert(isamples.end(), times, i);
//                }
                
//                // Debug
//                info()  << "sampling " << n_samples << "/" << isamples.size()
//                        << "/" << ilabels.size() << " samples ...";
//                for (size_t i = 0; i < n_labels(); i ++)
//                {
//                        info()  << "sampling label [" << m_labels[i]
//                                << "]: value = " << sampler.value(i)
//                                << ", prob = " << sampler.prob(i)
//                                << ", selected = " << sampler.scount(i);
//                }
                
//                // OK, create dataset
//                to_dataset(dtype, isamples, dataset);
//                return !dataset.empty();
//        }

//        //-------------------------------------------------------------------------------------------------

//        void class_task_t::describe() const
//        {
//                // Build sample statistics
//                counts_t train_counts(n_labels(), 0);
//                counts_t valid_counts(n_labels(), 0);
//                counts_t test_counts(n_labels(), 0);

//                const indices_t& trilabels = ilabels(data_enum::train);
//                const indices_t& vdilabels = ilabels(data_enum::valid);
//                const indices_t& teilabels = ilabels(data_enum::test);

//                for (index_t i = 0; i < trilabels.size(); i ++)
//                {
//                        const index_t l = trilabels[i];
//                        train_counts[l] ++;
//                }
//                for (index_t i = 0; i < vdilabels.size(); i ++)
//                {
//                        const index_t l = vdilabels[i];
//                        valid_counts[l] ++;
//                }
//                for (index_t i = 0; i < teilabels.size(); i ++)
//                {
//                        const index_t l = teilabels[i];
//                        test_counts[l] ++;
//                }

//                // Print task information
//                info()  << "#images = " << n_images() << " ["
//                        << n_images(data_enum::train) << " train + "
//                        << n_images(data_enum::valid) << " valid + "
//                        << n_images(data_enum::test) << " test].";

//                for (size_t l = 0; l < n_labels(); l ++)
//                {
//                        info()  << "[" << (l + 1) << "/" << n_labels() << "] label <" << labels()[l]
//                                << "> found in [" << train_counts[l] << " train + "
//                                << valid_counts[l] << " valid + "
//                                << test_counts[l] << " test] samples.";
//                }
//        }

//        //-------------------------------------------------------------------------------------------------

//        bool class_task_t::test(const model_t& model, const loss_t& loss) const
//        {
////                lvalue = lerror = 0.0;
////                                for (size_t s = 0; s < data.n_samples(); s ++)
////                                {
////                                        const vector_t& targets = data.targets(s);
////                                        const vector_t& scores = model.process(data.inputs(s));

////                                        lvalue += loss.value(targets, scores);
////                                        lerror += loss.error(targets, scores);
////                                }

////                                lvalue *= inverse(data.n_samples());
////                                lerror *= inverse(data.n_samples());

//                return false;
//        }

//        //-------------------------------------------------------------------------------------------------

//        void class_task_t::clear()
//        {
//                images(data_enum::train).clear();
//                images(data_enum::valid).clear();
//                images(data_enum::test).clear();

//                ilabels(data_enum::train).clear();
//                ilabels(data_enum::valid).clear();
//                ilabels(data_enum::test).clear();
//        }

//        //-------------------------------------------------------------------------------------------------

//        void class_task_t::set_labels(const strings_t& labels)
//        {
//                m_labels = labels;

//                m_targets.resize(n_labels());
//                for (size_t i = 0; i < n_labels(); i ++)
//                {
//                        m_targets[i] = class_target(i, n_labels());
//                }
//        }

//        //-------------------------------------------------------------------------------------------------

//        bool class_task_t::add_image(data_enum dtype, const image_t& image, index_t ilabel)
//        {
//                if (ilabel < n_labels() && image.has(to_string(channel_enum::luma)))
//                {
//                        images_t& images = this->images(dtype);
//                        indices_t& ilabels = this->ilabels(dtype);

//                        images.push_back(image);
//                        ilabels.push_back(ilabel);

//                        return true;
//                }
//                else
//                {
//                        return false;
//                }
//        }

//        //-------------------------------------------------------------------------------------------------

//        void class_task_t::to_dataset(data_enum dtype, const indices_t& isamples,
//                dataset_t& dataset) const
//        {
//                const images_t& images = this->images(dtype);
//                const indices_t& ilabels = this->ilabels(dtype);

//                // Allocate dataset
//                dataset.resize(*this);
//                dataset.reserve(isamples.size());

//                for (size_t is = 0; is < isamples.size(); is ++)
//                {
//                        const index_t s = isamples[is];
//                        const image_t& image = images[s];
//                        const size_t ilabel = ilabels[s];

//                        matrix_t nimage;
//                        norm<pixel_t, scalar_t>(
//                                image.get(to_string(channel_enum::luma)), 0x00, 0xFF,
//                                -1.0, 1.0, nimage);

//                        dataset.set(is, mat2vec(nimage), m_targets[ilabel], m_labels[ilabel]);
//                }
//        }

//        //-------------------------------------------------------------------------------------------------
}
