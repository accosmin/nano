#include "task_norb.h"
#include "../loss.h"
#include "../logger.h"
#include "../file/stream.h"
#include "../file/archive.h"

namespace ncv
{
        static const strings_t tlabels =
        {
                "animal",
                "human",
                "plane",
                "truck",
                "car",
                "blank"
        };

        norb_task_t::norb_task_t(const string_t& configuration)
                :       task_t(configuration)
        {
        }

        bool norb_task_t::load(const string_t& dir)
        {
                const size_t n_train_samples = 29160;// * 10;
                const size_t n_test_samples = 29160;// * 2;

                clear_memory(n_train_samples + n_test_samples);

                return  load(dir + "/norb-5x46789x9x18x6x2x108x108-training-01", protocol::train, n_train_samples) &&
                        load(dir + "/norb-5x46789x9x18x6x2x108x108-training-02", protocol::train, n_train_samples) &&
                        load(dir + "/norb-5x46789x9x18x6x2x108x108-training-03", protocol::train, n_train_samples) &&
                        load(dir + "/norb-5x46789x9x18x6x2x108x108-training-04", protocol::train, n_train_samples) &&
                        load(dir + "/norb-5x46789x9x18x6x2x108x108-training-05", protocol::train, n_train_samples) &&
                        load(dir + "/norb-5x46789x9x18x6x2x108x108-training-06", protocol::train, n_train_samples) &&
                        load(dir + "/norb-5x46789x9x18x6x2x108x108-training-07", protocol::train, n_train_samples) &&
                        load(dir + "/norb-5x46789x9x18x6x2x108x108-training-08", protocol::train, n_train_samples) &&
                        load(dir + "/norb-5x46789x9x18x6x2x108x108-training-09", protocol::train, n_train_samples) &&
                        load(dir + "/norb-5x46789x9x18x6x2x108x108-training-10", protocol::train, n_train_samples) &&

                        load(dir + "/norb-5x01235x9x18x6x2x108x108-testing-01", protocol::test, n_test_samples) &&
                        load(dir + "/norb-5x01235x9x18x6x2x108x108-testing-02", protocol::test, n_test_samples);
        }

        static bool read_header(io::stream_t& stream, int32_t& magic, std::vector<int32_t>& dims)
        {
                // read data type & #dimensions
                int32_t ndims;
                if (    !stream.read((char*)&magic, sizeof(int32_t)) ||
                        !stream.read((char*)&ndims, sizeof(int32_t)))
                {
                        return false;
                }

                // read 3D/4D dimensions
                int32_t dim1, dim2, dim3, dim4;
                if (    !stream.read((char*)&dim1, sizeof(int32_t)) ||
                        !stream.read((char*)&dim2, sizeof(int32_t)) ||
                        !stream.read((char*)&dim3, sizeof(int32_t)) ||
                        (ndims > 3 && !stream.read((char*)&dim4, sizeof(int32_t))))
                {
                        return false;
                }

                // OK
                dims.push_back(dim1);
                dims.push_back(dim2);
                dims.push_back(dim3);
                if (ndims > 3)
                {
                        dims.push_back(dim4);
                }

                return true;
        }

        bool norb_task_t::load(const string_t& bfile, protocol p, size_t count)
        {
                return load(bfile + "-dat.mat.gz", bfile + "-cat.mat.gz", p, count);
        }

        bool norb_task_t::load(const string_t& ifile, const string_t& gfile, protocol p, size_t count)
        {
//                 static const int magic_f32 = 0x1E3D4C51;
//                 static const int magic_f64 = 0x1E3D4C53;
                static const int magic_i32 = 0x1E3D4C54;
                static const int magic_i08 = 0x1E3D4C55;
//                 static const int magic_i16 = 0x1E3D4C56;
                
                size_t iindex = n_images();
                size_t icount = 0;
                size_t gcount = 0;
                
                // load images
                const auto iop = [&] (const string_t&, const io::data_t& data)
                {
                        io::stream_t stream(data.data(), data.size());
                        
                        // read header
                        int32_t magic;
                        std::vector<int32_t> dims;
                        if (!read_header(stream, magic, dims))
                        {
                                log_error() << "NORB: failed to read header!";
                                return false;
                        }
                        
                        if (    magic != magic_i08 ||
                                
                                dims.size() != 4 ||                                
                                dims[1] != 2 ||
                                dims[2] != static_cast<int>(irows()) ||
                                dims[3] != static_cast<int>(icols()))
                        {
                                log_error() << "NORB: invalid header!";
                                return false;
                        }
                        
                        // load images
                        const size_t n_cameras = 2;
                        const size_t n_pixels = irows() * icols();
                        const size_t cnt = dims[0];
                        
                        std::vector<char> dimage(n_pixels);                        
                        for (size_t i = 0; i < cnt; i ++)
                        {
                                for (size_t cam = 0; cam < n_cameras && stream.read(dimage.data(), dimage.size()); cam ++)
                                {
                                        image_t image;
                                        image.load_luma(dimage.data(), irows(), icols());
                                        add_image(image);
                                }
                                
                                ++ icount;
                        }
                        
                        return stream.tellg() == stream.size();
                };
                                
                log_info() << "NORB: loading file <" << ifile << "> ...";
                if (!io::decode(ifile, "NORB: ", iop))
                {
                        log_error() << "NORB: failed to load file <" << ifile << ">!";
                        return false;
                }

                // load ground truth
                const auto gop = [&] (const string_t& filename, const io::data_t& data)
                {
                        io::stream_t stream(data.data(), data.size());
                        
                        // read header
                        int32_t magic;
                        std::vector<int32_t> dims;
                        if (!read_header(stream, magic, dims))
                        {
                                log_error() << "NORB: failed to read header!";
                                return false;
                        }

                        if (    magic != magic_i32 ||

                                dims.size() != 3 ||
                                dims[1] != 1)
                        {
                                log_error() << "NORB: invalid header!";
                                return false;
                        }

                        // load annotations
                        const size_t n_cameras = 2;
                        const size_t cnt = dims[0];

                        int32_t label;
                        for (size_t i = 0; i < cnt && stream.read(reinterpret_cast<char*>(&label), sizeof(label)); i ++)
                        {
                                const size_t ilabel = label;                                
                                for (size_t cam = 0; cam < n_cameras; cam ++)
                                {
                                        sample_t sample(iindex, sample_region(0, 0));
                                        if (ilabel < osize())
                                        {
                                                sample.m_label = tlabels[ilabel];
                                                sample.m_target = ncv::class_target(ilabel, osize());
                                        }
                                        sample.m_fold = { 0, p };
                                        add_sample(sample);
                                        
                                        ++ iindex;
                                }
                                
                                ++ gcount;
                        }
                        
                        return stream.tellg() == stream.size();
                };
                
                log_info() << "NORB: loading file <" << gfile << "> ...";
                if (!io::decode(gfile, "NORB: ", gop))
                {
                        log_error() << "NORB: failed to load file <" << gfile << ">!";
                        return false;
                }
                
                // OK
                log_info() << "NORB: loaded " << icount << "/" << gcount << " samples.";
                return (count == gcount) && (count == icount);
        }
}
