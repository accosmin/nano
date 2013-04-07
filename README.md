**NanoCV** is a small (nano) library consisting of (non-linear) optimization, machine learning and image processing utilities. 

NanoCV consists of three modules: 

* the **core** - containing various utilities (e.g. types, thread pool, numerical optimization), 

* the (computer vision) **tasks** - (e.g. image classification, object detection) and 

* the **models** - trained and evaluated on these tasks  (e.g. linear models, neural networks). 

Command line programs are supplied for testing various components and for interfacing with the library.

The programming style is inspired by STL and boost and hopefully consistent across the library. Also C++11 is heavily used because it makes coding fun and 
easier. The library is written to be cross-platform (having only Boost, Eigen and Qt as dependencies), generic (via templates) and easy to use.


