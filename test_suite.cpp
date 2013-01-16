#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/test_case_template.hpp>
#include <boost/mpl/list.hpp>
#include <random>

#include "cl_utils.h"
#include "utils.h"
#include "pyr_impl.h"

using namespace DynamiCL;

typedef boost::mpl::list<int,long,unsigned char> pix_types;

// ========================================================
// TODO: uggggh have to handle manually
struct CLFixture {
    ComputeContext clcontext;
    cl::Program testkernels;

    CLFixture()  
        : clcontext(),
          testkernels(buildProgram(clcontext.context, clcontext.device, "tests.cl"))
    {}
};

BOOST_GLOBAL_FIXTURE( CLFixture );
// ========================================================

BOOST_AUTO_TEST_SUITE(utils)

BOOST_AUTO_TEST_CASE( stripExtension_test )
{
    using namespace std;

    BOOST_CHECK_EQUAL( "", stripExtension("") );
    BOOST_CHECK_EQUAL( "a", stripExtension("a") );
    BOOST_CHECK_EQUAL( "longer", stripExtension("longer") );
    BOOST_CHECK_EQUAL( "", stripExtension(".") );
    BOOST_CHECK_EQUAL( "", stripExtension(".jpg") );
    BOOST_CHECK_EQUAL( "hello", stripExtension("hello.jpg") );
    BOOST_CHECK_EQUAL( ".jpg", stripExtension(".jpg.bmp") );
    BOOST_CHECK_EQUAL( "hello.jpg", stripExtension("hello.jpg.bmp") );
}

BOOST_AUTO_TEST_CASE_TEMPLATE( array_ptr_tests, PixType, pix_types)
{
    typedef array_ptr<PixType> array_type;

    array_type a(50);

    BOOST_CHECK_EQUAL( a.size(), 50 );
    BOOST_CHECK( a.ptr() != nullptr );

    array_type b = std::move(a);

    BOOST_CHECK_EQUAL( a.size(), 0 );
    BOOST_CHECK( a.ptr() == nullptr );
    BOOST_CHECK_EQUAL( b.size(), 50 );
    BOOST_CHECK( b.ptr() != nullptr );

}

BOOST_AUTO_TEST_SUITE_END()
// ========================================================

BOOST_AUTO_TEST_SUITE( host_image )

BOOST_AUTO_TEST_CASE_TEMPLATE( common_ops, PixType, pix_types)
{
    size_t width = 4;
    size_t height = 3;
    HostImage<PixType, 2> image(width, height);

    BOOST_CHECK_EQUAL( image.valid(), true );
    BOOST_CHECK_EQUAL( image.view().totalSize(), width*height );
    BOOST_CHECK_EQUAL( image.view().width(), width );
    BOOST_CHECK_EQUAL( image.view().height(), height );

    char* b = reinterpret_cast<char*>(image.view().begin());
    char* e = reinterpret_cast<char*>(image.view().end());

    BOOST_CHECK_EQUAL( e-b, width*height*sizeof(PixType) );

    // generate random data
    std::vector<PixType> input(image.view().totalSize());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned char> d(0, 255);
    std::generate(input.begin(), input.end(),
                  [&]() { return d(gen); });

    // populate with random data
    std::copy(input.begin(), input.end(), image.view().begin());
    BOOST_CHECK_EQUAL_COLLECTIONS(input.begin(), input.end(),
                                  image.view().begin(), image.view().end());

    // move test
    HostImage<PixType, 2> moved(std::move(image));

    BOOST_CHECK_EQUAL( image.view().valid(), false );
    BOOST_CHECK_EQUAL( image.view().totalSize(), 0 );
    BOOST_CHECK_EQUAL( moved.view().valid(), true );
    BOOST_CHECK_EQUAL( moved.view().totalSize(), width*height );
    BOOST_CHECK_EQUAL_COLLECTIONS(input.begin(), input.end(),
                                  moved.view().begin(), moved.view().end());

}

BOOST_AUTO_TEST_CASE_TEMPLATE( image_array_construction, PixType, pix_types )
{
    size_t width = 40;
    size_t height = 30;
    size_t depth = 50;

    typedef HostImage<PixType, 2> subimage_type;
    typedef HostImageView<PixType, 2> view_type;
    std::vector< subimage_type > subimages;

    std::generate_n(std::back_inserter(subimages), depth,
                    [&]() { return subimage_type(width, height); });

    // assert sizes
    for (auto&& image : subimages)
    {
        BOOST_CHECK_EQUAL( image.valid(), true );
        BOOST_CHECK_EQUAL( image.view().totalSize(), width*height );
        BOOST_CHECK_EQUAL( image.view().width(), width );
        BOOST_CHECK_EQUAL( image.view().height(), height );
    }

    // generate random data
    size_t total = width*height*depth;
    std::vector<PixType> input(total);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned char> d(0, 255);
    std::generate(input.begin(), input.end(),
                  [&]() { return d(gen); });

    // populate with random data
    auto inputIt = input.begin();
    size_t pitch = width*height;
    for (auto&& image : subimages)
    {
        std::copy(inputIt, inputIt + pitch,
                  image.view().begin());
        BOOST_CHECK_EQUAL_COLLECTIONS(inputIt, inputIt + pitch,
                                      image.view().begin(), image.view().end());
        inputIt += pitch;
    }

    // produce views onto subimages
    std::vector<view_type> views;
    for (auto&& image : subimages)
    {
        views.push_back(image.view());
    }

    // create array
    HostImage<PixType, 3> imageArray(views);

    BOOST_CHECK_EQUAL( imageArray.valid(), true );
    BOOST_CHECK_EQUAL( imageArray.view().totalSize(), total );
    BOOST_CHECK_EQUAL( imageArray.view().width(), width );
    BOOST_CHECK_EQUAL( imageArray.view().height(), height );
    BOOST_CHECK_EQUAL( imageArray.view().depth(), depth );
    BOOST_CHECK_EQUAL_COLLECTIONS(input.begin(), input.end(),
                                  imageArray.view().begin(), imageArray.view().end());

    // access through views
    for (size_t i = 0; i < imageArray.view().depth(); ++i)
    {
        HostImageView<PixType, 2> subview = imageArray.view()[i];
        size_t offset = width*height*i;
        size_t offsetEnd = width*height*(i+1);
        BOOST_CHECK_EQUAL_COLLECTIONS(subview.begin(), subview.end(),
                                  imageArray.view().begin() + offset,
                                  imageArray.view().begin() + offsetEnd);

        for (size_t j = 0; j < imageArray.view().height(); ++j)
        {
            HostImageView<PixType, 1> imageLine = subview[j];
            size_t lineOffset = offset + width*j;
            size_t lineOffsetEnd = offset + width*(j+1);
            BOOST_CHECK_EQUAL_COLLECTIONS(imageLine.begin(), imageLine.end(),
                                      imageArray.view().begin() + lineOffset,
                                      imageArray.view().begin() + lineOffsetEnd);

        }
    }

    // collapse dimension
    //HostImage<PixType, 2> collapsed = collapseDimension(std::move(imageArray));
    HostImage<PixType, 2> collapsed(std::move(imageArray));
    BOOST_CHECK_EQUAL( imageArray.valid(), false );
    BOOST_CHECK_EQUAL( imageArray.view().totalSize(), 0 );
    BOOST_CHECK_EQUAL( collapsed.valid(), true );
    BOOST_CHECK_EQUAL( collapsed.view().totalSize(), total );
    BOOST_CHECK_EQUAL( collapsed.view().width(), width );
    BOOST_CHECK_EQUAL( collapsed.view().height(), height*depth );
    BOOST_CHECK_EQUAL_COLLECTIONS(input.begin(), input.end(),
                                  collapsed.view().begin(), collapsed.view().end());

}

BOOST_AUTO_TEST_SUITE_END()
// ========================================================

BOOST_AUTO_TEST_SUITE( cl_common_tests )

BOOST_AUTO_TEST_CASE( halve_test )
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> d(0, 1);

    // create random image
    typedef HostImage<RGBA<float>, 2> image_type;
    image_type image(256, 256);
    std::generate(image.view().begin(), image.view().end(),
                  [&]() { return d(gen); });

    auto pendinginput = makePendingImage(clcontext, image);


    image_type result(image.view().dimensions());
}


BOOST_AUTO_TEST_SUITE_END()
// ========================================================


BOOST_AUTO_TEST_SUITE( pyramid_tests )

BOOST_AUTO_TEST_CASE( pyramid_views )
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> d(0, 500);

    typedef ImagePyramid::pixel_type pixel_type;

    for (size_t example = 0; example < 10; ++example)
    {
        size_t width = d(gen);
        size_t height = d(gen);
        size_t numLevels = calculateNumLevels(width, height);

        array_ptr<pixel_type> ar(pyramidSize(width, height, numLevels));

        auto views =
            ImagePyramid::createPyramidViews(width, height, numLevels, halveDimension, ar.ptr());

        size_t totalBytes = std::accumulate(views.begin(), views.end(), 0,
                [](size_t acc, ImagePyramid::view_type const& v)
                {
                    return acc + v.totalSize();
                });

        BOOST_CHECK_EQUAL( totalBytes, ar.size() );
    }
}


BOOST_AUTO_TEST_SUITE_END()
// ========================================================

