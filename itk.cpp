#include <itkMattesMutualInformationImageToImageMetricv4.h>
//#include <itkMeanSquaresImageToImageMetricv4.h>
#include <itkNormalizeImageFilter.h>
#include <itkImageRegistrationMethodv4.h>
#include <itkEuler3DTransform.h>
#include <itkRegularStepGradientDescentOptimizerv4.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkCenteredTransformInitializer.h>
#include <itkRegistrationParameterScalesFromPhysicalShift.h>
#include <itkMultiThreaderBase.h>
#include <itkTransformFileWriter.h>
#include <itkCommand.h>

using PixelType = float;

class CommandIterationUpdate : public itk::Command {
public:
    using Self = CommandIterationUpdate;
    using Superclass = itk::Command;
    using Pointer = itk::SmartPointer<Self>;
    itkNewMacro(Self);

protected:
    CommandIterationUpdate() = default;

public:
    using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<PixelType>;
    using OptimizerPointer = const OptimizerType *;

    void Execute(itk::Object * caller, const itk::EventObject & event) override {
        Execute((const itk::Object *)caller, event);
    }
 
    void Execute(const itk::Object * object, const itk::EventObject & event) override {
        auto optimizer = static_cast<OptimizerPointer>(object);
        if (!itk::IterationEvent().CheckEvent(&event)) {
            return;
        }

        std::cout << optimizer->GetCurrentIteration() << " " << optimizer->GetValue() << std::endl;
        std::cout << optimizer->GetGradient() << std::endl;
        std::cout << optimizer->GetScales() << std::endl;
        std::cout << optimizer->GetCurrentPosition() << std::endl;
    }
};

template<typename T>
typename T::Pointer loadImage (std::string name) {
    using ImageReaderType = itk::ImageFileReader<T>;
    auto imageReader = ImageReaderType::New();
    imageReader->SetFileName(name);

    /*
     * Normalize pixel intensities. Again, for the first group of
     * testing images this step can be omitted, but it is a general
     * practice to do normalization.
     */
    using NormalizerType = itk::NormalizeImageFilter<T, T>;
    auto normalizer = NormalizerType::New();
    normalizer->SetInput(imageReader->GetOutput());
    normalizer->Update();
    return normalizer->GetOutput();
}

int main (int argc, char * argv[]) {
    if (argc < 5) {
        std::cerr << "Missing Parameters " << std::endl;
        std::cerr << "Usage: " << argv[0];
        std::cerr << "   fixedImageFile  movingImageFile outImage outTransform" << std::endl;
        return EXIT_FAILURE;
    }

    itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(4);

    constexpr unsigned int Dimension = 3;

    using FixedImageType = itk::Image<PixelType, Dimension>;
    using MovingImageType = itk::Image<PixelType, Dimension>;
    using TransformType = itk::Euler3DTransform<PixelType>; // rotation + translation

    using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<PixelType>;
    /*
     * Mean Squares is also a good choise for images with same pixel
     * intensity as in the first testing group. But because the mutual
     * information is a more common choice we use it.
     */
    //using MetricType =
    //    itk::MeanSquaresImageToImageMetricv4<FixedImageType, MovingImageType>;
    using MetricType =
        itk::MattesMutualInformationImageToImageMetricv4<FixedImageType, MovingImageType, FixedImageType, PixelType>;
    using RegistrationType = itk::
        ImageRegistrationMethodv4<FixedImageType, MovingImageType, TransformType>;

    auto metric = MetricType::New();
    auto optimizer = OptimizerType::New();
    auto registration = RegistrationType::New();

    auto fixedImage  = loadImage<FixedImageType>(argv[1]);
    auto movingImage = loadImage<FixedImageType>(argv[2]);

    registration->SetMetric(metric);
    registration->SetOptimizer(optimizer);
    auto transform = TransformType::New();

    registration->SetFixedImage(fixedImage);
    registration->SetMovingImage(movingImage);

    // Rotation is around center of the image.
    using TransformInitializerType =
        itk::CenteredTransformInitializer<TransformType,
                                          FixedImageType,
                                          MovingImageType>;
    auto initializer = TransformInitializerType::New();
    initializer->SetTransform(transform);
    initializer->SetFixedImage(fixedImage);
    initializer->SetMovingImage(movingImage);
    initializer->GeometryOn();
    initializer->InitializeTransform();

    registration->SetInitialTransform(transform);
    registration->InPlaceOn();

    /*
     * Use scales estimator to adjust learning rate for rotation and
     * translation parameters.
     */
    using ScalesEstimatorType =
        itk::RegistrationParameterScalesFromPhysicalShift<MetricType>;
    auto scalesEstimator = ScalesEstimatorType::New();
    scalesEstimator->SetMetric(metric);
    scalesEstimator->SetTransformForward(true);
    optimizer->SetScalesEstimator(scalesEstimator);
    optimizer->SetDoEstimateLearningRateOnce(false);
    optimizer->SetDoEstimateLearningRateAtEachIteration(true);
    optimizer->SetLearningRate(0.1);
    optimizer->SetGradientMagnitudeTolerance(1e-5);
    optimizer->SetMinimumStepLength(1e-5);
    optimizer->SetNumberOfIterations(500);

    auto observer = CommandIterationUpdate::New();
    optimizer->AddObserver(itk::IterationEvent(), observer);

    constexpr unsigned int numberOfLevels = 3;

    // Use multiresolution pyramid with 3 levels
    RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
    shrinkFactorsPerLevel.SetSize(numberOfLevels);
    shrinkFactorsPerLevel[0] = 4;
    shrinkFactorsPerLevel[1] = 2;
    shrinkFactorsPerLevel[2] = 1;

    RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
    smoothingSigmasPerLevel.SetSize(numberOfLevels);
    smoothingSigmasPerLevel[0] = 2;
    smoothingSigmasPerLevel[1] = 1;
    smoothingSigmasPerLevel[2] = 0;
 
    registration->SetNumberOfLevels(numberOfLevels);
    registration->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel);
    registration->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel);

    try {
        registration->Update();
        std::cout << "Optimizer stop condition: "
                  << registration->GetOptimizer()->GetStopConditionDescription()
                  << std::endl;
    } catch (const itk::ExceptionObject & err) {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }

    using ResampleFilterType =
        itk::ResampleImageFilter<MovingImageType, FixedImageType, PixelType>;
    auto resample = ResampleFilterType::New();

    auto resultTransform = registration->GetTransform();
    resample->SetTransform(resultTransform);
    resample->SetInput(movingImage);
 
    resample->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
    resample->SetOutputOrigin(fixedImage->GetOrigin());
    resample->SetOutputSpacing(fixedImage->GetSpacing());
    resample->SetOutputDirection(fixedImage->GetDirection());
    resample->SetDefaultPixelValue(0);

    using OutputPixelType = unsigned char;
    using OutputImageType = itk::Image<OutputPixelType, Dimension>;
    using CastFilterType =
        itk::CastImageFilter<FixedImageType, OutputImageType>;
    using WriterType = itk::ImageFileWriter<OutputImageType>;

    auto writer = WriterType::New();
    auto caster = CastFilterType::New();

    writer->SetFileName(argv[3]);
    caster->SetInput(resample->GetOutput());
    writer->SetInput(caster->GetOutput());
    writer->Update();

    auto transformWriter = itk::TransformFileWriter::New();
    transformWriter->SetInput(resultTransform);
    transformWriter->SetFileName(argv[4]);
    transformWriter->Update();

    return EXIT_SUCCESS;
}
