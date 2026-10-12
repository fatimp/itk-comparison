#include <itkMattesMutualInformationImageToImageMetricv4.h>
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

template<typename PixelType>
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
    imageReader->Update();
    return imageReader->GetOutput();
}

template<typename TransformType, typename ImageType>
typename TransformType::ConstPointer registerImages(ImageType *fixed, ImageType *moving) {
    using PixelType = typename ImageType::PixelType;
    using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<PixelType>;
    using MetricType =
        itk::MattesMutualInformationImageToImageMetricv4<ImageType, ImageType, ImageType, PixelType>;
    using RegistrationType = itk::
        ImageRegistrationMethodv4<ImageType, ImageType, TransformType>;
    using TransformInitializerType =
        itk::CenteredTransformInitializer<TransformType, ImageType, ImageType>;
    using ScalesEstimatorType =
        itk::RegistrationParameterScalesFromPhysicalShift<MetricType>;

    auto metric = MetricType::New();
    auto optimizer = OptimizerType::New();
    auto registration = RegistrationType::New();

    metric->SetUseFixedImageGradientFilter(false);
    metric->SetUseMovingImageGradientFilter(false);

    registration->SetMetric(metric);
    registration->SetOptimizer(optimizer);
    auto transform = TransformType::New();

    registration->SetFixedImage(fixed);
    registration->SetMovingImage(moving);

    // Rotation is around center of the image.

    auto initializer = TransformInitializerType::New();
    initializer->SetTransform(transform);
    initializer->SetFixedImage(fixed);
    initializer->SetMovingImage(moving);
    initializer->GeometryOn();
    initializer->InitializeTransform();

    registration->SetInitialTransform(transform);
    registration->InPlaceOn();

    /*
     * Use scales estimator to adjust learning rate for rotation and
     * translation parameters.
     */
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

    auto observer = CommandIterationUpdate<PixelType>::New();
    optimizer->AddObserver(itk::IterationEvent(), observer);

    constexpr unsigned int numberOfLevels = 3;
    // Use multiresolution pyramid with 3 levels
    typename RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
    shrinkFactorsPerLevel.SetSize(numberOfLevels);
    shrinkFactorsPerLevel[0] = 4;
    shrinkFactorsPerLevel[1] = 2;
    shrinkFactorsPerLevel[2] = 1;

    typename RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
    smoothingSigmasPerLevel.SetSize(numberOfLevels);
    smoothingSigmasPerLevel[0] = 2;
    smoothingSigmasPerLevel[1] = 1;
    smoothingSigmasPerLevel[2] = 0;
 
    registration->SetNumberOfLevels(numberOfLevels);
    registration->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel);
    registration->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel);

    registration->Update();
    std::cout << "Optimizer stop condition: "
              << registration->GetOptimizer()->GetStopConditionDescription()
              << std::endl;

    return registration->GetTransform();
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
    using PixelType = float;
    using ImageType = itk::Image<PixelType, Dimension>;
    using TransformType = itk::Euler3DTransform<PixelType>;
    using ResampleFilterType =
        itk::ResampleImageFilter<ImageType, ImageType, PixelType>;
    using OutputPixelType = unsigned char;
    using OutputImageType = itk::Image<OutputPixelType, Dimension>;
    using CastFilterType =
        itk::CastImageFilter<ImageType, OutputImageType>;
    using WriterType = itk::ImageFileWriter<OutputImageType>;

    try {
        auto fixedImage  = loadImage<ImageType>(argv[1]);
        auto movingImage = loadImage<ImageType>(argv[2]);
        auto transform = registerImages<TransformType, ImageType> (fixedImage, movingImage);

        auto transformWriter = itk::TransformFileWriter::New();
        transformWriter->SetInput(transform);
        transformWriter->SetFileName(argv[4]);
        transformWriter->Update();


        auto resample = ResampleFilterType::New();
        resample->SetTransform(transform);
        resample->SetInput(movingImage);
        resample->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
        resample->SetOutputOrigin(fixedImage->GetOrigin());
        resample->SetOutputSpacing(fixedImage->GetSpacing());
        resample->SetOutputDirection(fixedImage->GetDirection());
        resample->SetDefaultPixelValue(0);

        auto writer = WriterType::New();
        auto caster = CastFilterType::New();

        writer->SetFileName(argv[3]);
        caster->SetInput(resample->GetOutput());
        writer->SetInput(caster->GetOutput());
        writer->Update();
    } catch (const itk::ExceptionObject & err) {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
