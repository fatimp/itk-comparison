#include <itkImageRegistrationMethodv4.h>
#include <itkEuler3DTransform.h>
#include <itkMeanSquaresImageToImageMetricv4.h>
#include <itkRegularStepGradientDescentOptimizerv4.h>
#include <itkNormalizeImageFilter.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkCenteredTransformInitializer.h>
#include <itkRegistrationParameterScalesFromPhysicalShift.h>
#include <itkMultiThreaderBase.h>
#include <itkTransformFileWriter.h>
#include <itkCommand.h>

class CommandIterationUpdate : public itk::Command {
public:
    using Self = CommandIterationUpdate;
    using Superclass = itk::Command;
    using Pointer = itk::SmartPointer<Self>;
    itkNewMacro(Self);

protected:
    CommandIterationUpdate() = default;

public:
    using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;
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
        std::cout << optimizer->GetCurrentPosition() << std::endl;
    }
};

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

    using FixedImageType = itk::Image<PixelType, Dimension>;
    using MovingImageType = itk::Image<PixelType, Dimension>;
    using TransformType = itk::Euler3DTransform<double>;

    using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;
    using MetricType =
        itk::MeanSquaresImageToImageMetricv4<FixedImageType, MovingImageType>;
    using RegistrationType = itk::
        ImageRegistrationMethodv4<FixedImageType, MovingImageType, TransformType>;

    auto metric = MetricType::New();
    auto optimizer = OptimizerType::New();
    auto registration = RegistrationType::New();

    registration->SetMetric(metric);
    registration->SetOptimizer(optimizer);
    auto transform = TransformType::New();

    using FixedImageReaderType = itk::ImageFileReader<FixedImageType>;
    using MovingImageReaderType = itk::ImageFileReader<MovingImageType>;
    auto fixedImageReader = FixedImageReaderType::New();
    auto movingImageReader = MovingImageReaderType::New();
    fixedImageReader->SetFileName(argv[1]);
    movingImageReader->SetFileName(argv[2]);

    registration->SetFixedImage(fixedImageReader->GetOutput());
    registration->SetMovingImage(movingImageReader->GetOutput());

    using TransformInitializerType =
        itk::CenteredTransformInitializer<TransformType,
                                          FixedImageType,
                                          MovingImageType>;
    auto initializer = TransformInitializerType::New();
    initializer->SetTransform(transform);
    initializer->SetFixedImage(fixedImageReader->GetOutput());
    initializer->SetMovingImage(movingImageReader->GetOutput());
    initializer->MomentsOn();
    initializer->InitializeTransform();

    registration->SetInitialTransform(transform);
    registration->InPlaceOn();

    using ScalesEstimatorType =
        itk::RegistrationParameterScalesFromPhysicalShift<MetricType>;
    auto scalesEstimator = ScalesEstimatorType::New();
    scalesEstimator->SetMetric(metric);
    scalesEstimator->SetTransformForward(true);
    optimizer->SetScalesEstimator(scalesEstimator);
    optimizer->SetDoEstimateLearningRateOnce(true);
    optimizer->SetDoEstimateLearningRateAtEachIteration(false);
    optimizer->SetLearningRate(1.0);
    optimizer->SetMinimumStepLength(0.00001);
    optimizer->SetNumberOfIterations(100);

    auto observer = CommandIterationUpdate::New();
    optimizer->AddObserver(itk::IterationEvent(), observer);

    // constexpr unsigned int numberOfLevels = 3;

    // RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
    // shrinkFactorsPerLevel.SetSize(3);
    // shrinkFactorsPerLevel[0] = 4;
    // shrinkFactorsPerLevel[1] = 2;
    // shrinkFactorsPerLevel[2] = 1;

    // RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
    // smoothingSigmasPerLevel.SetSize(3);
    // smoothingSigmasPerLevel[0] = 2;
    // smoothingSigmasPerLevel[1] = 1;
    // smoothingSigmasPerLevel[2] = 0;
 
    // registration->SetNumberOfLevels(numberOfLevels);
    // registration->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel);
    // registration->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel);

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
        itk::ResampleImageFilter<MovingImageType, FixedImageType>;
    auto resample = ResampleFilterType::New();

    auto resultTransform = registration->GetTransform();
    resample->SetTransform(resultTransform);
    resample->SetInput(movingImageReader->GetOutput());
 
    FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();
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
