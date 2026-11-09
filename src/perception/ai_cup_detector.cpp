#include "demo_1f/perception/ai_cup_detector.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#ifdef DEMO_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

namespace demo_1f {
namespace {
constexpr int input_size = 640, class_count = 80, mask_count = 32;

cv::Rect image_bounds(cv::Size size) {
  return {0, 0, size.width, size.height};
}

void validate_roi(cv::Size size, const cv::Rect & roi) {
  if (roi.width <= 0 || roi.height <= 0 || (roi & image_bounds(size)) != roi) {
    throw std::invalid_argument("AI requires a valid image ROI");
  }
}

struct Letterbox {
  double sx, sy;
  int width, height, left, top;

  explicit Letterbox(cv::Size size) {
    const double scale =
        std::min(double(input_size) / size.width, double(input_size) / size.height);
    width = std::lround(size.width * scale);
    height = std::lround(size.height * scale);
    sx = double(width) / size.width;
    sy = double(height) / size.height;
    left = (input_size - width) / 2;
    top = (input_size - height) / 2;
  }
};
#ifdef DEMO_TENSORRT
void cuda_check(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

class TrtLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char * message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "TensorRT: %s\n", message);
    }
  }
};

struct DeviceBuffer {
  void * data{nullptr};

  ~DeviceBuffer() {
    if (data) {
      cudaFree(data);
    }
  }
};
#endif
} // namespace

struct AiCupDetector::Impl {
  AiCupConfig config;
  cv::dnn::Net net;
  std::string selected;
#ifdef DEMO_TENSORRT
  TrtLogger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  std::vector<std::unique_ptr<DeviceBuffer>> buffers;
  std::vector<std::string> names;
  std::vector<cv::Mat> tensors;
  int input_index{-1};
  cudaStream_t stream{nullptr};

  ~Impl() {
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }

  void load_engine() {
    std::ifstream file(config.engine, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Cannot open TensorRT engine: " + config.engine);
    }
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
    runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
      throw std::runtime_error("TensorRT runtime creation failed");
    }
    engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine) {
      throw std::runtime_error("TensorRT engine incompatible; rebuild on this GPU");
    }
    context.reset(engine->createExecutionContext());
    if (!context || engine->getNbIOTensors() != 3) {
      throw std::runtime_error("Expected YOLOv8 segmentation engine with three IO tensors");
    }
    cuda_check(cudaStreamCreate(&stream));
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
      std::string name = engine->getIOTensorName(i);
      auto dims = engine->getTensorShape(name.c_str());
      std::vector<int> shape(dims.d, dims.d + dims.nbDims);
      if (engine->getTensorDataType(name.c_str()) != nvinfer1::DataType::kFLOAT ||
          std::any_of(shape.begin(), shape.end(), [](int v) {
            return v <= 0;
          })) {
        throw std::runtime_error("Engine IO must be static float32 (internal FP16 is supported)");
      }
      tensors.emplace_back(dims.nbDims, shape.data(), CV_32F);
      buffers.push_back(std::make_unique<DeviceBuffer>());
      cuda_check(cudaMalloc(&buffers.back()->data, tensors.back().total() * sizeof(float)));
      if (!context->setTensorAddress(name.c_str(), buffers.back()->data)) {
        throw std::runtime_error("TensorRT tensor address failed");
      }
      names.push_back(name);
      if (engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT) {
        input_index = i;
        if (shape != std::vector<int>{1, 3, 640, 640}) {
          throw std::runtime_error("Expected static 640x640 RGB input");
        }
      }
    }
    if (input_index < 0) {
      throw std::runtime_error("Engine has no image input");
    }
  }
#endif
  std::vector<cv::Mat> forward(const cv::Mat & blob) {
    std::vector<cv::Mat> outputs;
    if (selected == "opencv_cpu") {
      net.setInput(blob);
      net.forward(outputs, net.getUnconnectedOutLayersNames());
    }
#ifdef DEMO_TENSORRT
    else {
      cuda_check(cudaMemcpyAsync(buffers[input_index]->data,
          blob.ptr<float>(),
          blob.total() * sizeof(float),
          cudaMemcpyHostToDevice,
          stream));
      if (!context->enqueueV3(stream)) {
        throw std::runtime_error("TensorRT enqueue failed");
      }
      for (size_t i = 0; i < tensors.size(); ++i) {
        if (int(i) == input_index) {
          continue;
        }
        cuda_check(cudaMemcpyAsync(tensors[i].ptr<float>(),
            buffers[i]->data,
            tensors[i].total() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream));
        outputs.push_back(tensors[i]);
      }
      cuda_check(cudaStreamSynchronize(stream));
    }
#endif
    return outputs;
  }
};

AiCupDetector::AiCupDetector(const AiCupConfig & config) : impl_(std::make_unique<Impl>()) {
  if (!std::isfinite(config.score_threshold) || config.score_threshold <= 0 ||
      config.score_threshold > 1 || config.classes.empty() ||
      std::any_of(config.classes.begin(),
          config.classes.end(),
          [](int c) {
            return c < 0 || c >= class_count;
          }) ||
      (config.backend != "auto" && config.backend != "tensorrt" &&
          config.backend != "opencv_cpu")) {
    throw std::invalid_argument("Invalid AI score threshold, COCO class IDs or backend");
  }
  impl_->config = config;
#ifdef DEMO_TENSORRT
  if (config.backend == "tensorrt" || (config.backend == "auto" && !config.engine.empty() &&
                                          std::filesystem::exists(config.engine))) {
    impl_->load_engine();
    impl_->selected = "tensorrt";
  } else
#endif
  {
    if (config.backend == "tensorrt") {
      throw std::runtime_error("This build has no TensorRT 10 support");
    }
    if (config.model.empty() || !std::filesystem::exists(config.model)) {
      throw std::runtime_error("AI ONNX model missing: " + config.model);
    }
    impl_->net = cv::dnn::readNetFromONNX(config.model);
    impl_->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    impl_->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    impl_->selected = "opencv_cpu";
  }
}

AiCupDetector::~AiCupDetector() = default;

std::string AiCupDetector::backend() const {
  return impl_->selected;
}

std::vector<CupMask> AiCupDetector::detect(const cv::Mat & bgr, const cv::Rect & roi) {
  if (bgr.empty() || bgr.type() != CV_8UC3) {
    throw std::invalid_argument("AI expects BGR8");
  }
  validate_roi(bgr.size(), roi);
  Letterbox letterbox(roi.size());
  cv::Mat resized, padded(input_size, input_size, CV_8UC3, cv::Scalar(114, 114, 114));
  cv::resize(bgr(roi), resized, {letterbox.width, letterbox.height});
  resized.copyTo(
      padded(cv::Rect(letterbox.left, letterbox.top, letterbox.width, letterbox.height)));
  auto blob = cv::dnn::blobFromImage(padded, 1. / 255, {}, {}, true, false);
  auto outputs = impl_->forward(blob);
  cv::Mat predictions, prototypes;
  for (const auto & output : outputs) {
    if (output.dims == 3) {
      predictions = output;
    }
    if (output.dims == 4) {
      prototypes = output;
    }
  }
  return decode_cup_masks(predictions, prototypes, bgr.size(), roi, impl_->config);
}

std::vector<CupMask> decode_cup_masks(const cv::Mat & predictions,
    const cv::Mat & prototypes,
    const cv::Size & size,
    const cv::Rect & roi,
    const AiCupConfig & config) {
  validate_roi(size, roi);
  if (predictions.type() != CV_32F || predictions.dims != 3 || predictions.size[0] != 1 ||
      predictions.size[1] != 116 || predictions.size[2] != 8400 || prototypes.type() != CV_32F ||
      prototypes.dims != 4 || prototypes.size[0] != 1 || prototypes.size[1] != mask_count ||
      prototypes.size[2] != 160 || prototypes.size[3] != 160 || !predictions.isContinuous() ||
      !prototypes.isContinuous()) {
    throw std::runtime_error(
        "Unsupported model outputs; require static YOLOv8 COCO segmentation 640, no embedded NMS");
  }
  Letterbox lb(roi.size());
  const int count = predictions.size[2];
  cv::Mat rows(116, count, CV_32F, const_cast<float *>(predictions.ptr<float>()));
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> indexes, classes;
  for (int i = 0; i < count; ++i) {
    int best = -1;
    float score = 0;
    for (int c = 0; c < class_count; ++c) {
      float value = rows.at<float>(4 + c, i);
      if (std::isfinite(value) && value > score) {
        score = value;
        best = c;
      }
    }
    if (score < config.score_threshold || score > 1 ||
        std::find(config.classes.begin(), config.classes.end(), best) == config.classes.end()) {
      continue;
    }
    float x = rows.at<float>(0, i), y = rows.at<float>(1, i), w = rows.at<float>(2, i),
          h = rows.at<float>(3, i);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(w) || !std::isfinite(h) ||
        w <= 0 || h <= 0 || std::abs(x) > 1280 || std::abs(y) > 1280 || w > 1280 || h > 1280) {
      continue;
    }
    int left = std::floor((x - w / 2 - lb.left) / lb.sx),
        top = std::floor((y - h / 2 - lb.top) / lb.sy);
    int right = std::ceil((x + w / 2 - lb.left) / lb.sx),
        bottom = std::ceil((y + h / 2 - lb.top) / lb.sy);
    cv::Rect box = cv::Rect(left + roi.x, top + roi.y, right - left, bottom - top) & roi;
    if (box.width < 6 || box.height < 6) {
      continue;
    }
    boxes.push_back(box);
    scores.push_back(score);
    indexes.push_back(i);
    classes.push_back(best);
  }
  std::vector<int> keep;
  cv::dnn::NMSBoxes(boxes, scores, config.score_threshold, .45f, keep, 1.f, 20);
  cv::Mat proto(mask_count, 160 * 160, CV_32F, const_cast<float *>(prototypes.ptr<float>()));
  std::vector<CupMask> results;
  for (int index : keep) {
    cv::Mat coefficients(1, mask_count, CV_32F);
    for (int m = 0; m < mask_count; ++m) {
      coefficients.at<float>(0, m) = rows.at<float>(84 + m, indexes[index]);
    }
    if (!cv::checkRange(coefficients)) {
      continue;
    }
    cv::Mat logits = coefficients * proto, expanded, unpadded;
    cv::resize(logits.reshape(1, 160), expanded, {640, 640}, 0, 0, cv::INTER_LINEAR);
    cv::resize(expanded(cv::Rect(lb.left, lb.top, lb.width, lb.height)),
        unpadded,
        roi.size(),
        0,
        0,
        cv::INTER_LINEAR);
    cv::Mat full = cv::Mat::zeros(size, CV_8U);
    cv::Mat mask = unpadded > 0; // sigmoid(logit) > 0.5
    mask.copyTo(full(roi));
    cv::Mat bounded = cv::Mat::zeros(size, CV_8U);
    full(boxes[index]).copyTo(bounded(boxes[index]));
    results.push_back({boxes[index], bounded, scores[index], classes[index]});
  }
  return results;
}

std::optional<BodyDetection> locate_cup_mask(const std::vector<CupMask> & candidates,
    const cv::Mat & depth,
    const cv::Matx33d & k,
    double radius,
    double wall_thickness,
    std::string & reason) {
  if (candidates.empty()) {
    reason = "AI: no cup above score threshold";
    return {};
  }
  if (candidates.size() != 1) {
    reason = "AI: multiple cup candidates; narrow ROI to the intended cup";
    return {};
  }
  const auto & cup = candidates.front();
  if (depth.empty() || depth.type() != CV_32FC1 || cup.mask.type() != CV_8UC1 ||
      cup.mask.size() != depth.size() || !std::isfinite(radius) || radius <= 0 ||
      !std::isfinite(wall_thickness) || wall_thickness < 0 || !std::isfinite(k(0, 0)) ||
      k(0, 0) <= 0 || !std::isfinite(k(1, 1)) || k(1, 1) <= 0 || !std::isfinite(k(0, 2)) ||
      !std::isfinite(k(1, 2))) {
    reason = "AI: invalid depth grid or calibration";
    return {};
  }
  auto bounds = cup.bounds & image_bounds(depth.size());
  if (bounds != cup.bounds || bounds.area() <= 0 || bounds.x <= 0 || bounds.y <= 0 ||
      bounds.br().x >= depth.cols || bounds.br().y >= depth.rows) {
    reason = "AI: cup is clipped by image boundary";
    return {};
  }
  cv::Mat interior;
  cv::erode(cup.mask, interior, cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));
  std::vector<double> values;
  std::vector<cv::Vec3d> points;
  int samples = 0, stride = std::max(1, int(std::sqrt(bounds.area() / 2000.0)));
  for (int y = bounds.y; y < bounds.br().y; y += stride) {
    for (int x = bounds.x; x < bounds.br().x; x += stride) {
      if (!interior.at<uint8_t>(y, x)) {
        continue;
      }
      ++samples;
      double z = depth.at<float>(y, x);
      if (std::isfinite(z) && z >= .05 && z <= 3) {
        values.push_back(z);
        points.push_back({double(x), double(y), z});
      }
    }
  }
  double coverage = samples ? double(values.size()) / samples : 0;
  if (values.size() < 30 || coverage < .15) {
    reason = "AI cup found; insufficient mask depth (need 30 samples and 15% coverage)";
    return {};
  }
  std::sort(values.begin(), values.end());
  double z = values[values.size() / 2],
         spread = values[values.size() * 3 / 4] - values[values.size() / 4];
  double diameter = 2 * (radius + wall_thickness);
  if (spread > std::max(.04, diameter * .6)) {
    reason = "AI cup found; inconsistent mask depth";
    return {};
  }
  double width = bounds.width * z / k(0, 0), height = bounds.height * z / k(1, 1);
  if (width < diameter * .4 || width > diameter * 1.8 || height < diameter * .35 ||
      height > diameter * 3.5) {
    reason = "AI cup found; metric dimensions inconsistent with configured cup";
    return {};
  }
  // Median measured 3D coordinates: never use a background pixel at the box centre.
  std::vector<double> xs, ys;
  for (const auto & point : points) {
    if (std::abs(point[2] - z) > std::max(.01, spread)) {
      continue;
    }
    xs.push_back((point[0] - k(0, 2)) * point[2] / k(0, 0));
    ys.push_back((point[1] - k(1, 2)) * point[2] / k(1, 1));
  }
  if (xs.size() < 30) {
    reason = "AI cup found; insufficient consistent depth";
    return {};
  }
  std::sort(xs.begin(), xs.end());
  std::sort(ys.begin(), ys.end());
  reason = cv::format("AI_BODY_COARSE: class=%d score=%.2f depth=%.0f%%; rim not verified",
      cup.class_id,
      cup.score,
      coverage * 100);
  return BodyDetection{{xs[xs.size() / 2], ys[ys.size() / 2], z},
      bounds,
      radius + wall_thickness + .02 + spread,
      coverage};
}

} // namespace demo_1f
