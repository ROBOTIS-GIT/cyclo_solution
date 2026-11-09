#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <vector>

namespace demo_1f {
struct BodyDetection {
  cv::Vec3d position; // Visible surface point, not a measured cup center/opening.
  cv::Rect bounds;
  double error;    // Includes surface-to-axis ambiguity.
  double coverage; // Valid depth fraction.
};

struct CupMask {
  cv::Rect bounds;
  cv::Mat mask; // CV_8U, full input image grid; no depth inferred by the model.
  float score;
  int class_id;
};

struct AiCupConfig {
  std::string model, engine, backend{"auto"};
  float score_threshold{.35f};
  std::vector<int> classes{41}; // COCO cup. Bottles are not implicitly accepted.
};

class AiCupDetector {
public:
  explicit AiCupDetector(const AiCupConfig & config);
  ~AiCupDetector();
  std::vector<CupMask> detect(const cv::Mat & bgr, const cv::Rect & roi);
  std::string backend() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Separate decoding and depth projection permit testing without a GPU/model.
std::vector<CupMask> decode_cup_masks(const cv::Mat & predictions,
    const cv::Mat & prototypes,
    const cv::Size & image_size,
    const cv::Rect & roi,
    const AiCupConfig & config);
std::optional<BodyDetection> locate_cup_mask(const std::vector<CupMask> & candidates,
    const cv::Mat & depth,
    const cv::Matx33d & k,
    double radius,
    double wall_thickness,
    std::string & reason);
} // namespace demo_1f
