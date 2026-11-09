#include "demo_1f/msg/cup_observation.hpp"
#include "demo_1f/perception/ai_cup_detector.hpp"
#include <algorithm>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>

namespace demo_1f {
// Shared perception process: detectors produce observations; no robot commands here.
class PerceptionNode : public rclcpp::Node {
  struct Camera {
    std::string source;
    cv::Rect body_bounds;
    std::unique_ptr<AiCupDetector> ai;
    std::vector<CupMask> masks;
    double inference_ms{0};
    sensor_msgs::msg::Image::ConstSharedPtr rgb, depth;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr info;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub, depth_sub;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub;
    std::vector<int64_t> roi;
    int64_t processed{0};
    bool require_zero_distortion{false};
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debug_status;
  };

public:
  PerceptionNode() : Node("perception") {
    radius_ = declare_parameter("cup.inner_radius", 0.0);
    wall_thickness_ = declare_parameter("cup.wall_thickness", 0.005);
    cup_id_ = declare_parameter("cup.id", std::string("cup"));
    debug_ = declare_parameter("debug.enabled", false);
    full_frame_ = declare_parameter("debug.allow_full_frame_roi", false);
    publisher_ = create_publisher<msg::CupObservation>("/perception/cup", rclcpp::QoS(5));
    for (const std::string source : {"head"}) {
      auto camera = std::make_shared<Camera>();
      camera->source = source;
      std::string prefix = "cameras." + source + ".";
      auto rgb = declare_parameter(prefix + "rgb_topic", std::string{});
      auto depth = declare_parameter(prefix + "depth_topic", std::string{});
      auto info = declare_parameter(prefix + "info_topic", std::string{});
      camera->roi = declare_parameter(prefix + "roi", std::vector<int64_t>{0, 0, 0, 0});
      camera->require_zero_distortion =
          declare_parameter(prefix + "require_zero_distortion", false);
      if (rgb.empty() || depth.empty() || info.empty()) {
        continue;
      }
      {
        AiCupConfig ai;
        const auto models = ament_index_cpp::get_package_share_directory("demo_1f") + "/models/";
        ai.model = declare_parameter("ai.model", models + "yolov8n-seg.onnx");
        ai.engine = declare_parameter("ai.engine", models + "yolov8n-seg.engine");
        ai.backend = declare_parameter("ai.backend", std::string("auto"));
        ai.score_threshold = declare_parameter("ai.score_threshold", .35);
        auto classes = declare_parameter("ai.classes", std::vector<int64_t>{41});
        for (auto id : classes) {
          if (id < 0 || id >= 80) {
            throw std::invalid_argument("AI class ID must be in [0,79]");
          }
        }
        ai.classes.assign(classes.begin(), classes.end());
        camera->ai = std::make_unique<AiCupDetector>(ai);
        RCLCPP_INFO(get_logger(), "Head AI backend: %s", camera->ai->backend().c_str());
      }
      if (debug_) {
        camera->debug_image = create_publisher<sensor_msgs::msg::Image>(
            "~/" + source + "/image", rclcpp::SensorDataQoS());
        camera->debug_status = create_publisher<std_msgs::msg::String>(
            "~/" + source + "/status", rclcpp::QoS(1).transient_local());
      }
      camera->rgb_sub = create_subscription<sensor_msgs::msg::Image>(
          rgb, rclcpp::SensorDataQoS(), [camera](sensor_msgs::msg::Image::ConstSharedPtr m) {
            camera->rgb = m;
          });
      camera->depth_sub = create_subscription<sensor_msgs::msg::Image>(
          depth, rclcpp::SensorDataQoS(), [camera](sensor_msgs::msg::Image::ConstSharedPtr m) {
            camera->depth = m;
          });
      camera->info_sub = create_subscription<sensor_msgs::msg::CameraInfo>(
          info, rclcpp::SensorDataQoS(), [camera](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) {
            camera->info = m;
          });
      cameras_.push_back(camera);
    }
    timer_ = create_wall_timer(std::chrono::milliseconds(100), [this] {
      for (const auto & camera : cameras_) {
        process(*camera);
      }
    });
    RCLCPP_INFO(get_logger(),
        "Head BODY perception ready: %zu camera sources; no motion output",
        cameras_.size());
  }

private:
  void debug(Camera & c, const std::string & detail, const msg::CupObservation * result = nullptr) {
    if (!debug_) {
      return;
    }
    std_msgs::msg::String status;
    status.data = c.source + ": " + detail;
    c.debug_status->publish(status);
    if (!c.rgb) {
      return;
    }
    try {
      auto frame = cv_bridge::toCvCopy(c.rgb, "bgr8");
      if (c.ai) {
        for (const auto & candidate : c.masks) {
          cv::Mat tint = frame->image.clone();
          tint.setTo(cv::Scalar(0, 180, 255), candidate.mask);
          cv::addWeighted(frame->image, .65, tint, .35, 0, frame->image);
          cv::rectangle(frame->image, candidate.bounds, cv::Scalar(0, 180, 255), 2);
          cv::putText(frame->image,
              cv::format("class %d %.2f", candidate.class_id, candidate.score),
              {candidate.bounds.x, std::max(75, candidate.bounds.y - 6)},
              cv::FONT_HERSHEY_SIMPLEX,
              .5,
              {0, 180, 255},
              1);
        }
        cv::putText(frame->image,
            cv::format("AI %s %.0f ms", c.ai->backend().c_str(), c.inference_ms),
            {12, frame->image.rows - 12},
            cv::FONT_HERSHEY_SIMPLEX,
            .5,
            {0, 180, 255},
            1);
      }
      const auto color = result && result->valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 180, 255);
      if (result && result->valid && c.body_bounds.area() > 0) {
        cv::rectangle(frame->image, c.body_bounds, color, 2);
      }
      if (c.roi.size() == 4 && c.roi[2] > 0 && c.roi[3] > 0) {
        cv::rectangle(frame->image, cv::Rect(c.roi[0], c.roi[1], c.roi[2], c.roi[3]), color, 2);
      }
      if (result && result->valid && c.info) {
        const auto & p = result->body_position;
        const auto & k = c.info->p;
        if (p.z > 0) {
          cv::Point center(
              std::lround(k[0] * p.x / p.z + k[2]), std::lround(k[5] * p.y / p.z + k[6]));
          cv::drawMarker(frame->image, center, color, cv::MARKER_CROSS, 24, 2);
          auto coordinates = cv::format("camera XYZ: %.3f %.3f %.3f m", p.x, p.y, p.z);
          cv::putText(frame->image, coordinates, {12, 56}, cv::FONT_HERSHEY_SIMPLEX, .6, color, 2);
        }
      }
      cv::putText(frame->image,
          status.data.substr(0, std::min<size_t>(110, status.data.find('\n'))),
          {12, 28},
          cv::FONT_HERSHEY_SIMPLEX,
          .55,
          color,
          2);
      c.debug_image->publish(*frame->toImageMsg());
    } catch (const std::exception & e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Debug image: %s", e.what());
    }
  }

  void process(Camera & c) {
    if (!c.rgb || !c.depth || !c.info) {
      debug(c,
          std::string("WAITING:") + (!c.rgb ? " RGB" : "") + (!c.depth ? " depth" : "") +
              (!c.info ? " CameraInfo" : ""));
      return;
    }
    auto stamp = rclcpp::Time(c.rgb->header.stamp);
    if (stamp.nanoseconds() != c.processed) {
      c.masks.clear();
    }
    if (stamp.nanoseconds() <= c.processed) {
      if ((now() - stamp).seconds() > .3) {
        debug(c, "STALE: RGB stream stopped or timestamp is old");
      }
      return;
    }
    if (std::abs((stamp - rclcpp::Time(c.depth->header.stamp)).seconds()) > .04) {
      debug(c, "UNSYNCED: RGB/depth timestamp difference exceeds 40 ms");
      return;
    }
    c.processed = stamp.nanoseconds();
    msg::CupObservation output;
    output.header = c.rgb->header;
    output.source = c.source;
    output.cup_id = cup_id_;
    try {
      if (stamp.nanoseconds() <= 0 || (now() - stamp).seconds() > .3 ||
          (now() - stamp).seconds() < -.1) {
        throw std::runtime_error("Stale/future image");
      }
      if (c.rgb->header.frame_id.empty() || c.rgb->header.frame_id != c.depth->header.frame_id ||
          c.rgb->header.frame_id != c.info->header.frame_id || c.info->width != c.rgb->width ||
          c.info->height != c.rgb->height || c.depth->width != c.rgb->width ||
          c.depth->height != c.rgb->height) {
        throw std::runtime_error(
            "RGB/depth/CameraInfo must describe one registered rectified optical grid");
      }
      // P is the rectified camera matrix. The configured RGB topic must already be rectified.
      const auto & p = c.info->p;
      if (c.require_zero_distortion) {
        const auto & k = c.info->k;
        const bool distorted = std::any_of(c.info->d.begin(), c.info->d.end(), [](double value) {
          return !std::isfinite(value) || std::abs(value) > 1e-9;
        });
        const double differences[] = {k[0] - p[0], k[2] - p[2], k[4] - p[5], k[5] - p[6]};
        if (distorted ||
            std::any_of(std::begin(differences), std::end(differences), [](double value) {
              return !std::isfinite(value) || std::abs(value) > 1e-6;
            })) {
          throw std::runtime_error(
              "Raw RGB is distorted or K/P differ; supply a rectified registered RGB/depth pair");
        }
      }
      cv::Matx33d k(p[0], 0, p[2], 0, p[5], p[6], 0, 0, 1);
      auto rgb = cv_bridge::toCvShare(c.rgb, "bgr8");
      cv::Mat depth;
      if (c.depth->encoding == "16UC1") {
        cv_bridge::toCvShare(c.depth)->image.convertTo(depth, CV_32F, .001);
      } else if (c.depth->encoding == "32FC1") {
        depth = cv_bridge::toCvShare(c.depth)->image;
      } else {
        throw std::runtime_error("Depth encoding must be 16UC1 (mm) or 32FC1 (m)");
      }
      if (c.roi.size() != 4) {
        throw std::runtime_error("ROI must be [x,y,width,height]");
      }
      cv::Rect roi(c.roi[0], c.roi[1], c.roi[2], c.roi[3]);
      if (full_frame_ && roi == cv::Rect()) {
        roi = cv::Rect(0, 0, rgb->image.cols, rgb->image.rows);
      }
      c.body_bounds = {};
      output.observation_type = msg::CupObservation::BODY;
      auto begin = std::chrono::steady_clock::now();
      c.masks = c.ai->detect(rgb->image, roi);
      c.inference_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
              .count();
      // A clipped mask cannot provide a whole-body estimate.
      const bool clipped = std::any_of(c.masks.begin(), c.masks.end(), [&](const CupMask & mask) {
        return mask.bounds.x <= roi.x || mask.bounds.y <= roi.y ||
               mask.bounds.br().x >= roi.br().x || mask.bounds.br().y >= roi.br().y;
      });
      std::optional<BodyDetection> body;
      if (clipped) {
        output.detail = "AI cup clipped by ROI; enlarge ROI";
      } else {
        body = locate_cup_mask(c.masks, depth, k, radius_, wall_thickness_, output.detail);
      }
      if (body) {
        output.valid = true;
        output.body_position.x = body->position[0];
        output.body_position.y = body->position[1];
        output.body_position.z = body->position[2];
        output.inner_radius = radius_; // Configured size, not a measured opening.
        output.position_error = body->error;
        output.confidence = body->coverage;
        c.body_bounds = body->bounds;
      }
    } catch (const std::exception & error) {
      output.detail = error.what();
    }
    if (output.valid && (now() - stamp).seconds() > .3) {
      output.valid = false;
      output.detail = "Perception result stale: input-to-output latency exceeds 300 ms";
    }
    publisher_->publish(output);
    debug(c, std::string(output.valid ? "VALID: " : "INVALID: ") + output.detail, &output);
    if (!output.valid) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000, "%s: %s", c.source.c_str(), output.detail.c_str());
    }
  }

  bool debug_, full_frame_;
  double radius_, wall_thickness_;
  std::string cup_id_;
  std::vector<std::shared_ptr<Camera>> cameras_;
  rclcpp::Publisher<msg::CupObservation>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};
} // namespace demo_1f

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<demo_1f::PerceptionNode>());
  } catch (const std::exception & e) {
    std::fprintf(stderr, "perception: %s\n", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
