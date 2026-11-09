#include "demo_1f/perception/ai_cup_detector.hpp"
#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

using namespace demo_1f;

namespace {
CupMask cup_mask() {
  cv::Mat mask = cv::Mat::zeros(300, 400, CV_8U);
  cv::rectangle(mask, {150, 50, 100, 180}, cv::Scalar(255), cv::FILLED);
  return {{150, 50, 100, 180}, mask, .9, 41};
}

const cv::Matx33d k(500, 0, 200, 0, 500, 150, 0, 0, 1);
} // namespace

TEST(AiCup, DepthUsesMaskAndDoesNotRequireRim) {
  auto cup = cup_mask();
  cv::Mat depth(300, 400, CV_32F, cv::Scalar(2.5));
  depth.setTo(.5, cup.mask);
  std::string reason;
  auto result = locate_cup_mask({cup}, depth, k, .045, .005, reason);
  ASSERT_TRUE(result) << reason;
  EXPECT_NEAR(result->position[2], .5, 1e-6);
  EXPECT_NEAR(result->coverage, 1, 1e-6);
  EXPECT_NEAR(result->error, .07, 1e-6);
}

TEST(AiCup, MissingDepthNeverBecomesAnObservation) {
  std::string reason;
  auto result = locate_cup_mask(
      {cup_mask()}, cv::Mat(300, 400, CV_32F, cv::Scalar(NAN)), k, .045, .005, reason);
  EXPECT_FALSE(result);
  EXPECT_NE(reason.find("insufficient mask depth"), std::string::npos);
}

TEST(AiCup, MultipleCupsAreAmbiguousEvenWithDifferentScores) {
  std::string reason;
  auto second = cup_mask();
  second.score = .4;
  EXPECT_FALSE(locate_cup_mask(
      {cup_mask(), second}, cv::Mat(300, 400, CV_32F, cv::Scalar(.5)), k, .045, .005, reason));
  EXPECT_NE(reason.find("multiple"), std::string::npos);
}

TEST(AiCup, InvalidCalibrationAndWrongSizeRejected) {
  std::string reason;
  auto bad = k;
  bad(0, 0) = NAN;
  cv::Mat depth(300, 400, CV_32F, cv::Scalar(.5));
  EXPECT_FALSE(locate_cup_mask({cup_mask()}, depth, bad, .045, .005, reason));
  depth.setTo(2.0);
  EXPECT_FALSE(locate_cup_mask({cup_mask()}, depth, k, .045, .005, reason));
}

TEST(AiCup, DecoderUnpadsMasksAndSuppressesDuplicates) {
  int ps[] = {1, 116, 8400}, ms[] = {1, 32, 160, 160};
  cv::Mat predictions(3, ps, CV_32F, cv::Scalar(0)), proto(4, ms, CV_32F, cv::Scalar(0));
  cv::Mat rows(116, 8400, CV_32F, predictions.ptr<float>());
  // 400x300 -> 640x480, top padding=80; expected rect [150,50,100,180].
  for (int i = 0; i < 2; ++i) {
    rows.at<float>(0, i) = 320;
    rows.at<float>(1, i) = 304;
    rows.at<float>(2, i) = 160;
    rows.at<float>(3, i) = 288;
    rows.at<float>(45, i) = .9f - i * .1f;
    rows.at<float>(84, i) = 1;
  }
  std::fill(proto.ptr<float>(), proto.ptr<float>() + 160 * 160, 1.f);
  auto masks = decode_cup_masks(predictions, proto, {400, 300}, {0, 0, 400, 300}, {});
  ASSERT_EQ(masks.size(), 1U);
  EXPECT_EQ(masks[0].bounds, cv::Rect(150, 50, 100, 180));
  EXPECT_EQ(cv::countNonZero(masks[0].mask), 18000);
  // Stronger bottle prediction must not be accepted merely because cup score is high.
  rows.at<float>(43, 0) = .95;
  rows.at<float>(43, 1) = .95;
  EXPECT_TRUE(decode_cup_masks(predictions, proto, {400, 300}, {0, 0, 400, 300}, {}).empty());
}

TEST(AiCup, DecoderRejectsOtherModelContracts) {
  EXPECT_THROW(decode_cup_masks({}, {}, {400, 300}, {0, 0, 400, 300}, {}), std::runtime_error);
}
