#define USE_CAFFE

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

#include "openpose_ros/PersonPose.h"
#include "openpose_ros/PersonPoseArray.h"


DEFINE_int32(logging_level,             4,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                                                        " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                                                        " low priority messages and 4 for important ones.");
// Producer
DEFINE_string(image_path,               "/home/kerberos/openpose/examples/media/COCO_val2014_000000000192.jpg",     "Process the desired image.");
// OpenPose
DEFINE_string(model_pose,               "COCO",         "Model to be used (e.g. COCO, MPI, MPI_4_layers).");
DEFINE_string(model_folder,             "/home/kerberos/openpose/models/",      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(net_resolution,           "656x368",      "Multiples of 16. If it is increased, the accuracy usually increases. If it is decreased,"
                                                        " the speed increases.");
DEFINE_string(resolution,               "640x480",     "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
                                                        " default images resolution.");
DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless num_scales>1. Initial scale is always 1. If you"
                                                        " want to change the initial scale, you actually want to multiply the `net_resolution` by"
                                                        " your desired initial scale.");
DEFINE_int32(num_scales,                1,              "Number of scales to average.");
// OpenPose Rendering
DEFINE_bool(disable_blending,           false,          "If blending is enabled, it will merge the results with the original frame. If disabled, it"
                                                        " will only display the results.");
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
                                                        " hide it. Only valid for GPU rendering.");

// 未実装
DEFINE_bool(disable_render,             false,          "if true, this node doesn't output the topic /openpose/image_raw");


class OpenposeDetecter
{
  ros::NodeHandle n;
  ros::Subscriber image_sub;
  ros::Publisher image_pub, poses_pub;

  op::CvMatToOpInput *cvMatToOpInput;
  op::CvMatToOpOutput *cvMatToOpOutput;
  op::PoseExtractorCaffe *poseExtractorCaffe;
  op::PoseRenderer *poseRenderer;
  op::OpOutputToCvMat *opOutputToCvMat;

public:
  OpenposeDetecter()
  {
    // Step 1 - logging levelを設定する "0 : 全部loggingする" , "255 : loggingしない"
    op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);

    // Step 2 - Google flags からパラメータを読み出す
    op::Point<int> outputSize;
    op::Point<int> netInputSize;
    op::Point<int> netOutputSize;
    op::PoseModel poseModel;
    std::tie(outputSize, netInputSize, netOutputSize, poseModel) = gflagsToOpParameters();

    // Step 3 - ディープラーニングの初期化
    op::CvMatToOpInput cvMatToOpInput_{netInputSize, FLAGS_num_scales, (float)FLAGS_scale_gap};
    cvMatToOpInput = &cvMatToOpInput_;
    op::CvMatToOpOutput cvMatToOpOutput_{outputSize};
    cvMatToOpOutput = &cvMatToOpOutput_;

    op::PoseExtractorCaffe poseExtractorCaffe_{netInputSize, netOutputSize, outputSize, FLAGS_num_scales, poseModel,
                                              FLAGS_model_folder, FLAGS_num_gpu_start};
    poseExtractorCaffe = &poseExtractorCaffe_;

    op::PoseRenderer poseRenderer_{netOutputSize, outputSize, poseModel, nullptr, !FLAGS_disable_blending, (float)FLAGS_alpha_pose};
    poseRenderer = &poseRenderer_;

    op::OpOutputToCvMat opOutputToCvMat_{outputSize};
    opOutputToCvMat = &opOutputToCvMat_;

    (*poseExtractorCaffe).initializationOnThread();
    (*poseRenderer).initializationOnThread();

    // Subscrive to input video feed and publish output video feed
    image_sub = n.subscribe("/usb_cam/image_raw", 1, &OpenposeDetecter::imageCallback, this);
    image_pub = n.advertise<sensor_msgs::Image>("/openpose/image_raw", 1);
    poses_pub = n.advertise<openpose_ros::PersonPoseArray>("/openpose/poses", 3);

    // continuing loop until receive Ctrl-C
    ros::spin();
  }

  ~OpenposeDetecter()
  {}

  void imageCallback(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        //ROS_INFO("imageCallBack");

        // Step 1 - 画像の読み出し
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        if(cv_ptr->image.empty()) {
            op::error("Could not open or find the image: " + FLAGS_image_path, 
                      __LINE__, __FUNCTION__, __FILE__);
            return;
        }
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Step 2 - cv::Matの入力画像からOpenPoseの画像形式に変換する
    op::Array<float> netInputArray;
    std::vector<float> scaleRatios;
    std::tie(netInputArray, scaleRatios) = (*cvMatToOpInput).format(cv_ptr->image);
    double scaleInputToOutput;
    op::Array<float> outputArray;
    std::tie(scaleInputToOutput, outputArray) = (*cvMatToOpOutput).format(cv_ptr->image);

    // Step 3 - poseの特徴点を演算する
    (*poseExtractorCaffe).forwardPass(netInputArray, {cv_ptr->image.cols, cv_ptr->image.rows}, scaleRatios);
    const auto poseKeypoints = (*poseExtractorCaffe).getPoseKeypoints();

    openpose_ros::PersonPoseArray personPoseArray;

    // 1つも検出していない場合は処理をスキップする
    if (poseKeypoints.empty()){
        personPoseArray.numPerson = 0;
    }else{       
        int numPerson = poseKeypoints.getSize()[0];
        int numOutput = poseKeypoints.getSize()[1];

        personPoseArray.numPerson = numPerson;

        for (int iPerson = 0; iPerson < numPerson; iPerson++) {
            openpose_ros::PersonPose personPose;
            for (int iOutput = 0; iOutput < numOutput; iOutput++) {
                std::string bodypart = op::POSE_COCO_BODY_PARTS.at(iOutput);
                int startIndex = 3 * (iPerson * numOutput + iOutput);

                if (bodypart == "Nose"){
                    personPose.Nose.x = poseKeypoints[startIndex];
                    personPose.Nose.y = poseKeypoints[startIndex + 1];
                    personPose.Nose.conf = poseKeypoints[startIndex + 2];
                }else if(bodypart == "Neck"){
                    personPose.Neck.x = poseKeypoints[startIndex];
                    personPose.Neck.y = poseKeypoints[startIndex + 1];
                    personPose.Neck.conf = poseKeypoints[startIndex + 2];
                }else if(bodypart == "RShoulder"){
                    personPose.RShoulder.x = poseKeypoints[startIndex];
                    personPose.RShoulder.y = poseKeypoints[startIndex + 1];
                    personPose.RShoulder.conf = poseKeypoints[startIndex + 2];
                }else if(bodypart == "RElbow"){
                    personPose.RElbow.x = poseKeypoints[startIndex];
                    personPose.RElbow.y = poseKeypoints[startIndex + 1];
                    personPose.RElbow.conf = poseKeypoints[startIndex + 2];
                }else if(bodypart == "RWrist"){
                    personPose.RWrist.x = poseKeypoints[startIndex];
                    personPose.RWrist.y = poseKeypoints[startIndex + 1];
                    personPose.RWrist.conf = poseKeypoints[startIndex + 2];
                }else if(bodypart == "LShoulder"){
                    personPose.LShoulder.x = poseKeypoints[startIndex];
                    personPose.LShoulder.y = poseKeypoints[startIndex + 1];
                    personPose.LShoulder.conf = poseKeypoints[startIndex + 2];
                }else if(bodypart == "LElbow"){
                    personPose.LElbow.x = poseKeypoints[startIndex];
                    personPose.LElbow.y = poseKeypoints[startIndex + 1];
                    personPose.LElbow.conf = poseKeypoints[startIndex + 2];
                }else if(bodypart == "LWrist"){
                    personPose.LWrist.x = poseKeypoints[startIndex];
                    personPose.LWrist.y = poseKeypoints[startIndex + 1];
                    personPose.LWrist.conf = poseKeypoints[startIndex + 2];
                }else if(bodypart == "RHip"){
                    personPose.RHip.x = poseKeypoints[startIndex];
                    personPose.RHip.y = poseKeypoints[startIndex + 1];
                    personPose.RHip.conf = poseKeypoints[startIndex + 2];
                }else if(bodypart == "RKnee"){
                    personPose.RKnee.x = poseKeypoints[startIndex];
                    personPose.RKnee.y = poseKeypoints[startIndex + 1];
                    personPose.RKnee.conf = poseKeypoints[startIndex + 2];
                }else if(bodypart == "RAnkle"){
                    personPose.RAnkle.x = poseKeypoints[startIndex];
                    personPose.RAnkle.y = poseKeypoints[startIndex + 1];
                    personPose.RAnkle.conf = poseKeypoints[startIndex + 2];
                }else if(bodypart == "LHip"){
                    personPose.LHip.x = poseKeypoints[startIndex];
                    personPose.LHip.y = poseKeypoints[startIndex + 1];
                    personPose.LHip.conf = poseKeypoints[startIndex + 2];
                }else if(bodypart == "LKnee"){
                    personPose.LKnee.x = poseKeypoints[startIndex];
                    personPose.LKnee.y = poseKeypoints[startIndex + 1];
                    personPose.LKnee.conf = poseKeypoints[startIndex + 2];
                }else if(bodypart == "LAnkle"){
                    personPose.LAnkle.x = poseKeypoints[startIndex];
                    personPose.LAnkle.y = poseKeypoints[startIndex + 1];
                    personPose.LAnkle.conf = poseKeypoints[startIndex + 2];
                }else if(bodypart == "REye"){
                    personPose.REye.x = poseKeypoints[startIndex];
                    personPose.REye.y = poseKeypoints[startIndex + 1];
                    personPose.REye.conf = poseKeypoints[startIndex + 2];
                }else if(bodypart == "LEye"){
                    personPose.LEye.x = poseKeypoints[startIndex];
                    personPose.LEye.y = poseKeypoints[startIndex + 1];
                    personPose.LEye.conf = poseKeypoints[startIndex + 2];
                }else if(bodypart == "REar"){
                    personPose.REar.x = poseKeypoints[startIndex];
                    personPose.REar.y = poseKeypoints[startIndex + 1];
                    personPose.REar.conf = poseKeypoints[startIndex + 2];
                }else if(bodypart == "LEar"){
                    personPose.LEar.x = poseKeypoints[startIndex];
                    personPose.LEar.y = poseKeypoints[startIndex + 1];
                    personPose.LEar.conf = poseKeypoints[startIndex + 2];
                }else{
                    ROS_ERROR("unknown bodypart : %s", bodypart.c_str());
                }
            }

            personPoseArray.personPoses.push_back(personPose);
        }

        // Step 4 - poseの特徴点を表示する。
        (*poseRenderer).renderPose(outputArray, poseKeypoints);

        // Step 5 - OpenPoseの画像形式からcv::Matの出力画像に変換する
        auto outputImage = (*opOutputToCvMat).formatToCvMat(outputArray);

        // Step 6
        cv_ptr->image = outputImage;  
    }

    // Output modified video stream
    image_pub.publish(cv_ptr->toImageMsg());
    poses_pub.publish(personPoseArray);
  }

  op::PoseModel gflagToPoseModel(const std::string& poseModeString)
  {
     op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
      if (poseModeString == "COCO")
          return op::PoseModel::COCO_18;
      else if (poseModeString == "MPI")
          return op::PoseModel::MPI_15;
      else if (poseModeString == "MPI_4_layers")
          return op::PoseModel::MPI_15_4;
      else
      {
          op::error("String does not correspond to any model (COCO, MPI, MPI_4_layers)", __LINE__, __FUNCTION__, __FILE__);
          return op::PoseModel::COCO_18;
      }
  }

  // Google flags into program variables
  std::tuple<op::Point<int>, op::Point<int>, op::Point<int>, op::PoseModel> gflagsToOpParameters()
  {
      op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
      // outputSize
      op::Point<int> outputSize;
      auto nRead = sscanf(FLAGS_resolution.c_str(), "%dx%d", &outputSize.x, &outputSize.y);
      op::checkE(nRead, 2, "Error, resolution format (" +  FLAGS_resolution + ") invalid, should be e.g., 960x540 ",
                 __LINE__, __FUNCTION__, __FILE__);
      // netInputSize
      op::Point<int> netInputSize;
      nRead = sscanf(FLAGS_net_resolution.c_str(), "%dx%d", &netInputSize.x, &netInputSize.y);
      op::checkE(nRead, 2, "Error, net resolution format (" +  FLAGS_net_resolution + ") invalid, should be e.g., 656x368 (multiples of 16)",
                 __LINE__, __FUNCTION__, __FILE__);
      // netOutputSize
      const auto netOutputSize = netInputSize;
      // poseModel
      const auto poseModel = gflagToPoseModel(FLAGS_model_pose);
      // Check no contradictory flags enabled
      if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
          op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
      if (FLAGS_scale_gap <= 0. && FLAGS_num_scales > 1)
          op::error("Incompatible flag configuration: scale_gap must be greater than 0 or num_scales = 1.", __LINE__, __FUNCTION__, __FILE__);
      // Logging and return result
      op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
      return std::make_tuple(outputSize, netInputSize, netOutputSize, poseModel);
  }

};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "openpose_ros");
  google::InitGoogleLogging("openpose_ros");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  OpenposeDetecter op_detecter;
  
  return 0;
}