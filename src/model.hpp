#ifndef KL_CHALLENGE_MODEL_H
#define KL_CHALLENGE_MODEL_H

#include "kl_error.hpp"

#include "../vendors/opencv2/core.hpp"

#include "../vendors/tflite/include/tensorflow/lite/model.h"
#include "../vendors/tflite/include/tensorflow/lite/interpreter.h"
#include "../vendors/tflite/include/tensorflow/lite/kernels/register.h"
#include "../vendors/tflite/include/tensorflow/lite/tools/gen_op_registration.h"
#include "../vendors/tflite/include/tensorflow/lite/optional_debug_tools.h"
class Model {

public:

    KLError init(const char *model_path);
    float inference(const char *img_path);

private:

    static void convert_image(const cv::Mat &src, float *dest);

    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
};

#endif //KL_CHALLENGE_MODEL_H
