#include "model.hpp"
#include <stdio.h>
#include "../vendors/opencv2/core.hpp"
#include "../vendors/opencv2/opencv.hpp"
#include "kl_error.hpp"


KLError Model::init(const char *model_path) {
//    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);   //CHANGED
//    tflite::ops::builtin::BuiltinOpResolver resolver;
//    tflite::InterpreterBuilder builder(*model, resolver);
//    std::unique_ptr<tflite::Interpreter> interpreter;
//    builder(&interpreter,-1);
//    interpreter->AllocateTensors();
    this->model_ = tflite::FlatBufferModel::BuildFromFile(model_path);
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*(this->model_), resolver);
    builder(&(this->interpreter_),-1);
    this->interpreter_->AllocateTensors();
    return KLError::NONE;
}

float Model::inference(const char *img_path) {
    //cv::Mat img = cv::imread(img_path);
    // Implementation missing...
    //float* dest;
    //float* input = interpreter->typed_input_tensor<float>(0);
    //convert_image(src, dest);
//    float* input = this->interpreter_->typed_input_tensor<float>(0);
//    this->interpreter_->Invoke();
//    float* output = this->interpreter_->typed_output_tensor<float>(0);

    float* input = this->interpreter_->typed_input_tensor<float>(0);
    for(int i = 0; i < 3*80*80; i++)
        input[i] = 0;
    this->interpreter_->Invoke();
    float* output = this->interpreter_->typed_output_tensor<float>(0);
    printf("Result is: %f\n", *output);
    return 0.0;
}

void Model::convert_image(const cv::Mat &src, float *dest) {
    // Implementation missing...
}
