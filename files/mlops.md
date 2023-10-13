### Q. Why we have to optimize and reduce model size and How to do it?
Optimization and model size reduction is required to make inferece faster on devices having less memory (like Edge devices (mobile phones), micro-controller etc.)
One such technique to achieve this is **Quantization** (Approximate floating point numbers with lower bits e.g. int8). By this technique we can reduce the memory footprint

We can do the below conversion for weight and baiases in order to reduce memory size
float 64(8 byte) --> float 16(2 byte)
Float (8 byte) --> int(1 byte)

There are two forms of quantization
- [Post-training quantization](https://www.tensorflow.org/model_optimization/guide/quantization/post_training)
- [Quantization aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training)
### <ins>Reference</ins>
- [Quantization Aware Training with TensorFlow Model Optimization Toolkit - Performance with Accuracy](https://blog.tensorflow.org/2020/04/quantization-aware-training-with-tensorflow-model-optimization-toolkit.html)
