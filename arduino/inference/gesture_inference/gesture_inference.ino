#include <Arduino_BMI270_BMM150.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <math.h>
#include "gesture_model.h"

const int NUM_FEATURES = 60;
const int NUM_CLASSES = 5;
const char* CLASS_NAMES[] = {"background", "circle", "rest", "shake", "updown"};

const float SCALER_MEAN[60] = {
  -0.009563f, 0.159779f, 0.230537f, -0.339502f, 0.287114f, -0.113168f,
  0.262355f, 0.373151f, -0.616667f, 0.399613f, 0.885751f, 0.242848f,
  0.991056f, 0.430665f, 1.375752f, 0.209583f, 42.138461f, 43.047510f,
  -100.132157f, 84.731054f, 0.729481f, 39.285409f, 39.663778f, -76.074645f,
  79.888729f, 1.555488f, 48.934521f, 49.380999f, -87.182121f, 94.603568f,
  6.373422f, 10.866618f, 6.418313f, 4.136916f, 5.227263f, 11.072044f,
  15.389019f, 12.773253f, 30.066066f, 41.960031f, 27.028416f, 51.553695f,
  25.341868f, 13.924404f, 12.647660f, 1804114.455018f, 2150772.201762f,
  1209919.083383f, 1159776.145863f, 649020.228859f, 362377.629011f,
  1456988.598149f, 799754.964728f, 455167.308610f, 240735.072637f,
  434816.946683f, 719200.711352f, 506657.624367f, 1476682.224758f,
  1335001.268316f
};

const float SCALER_SCALE[60] = {
  0.185938f, 0.216152f, 0.231880f, 0.495617f, 0.498777f, 0.200654f,
  0.264235f, 0.229080f, 0.383230f, 0.713863f, 0.278328f, 0.268281f,
  0.103830f, 0.687261f, 0.517562f, 17.864315f, 70.719115f, 72.408097f,
  175.667856f, 148.545452f, 8.711717f, 41.314560f, 41.874180f, 83.978777f,
  95.027529f, 8.570967f, 48.350944f, 48.680582f, 93.850078f, 90.873937f,
  25.882939f, 15.797828f, 19.316430f, 13.174370f, 24.088894f, 35.828001f,
  32.321356f, 55.370041f, 80.600803f, 102.543851f, 98.230801f, 99.008802f,
  73.491833f, 51.636406f, 58.318857f, 10595048.083616f, 16307557.718056f,
  8861814.530257f, 8470458.609407f, 1802173.075458f, 1770137.455022f,
  2745742.991137f, 2630393.496718f, 2229117.451434f, 806388.360463f,
  2415867.555786f, 1398954.385643f, 1770767.179245f, 5283690.355789f,
  3324466.922670f
};

#define WINDOW_SIZE  50
#define SAMPLE_MS    20

const tflite::Model* tfl_model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
constexpr int kTensorArenaSize = 32 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

float buf_ax[WINDOW_SIZE], buf_ay[WINDOW_SIZE], buf_az[WINDOW_SIZE];
float buf_gx[WINDOW_SIZE], buf_gy[WINDOW_SIZE], buf_gz[WINDOW_SIZE];
int buf_idx = 0;

float safe_div(float a, float b) {
  if (b == 0.0f || isinf(b) || isnan(b)) return 0.0f;
  float r = a / b;
  return (isnan(r) || isinf(r)) ? 0.0f : r;
}

void extract_features(float* feat) {
  float* ch[6] = {buf_ax, buf_ay, buf_az, buf_gx, buf_gy, buf_gz};
  int fi = 0;

  for (int c = 0; c < 6; c++) {
    float mean = 0, rms = 0, mn = 1e9, mx = -1e9;
    for (int i = 0; i < WINDOW_SIZE; i++) {
      mean += ch[c][i];
      rms  += ch[c][i] * ch[c][i];
      if (ch[c][i] < mn) mn = ch[c][i];
      if (ch[c][i] > mx) mx = ch[c][i];
    }
    mean = safe_div(mean, WINDOW_SIZE);
    rms  = sqrt(safe_div(rms, WINDOW_SIZE));
    float std_dev = 0;
    for (int i = 0; i < WINDOW_SIZE; i++)
      std_dev += (ch[c][i] - mean) * (ch[c][i] - mean);
    std_dev = sqrt(safe_div(std_dev, WINDOW_SIZE));

    feat[fi++] = mean;
    feat[fi++] = std_dev;
    feat[fi++] = rms;
    feat[fi++] = mn;
    feat[fi++] = mx;
  }

  for (int c = 0; c < 6; c++) {
    for (int k = 1; k <= 5; k++) {
      float re = 0, im = 0;
      for (int n = 0; n < WINDOW_SIZE; n++) {
        float angle = 2.0f * PI * k * n / (float)WINDOW_SIZE;
        re += ch[c][n] * cos(angle);
        im -= ch[c][n] * sin(angle);
      }
      float psd = re * re + im * im;
      feat[fi++] = (isnan(psd) || isinf(psd)) ? 0.0f : psd;
    }
  }
}

void softmax(float* arr, int n) {
  float mx = arr[0];
  for (int i = 1; i < n; i++) if (arr[i] > mx) mx = arr[i];
  float sum = 0;
  for (int i = 0; i < n; i++) {
    arr[i] = exp(arr[i] - mx);
    sum += arr[i];
  }
  for (int i = 0; i < n; i++) arr[i] = safe_div(arr[i], sum);
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("IMU failed!");
    while (1);
  }
  Serial.println("IMU OK!");

  tfl_model = tflite::GetModel(model_tflite);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed! Increase tensor arena size.");
    while (1);
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Ready. Hold board in hand and perform a gesture.");
  Serial.println("Gestures: background / circle / rest / shake / updown");
}

void loop() {
  float ax, ay, az, gx, gy, gz;

  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);

    if (isnan(ax) || isnan(ay) || isnan(az) ||
        isnan(gx) || isnan(gy) || isnan(gz)) {
      delay(SAMPLE_MS);
      return;
    }

    buf_ax[buf_idx] = ax; buf_ay[buf_idx] = ay; buf_az[buf_idx] = az;
    buf_gx[buf_idx] = gx; buf_gy[buf_idx] = gy; buf_gz[buf_idx] = gz;
    buf_idx++;

    if (buf_idx >= WINDOW_SIZE) {
      buf_idx = 0;

      float features[NUM_FEATURES];
      extract_features(features);

      for (int i = 0; i < NUM_FEATURES; i++) {
        float val = safe_div(
          features[i] - SCALER_MEAN[i], SCALER_SCALE[i]);
        input->data.f[i] = (isnan(val) || isinf(val)) ? 0.0f : val;
      }

      if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke failed!");
        return;
      }

      float probs[NUM_CLASSES];
      for (int i = 0; i < NUM_CLASSES; i++)
        probs[i] = output->data.f[i];
      softmax(probs, NUM_CLASSES);

      int best = 0;
      float best_val = probs[0];
      for (int i = 1; i < NUM_CLASSES; i++) {
        if (probs[i] > best_val) {
          best_val = probs[i];
          best = i;
        }
      }

     Serial.print("Gesture: ");
Serial.print(CLASS_NAMES[best]);
Serial.print("  confidence: ");
Serial.print(best_val * 100, 1);
Serial.println("%");
    }
    delay(SAMPLE_MS);
  }
}
