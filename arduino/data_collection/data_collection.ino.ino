#include <Arduino_BMI270_BMM150.h>

// ======================================
// CHANGE THIS BEFORE EACH SESSION:
// "rest"   "shake"   "updown"   "circle"   "background"
// ======================================
String LABEL = "rest";

#define SAMPLE_RATE_MS  20
#define WINDOW_SAMPLES  50
#define NUM_WINDOWS     40

int sampleCount = 0;
int windowCount = 0;
bool done = false;

void setup() {
  Serial.begin(115200);
  while (!Serial);
  if (!IMU.begin()) {
    Serial.println("IMU failed!");
    while (1);
  }
  Serial.println("ax,ay,az,gx,gy,gz,label");
  delay(3000);
}

void loop() {
  if (done) return;
  float ax, ay, az, gx, gy, gz;
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);
    Serial.print(ax, 4); Serial.print(",");
    Serial.print(ay, 4); Serial.print(",");
    Serial.print(az, 4); Serial.print(",");
    Serial.print(gx, 4); Serial.print(",");
    Serial.print(gy, 4); Serial.print(",");
    Serial.print(gz, 4); Serial.print(",");
    Serial.println(LABEL);
    sampleCount++;
    if (sampleCount >= WINDOW_SAMPLES) {
      sampleCount = 0;
      windowCount++;
      delay(800);
      if (windowCount >= NUM_WINDOWS) {
        for (int i = 0; i < 10; i++) {
          Serial.println("# DONE");
          delay(300);
        }
        done = true;
      }
    }
    delay(SAMPLE_RATE_MS);
  }
}
