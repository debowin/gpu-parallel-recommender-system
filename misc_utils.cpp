#include "misc_utls.h"

void startTime(Timer *timer) {
    gettimeofday(&(timer->startTime), nullptr);
}

void stopTime(Timer *timer) {
    gettimeofday(&(timer->endTime), nullptr);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
 + (timer.endTime.tv_usec - timer.startTime.tv_usec) / 1.0e6));
}
